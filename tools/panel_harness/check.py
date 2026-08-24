"""Mount the sidebar panel in a headless browser and check its layout.

The panel is the one part of this package that unit tests cannot reach, and
layout bugs there are invisible until someone opens the sidebar with a long list
and a running queue. This mounts `web/panel.js` against stubbed `app`/`api`
modules -- the same relative specifiers ComfyUI serves them under -- fills it with
fixture data, and asserts that no section spills its content over the next one.

That spill is the specific failure this exists to catch: a flex item shrunk below
its content, painting over the section after it.

    pip install playwright && playwright install chromium
    python tools/panel_harness/check.py            # pass/fail
    python tools/panel_harness/check.py --shots out/   # also write screenshots
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import os
import shutil
import socketserver
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
WEB = os.path.join(REPO, "web")

# width, height, resolved items, queued jobs, open the settings section
SCENARIOS = [
    (520, 900, 14, 0, False, "idle"),
    (520, 900, 14, 8, False, "downloading"),
    (520, 900, 40, 12, False, "heavy"),
    (320, 700, 25, 6, False, "narrow"),
    (480, 520, 25, 6, False, "short"),
    (520, 900, 25, 6, True, "settings-open"),
    (520, 560, 25, 6, True, "settings-open-short"),
]


@contextlib.contextmanager
def serve():
    """Serve the harness with panel.js at the path ComfyUI serves it from."""
    root = tempfile.mkdtemp(prefix="wmd-harness-")
    try:
        shutil.copytree(os.path.join(HERE, "scripts"), os.path.join(root, "scripts"))
        target = os.path.join(root, "extensions", "wmd")
        os.makedirs(target)
        for name in ("panel.js", "panel.css"):
            shutil.copy(os.path.join(WEB, name), os.path.join(target, name))
        shutil.copy(os.path.join(HERE, "index.html"), os.path.join(root, "index.html"))

        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args):  # noqa: A003 - silence per-request logging
                pass

        handler = functools.partial(Quiet, directory=root)
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("127.0.0.1", 0), handler) as httpd:
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{httpd.server_address[1]}"
            finally:
                httpd.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)


MEASURE = """() => {
  const panel = document.querySelector('.wmd-panel');
  const box = panel.getBoundingClientRect();
  return [...panel.children].map((child) => {
    const rect = child.getBoundingClientRect();
    const label = child.querySelector('h4,summary')?.textContent || child.className;
    return {
      label: label.trim().slice(0, 24),
      // Content taller than its box only harms anything when the box lets it
      // show: that is what paints one section across the next. A region with its
      // own scrollbar is the intended design, not a fault.
      spill: getComputedStyle(child).overflowY !== 'visible'
        ? 0
        : child.scrollHeight - child.clientHeight,
      past: Math.round(rect.bottom - box.bottom),
      top: Math.round(rect.top),
      bottom: Math.round(rect.bottom),
    };
  });
}"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", help="directory to write screenshots into")
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright && playwright install chromium")
        return 2

    if args.shots:
        os.makedirs(args.shots, exist_ok=True)

    failures = 0
    with serve() as base, sync_playwright() as p:
        launch = {}
        bundled = os.environ.get("PLAYWRIGHT_CHROMIUM", "/opt/pw-browsers/chromium")
        if os.path.exists(bundled):
            launch["executable_path"] = bundled
        browser = p.chromium.launch(**launch)

        for width, height, items, jobs, settings, label in SCENARIOS:
            page = browser.new_page(viewport={"width": width + 40, "height": height})
            page.goto(f"{base}/index.html?items={items}&jobs={jobs}")
            page.wait_for_function("window.__ready === true")
            page.eval_on_selector("#sidebar", f"e => e.style.width = '{width}px'")
            page.click("text=Scan workflow")
            page.wait_for_timeout(700)
            if settings:
                page.click("summary")
                page.wait_for_timeout(300)
            if args.shots:
                page.screenshot(path=os.path.join(args.shots, f"{label}.png"))

            sections = page.evaluate(MEASURE)
            problems = [s for s in sections if s["spill"] > 0 or s["past"] > 1]
            overlaps = [
                (a, b) for a, b in zip(sections, sections[1:], strict=False)
                if b["top"] < a["bottom"]
            ]
            status = "ok " if not problems and not overlaps else "FAIL"
            print(f"{status} {label:20} {width}x{height}  items={items:3} jobs={jobs:3}")
            for section in problems:
                print(
                    f"       {section['label']:24} spills {section['spill']}px, "
                    f"{section['past']}px past the panel"
                )
            for a, b in overlaps:
                print(f"       {a['label']!r} overlaps {b['label']!r}")
            failures += bool(problems or overlaps)
            page.close()

        browser.close()

    print(f"\n{len(SCENARIOS) - failures}/{len(SCENARIOS)} layouts clean")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
