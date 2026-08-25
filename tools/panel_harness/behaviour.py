"""Behavioural checks for the panel: what it remembers, and for which workflow.

Layout is checked by check.py. This covers the state rules that are easy to get
wrong and impossible to see in a screenshot: that a model stays pinnable once it
is on disk, that one workflow's download queue does not show up under another,
and that each workflow keeps its own resolutions.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from check import serve  # noqa: E402


def mount(page, base, *, workflow="workflows/one.json", items=8, jobs=0):
    page.add_init_script(f"window.__workflow = {workflow!r};")
    page.goto(f"{base}/index.html?items={items}&jobs={jobs}")
    page.wait_for_function("window.__ready === true")


DOWNLOAD_BUTTON = ".wmd-grow .wmd-primary"


def scan(page):
    page.click("text=Scan workflow")
    page.wait_for_timeout(500)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright && playwright install chromium")
        return 2

    failures: list[str] = []
    total = 0

    def check(name, condition, detail=""):
        nonlocal total
        total += 1
        print(f"{'ok  ' if condition else 'FAIL'} {name}" + (f"  -- {detail}" if not condition and detail else ""))
        if not condition:
            failures.append(name)

    with serve() as base, sync_playwright() as p:
        launch = {}
        bundled = os.environ.get("PLAYWRIGHT_CHROMIUM", "/opt/pw-browsers/chromium")
        if os.path.exists(bundled):
            launch["executable_path"] = bundled
        browser = p.chromium.launch(**launch)
        context = browser.new_context()

        # 1. A model already on disk is still listed, and still pinnable.
        page = context.new_page()
        mount(page, base)
        page.evaluate("window.__allPresent = true")
        scan(page)
        listed = page.eval_on_selector_all(".wmd-item", "els => els.length")
        pin_disabled = page.eval_on_selector("text=Save to workflow", "el => el.disabled")
        check("models on disk stay listed", listed > 0, f"{listed} rows")
        check("and stay pinnable", pin_disabled is False)

        # 2. Downloading, then re-scanning with nothing missing, keeps them pinnable.
        page.evaluate("window.__allPresent = false")
        scan(page)
        page.click(DOWNLOAD_BUTTON)
        page.wait_for_timeout(300)
        page.evaluate("window.__noneMissing = true")   # nothing left to find
        scan(page)
        after = page.eval_on_selector_all(".wmd-item", "els => els.length")
        pin_disabled = page.eval_on_selector("text=Save to workflow", "el => el.disabled")
        check("a re-scan that finds nothing keeps the earlier resolutions", after > 0, f"{after} rows")
        check("pinning is still possible after downloading", pin_disabled is False)

        # 3. Pinning writes the manifest, and re-scanning recovers it from the node.
        if pin_disabled:
            # Clicking a disabled button just hangs; report the real problem instead.
            check("pin writes every resolved model", False, "the pin button is disabled")
        else:
            page.click("text=Save to workflow")
            page.wait_for_timeout(300)
            pinned = page.evaluate("window.__pinned?.length ?? 0")
            check("pin writes every resolved model", pinned > 0, f"{pinned} entries")

        # 4. Only what the graph asks for is ticked, and the size is stated.
        gate = context.new_page()
        mount(gate, base, workflow="workflows/gate.json", items=12, jobs=0)
        scan(gate)
        counts = gate.evaluate("""() => {
          const rows = [...document.querySelectorAll('.wmd-item')];
          return {
            rows: rows.length,
            ticked: rows.filter(r => r.querySelector('input[type=checkbox]')?.checked).length,
            optional: rows.filter(r => r.textContent.includes('not used here')).length,
            button: document.querySelector('.wmd-grow .wmd-primary')?.textContent ?? '',
          };
        }""")
        check("links the graph does not use are listed", counts["optional"] > 0,
              f"{counts['optional']} of {counts['rows']}")
        check("but are not ticked for download", counts["ticked"] + counts["optional"] <= counts["rows"]
              and counts["ticked"] < counts["rows"], f"{counts['ticked']} ticked of {counts['rows']}")
        check("the download button states the size", "(" in counts["button"], counts["button"])
        gate.close()

        # 5. Pushed progress for another workflow's download must not leak in.
        leak = context.new_page()
        mount(leak, base, workflow="workflows/mine.json", items=4, jobs=0)
        leak.wait_for_timeout(400)

        def queue_text():
            return leak.eval_on_selector(".wmd-jobs", "n => n.textContent")

        leak.evaluate("""() => window.__emit('wmd.progress', {
          id: 'other-1', filename: 'theirs.safetensors', folder: 'loras',
          status: 'downloading', downloaded: 10, total: 100, speed: 1,
          workflow: 'workflows/theirs.json',
        })""")
        leak.wait_for_timeout(200)
        check("progress from another workflow is ignored", "theirs.safetensors" not in queue_text())

        leak.evaluate("""() => window.__emit('wmd.progress', {
          id: 'mine-1', filename: 'mine.safetensors', folder: 'loras',
          status: 'downloading', downloaded: 10, total: 100, speed: 1,
          workflow: 'workflows/mine.json',
        })""")
        leak.wait_for_timeout(200)
        check("progress for this workflow is shown", "mine.safetensors" in queue_text())

        # A prompt-driven download carries no workflow and belongs to all of them.
        leak.evaluate("""() => window.__emit('wmd.progress', {
          id: 'node-1', filename: 'queued.safetensors', folder: 'loras',
          status: 'downloading', downloaded: 10, total: 100, speed: 1, workflow: '',
        })""")
        leak.wait_for_timeout(200)
        check("a download started by a queued prompt is still shown",
              "queued.safetensors" in queue_text())
        leak.close()

        # 6. A second workflow starts clean and does not inherit the first's queue.
        page.evaluate("window.__noneMissing = false")
        second = context.new_page()
        mount(second, base, workflow="workflows/two.json", items=5, jobs=0)
        second.wait_for_timeout(2500)   # let the panel notice the workflow
        rows = second.eval_on_selector_all(".wmd-item", "els => els.length")
        queue = second.evaluate(
            "() => [...document.querySelectorAll('.wmd-jobs')].map(n => n.textContent).join('')"
        )
        check("a different workflow starts with no resolutions", rows == 0, f"{rows} rows")
        check("and with no queue from the previous one", "No downloads yet." in queue)

        # 7. Reopening the first workflow restores what it had.
        third = context.new_page()
        mount(third, base, workflow="workflows/one.json", items=8, jobs=0)
        third.wait_for_timeout(600)
        restored = third.eval_on_selector_all(".wmd-item", "els => els.length")
        check("reopening a workflow restores its resolutions", restored > 0, f"{restored} rows")

        # 8. The clear button is offered only when something can be cleared.
        fourth = context.new_page()
        mount(fourth, base, workflow="workflows/three.json", items=4, jobs=0)
        fourth.wait_for_timeout(400)
        idle = fourth.eval_on_selector("text=clear finished", "el => el.disabled")
        check("clear finished is disabled with an empty queue", idle is True)

        browser.close()

    print(f"\n{total - len(failures)}/{total} behaviours correct")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
