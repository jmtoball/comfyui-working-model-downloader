"""Command line access to the whole pipeline, without ComfyUI.

``python -m wmd.cli scan workflow.json`` is the fastest way to see what the
scanner makes of a workflow, and ``resolve`` shows which heuristic picked each
destination -- the same output the sidebar renders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import config, http, manifest, resolve, scan, sources
from .models import Resolution


def _human(size: int | None) -> str:
    if not size:
        return "?"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


def _load_workflow(path: str) -> tuple[Any, dict[str, Any] | None]:
    """Accept a UI workflow, an API prompt, or a file containing both."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and ("workflow" in data or "prompt" in data):
        return data.get("workflow"), data.get("prompt")
    if isinstance(data, dict) and "nodes" in data:
        return data, None
    return None, data if isinstance(data, dict) else None


def _print_resolutions(resolutions: list[Resolution]) -> None:
    for resolution in resolutions:
        target = f"{resolution.folder or '?'}/{resolution.filename or resolution.ref.raw}"
        state = "on disk" if resolution.existing_path else resolution.tier
        print(f"  [{state:11}] {target}")
        print(f"                {_human(resolution.file.size if resolution.file else None)}  {resolution.reason}")
        if resolution.error:
            print(f"                error: {resolution.error}")
        for candidate in resolution.candidates:
            print(f"                candidate: {candidate.provider}  {candidate.url}")


def cmd_scan(args: argparse.Namespace) -> int:
    workflow, prompt = _load_workflow(args.workflow)
    result = scan.scan(workflow, prompt)
    if args.json:
        print(json.dumps(result.to_json(), indent=2))
        return 0
    print(f"Documented links ({len(result.documented)}):")
    for ref in result.documented:
        names = f"  ~ {', '.join(ref.nearby_names)}" if ref.nearby_names else ""
        print(f"  {ref.provider:12} {ref.raw}{names}")
    print(f"Missing model slots ({len(result.missing)}):")
    for slot in result.missing:
        print(f"  {slot.folder}/{slot.filename}  <- {slot.node_type}.{slot.input_name}")
    if result.not_connected:
        print(f"Missing but not connected to an output ({len(result.not_connected)}):")
        for slot in result.not_connected:
            print(f"  {slot.folder}/{slot.filename}  <- {slot.node_type}.{slot.input_name}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    cfg = config.load()
    session = http.new_session()
    refs = sources.refs_from_input("\n".join(args.urls))
    slots = []
    if args.workflow:
        workflow, prompt = _load_workflow(args.workflow)
        result = scan.scan(workflow, prompt)
        refs = sources.dedupe([*result.documented, *refs])
        slots = result.missing
    resolutions = resolve.resolve_refs(
        refs, cfg=cfg, session=session, slots=slots, allow_search=args.search
    )
    if args.json:
        print(json.dumps([resolve.resolution_json(r) for r in resolutions], indent=2))
        return 0
    print(f"Resolved {len(resolutions)} item(s):")
    _print_resolutions(resolutions)
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    cfg = config.load()
    session = http.new_session()
    refs = sources.refs_from_input("\n".join(args.urls))
    resolutions = resolve.resolve_refs(refs, cfg=cfg, session=session)
    document = manifest.build(resolutions)
    unresolved = [r for r in resolutions if not r.resolved]
    for item in unresolved:
        print(f"unresolved: {item.ref.raw} -- {item.error or item.reason}", file=sys.stderr)
    if args.dry_run:
        print(manifest.dumps(document))
        return 1 if unresolved else 0
    return _apply(manifest.parse(document), cfg, session)


def cmd_apply(args: argparse.Namespace) -> int:
    with open(args.manifest, encoding="utf-8") as handle:
        entries = manifest.parse(handle.read())
    return _apply(entries, config.load(), http.new_session())


def _apply(entries: list[Any], cfg: config.Config, session: Any) -> int:
    state = {"name": ""}

    def on_entry(entry: Any) -> None:
        state["name"] = entry.filename
        print(f"-> {entry.folder}/{entry.filename}", file=sys.stderr)

    def on_progress(done: int, total: int | None) -> None:
        share = f"{done * 100 / total:5.1f}%" if total else _human(done)
        print(f"\r   {state['name']}  {share}", end="", file=sys.stderr)

    report = manifest.apply(
        entries, cfg=cfg, session=session, progress=on_progress, on_entry=on_entry
    )
    print("", file=sys.stderr)
    print(report.summary())
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wmd", description="Working Model Downloader (ComfyUI-free CLI)"
    )
    parser.add_argument(
        "--models-dir",
        help="where models/ lives when ComfyUI is not importable (default $WMD_MODELS_DIR)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="show what a workflow says about its models")
    p_scan.add_argument("workflow")
    p_scan.add_argument("--json", action="store_true")
    p_scan.set_defaults(func=cmd_scan)

    p_resolve = sub.add_parser("resolve", help="resolve URLs (and optionally a workflow)")
    p_resolve.add_argument("urls", nargs="*")
    p_resolve.add_argument("--workflow")
    p_resolve.add_argument("--search", action="store_true", help="allow the search fallback")
    p_resolve.add_argument("--json", action="store_true")
    p_resolve.set_defaults(func=cmd_resolve)

    p_download = sub.add_parser("download", help="resolve and download URLs")
    p_download.add_argument("urls", nargs="+")
    p_download.add_argument("--dry-run", action="store_true")
    p_download.set_defaults(func=cmd_download)

    p_apply = sub.add_parser("apply", help="download everything pinned in a manifest")
    p_apply.add_argument("manifest")
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args(argv)
    if args.models_dir:
        os.environ["WMD_MODELS_DIR"] = args.models_dir
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
