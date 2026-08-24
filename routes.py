"""HTTP surface for the sidebar panel, plus the queue-time download hook.

Everything intelligent lives here rather than in the node: the panel scans,
resolves, searches and overrides, and the only thing that ends up in the workflow
is the manifest it produces.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .wmd import (
    comfy_env,
    config,
    http,
    jobs,
    resolve,
    rules,
    scan,
    sources,
)
from .wmd import manifest as manifest_mod
from .wmd.errors import WmdError
from .wmd.models import Slot

log = logging.getLogger("working_model_downloader")

PREFIX = "/working_model_downloader"
NODE_TYPE = "WMD_ModelDownloader"
PROGRESS_EVENT = "wmd.progress"


def _prompt_server():
    try:
        from server import PromptServer
    except Exception:
        return None
    return getattr(PromptServer, "instance", None)


def _json(data: Any, status: int = 200):
    from aiohttp import web

    return web.json_response(data, status=status)


def _error(exc: Exception, status: int = 400):
    return _json({"error": config.redact(str(exc))}, status=status)


# -- request handling ---------------------------------------------------------


def _refs_from_request(body: dict[str, Any]) -> tuple[list, list]:
    """Build the reference list and slot list a resolve/scan request implies.

    Present models are included by default. Once a download finishes its slot is
    no longer missing, and dropping it here would mean the panel forgets a model
    the moment it arrives -- you could no longer pin it into the workflow, and any
    link still listed would lose the loader's own folder and fall back to guessing
    from its filename.
    """
    refs = []
    slots = []
    if body.get("workflow") is not None or body.get("prompt"):
        result = scan.scan(body.get("workflow"), body.get("prompt"))
        refs.extend(result.documented)
        slots = list(result.missing)
        if body.get("include_present", True):
            slots.extend(result.present)
    if body.get("urls"):
        text = body["urls"]
        refs.extend(sources.refs_from_input(text if isinstance(text, str) else "\n".join(text)))
    return sources.dedupe(refs), slots


def _apply_overrides(refs: list, overrides: list[dict[str, Any]]) -> None:
    """Fold the panel's per-item folder and filename choices back into the refs."""
    by_url = {ref.raw: ref for ref in refs}
    for override in overrides or []:
        ref = by_url.get(override.get("source_url") or override.get("url"))
        if ref is None:
            continue
        if override.get("folder"):
            ref.folder_override = str(override["folder"])
        if override.get("filename"):
            ref.filename_override = str(override["filename"])


async def handle_get_config(_request):
    return _json(config.masked())


async def handle_post_config(request):
    body = await request.json()
    updates: dict[str, Any] = {}
    for key in ("hf_token", "civitai_api_key"):
        if key in body:
            updates[key] = body[key]
    if isinstance(body.get("prefs"), dict):
        updates["prefs"] = body["prefs"]
    return _json(config.masked(config.save(updates)))


async def handle_folders(_request):
    return _json(
        {
            "folders": [
                {"key": key, "paths": comfy_env.folder_paths_for(key)}
                for key in comfy_env.folder_keys()
            ]
        }
    )


async def handle_scan(request):
    body = await request.json()
    result = await asyncio.to_thread(scan.scan, body.get("workflow"), body.get("prompt"))
    return _json(result.to_json())


async def handle_resolve(request):
    body = await request.json()

    def work():
        refs, slots = _refs_from_request(body)
        _apply_overrides(refs, body.get("overrides") or [])
        return resolve.resolve_refs(
            refs,
            cfg=config.load(),
            session=http.new_session(),
            slots=slots,
            allow_search=bool(body.get("allow_search")),
        )

    try:
        resolutions = await asyncio.to_thread(work)
    except WmdError as exc:
        return _error(exc)
    return _json({"items": [resolve.resolution_json(item) for item in resolutions]})


async def handle_search(request):
    body = await request.json()
    filename = str(body.get("filename") or "").strip()
    if not filename:
        return _json({"error": "filename is required"}, status=400)

    folder = str(body.get("folder") or "") or None
    slot = Slot(node_id="", input_name="", value=filename, folder=folder)

    def work():
        return resolve.search_for_slots([slot], cfg=config.load(), session=http.new_session())

    try:
        resolutions = await asyncio.to_thread(work)
    except WmdError as exc:
        return _error(exc)
    return _json({"items": [resolve.resolution_json(item) for item in resolutions]})


async def handle_download(request):
    body = await request.json()
    try:
        entries = manifest_mod.parse({"version": manifest_mod.VERSION, "entries": body.get("items") or []})
    except WmdError as exc:
        return _error(exc)

    workflow = str(body.get("workflow_key") or "")
    manager = jobs.manager()
    started = []
    for entry in entries:
        try:
            started.append(
                manager.submit_entry(entry, source="panel", workflow=workflow).to_json()
            )
        except Exception as exc:  # noqa: BLE001 - reported per item
            started.append({"filename": entry.filename, "status": jobs.FAILED, "error": str(exc)})
    return _json({"jobs": started})


async def handle_jobs(request):
    """The queue, scoped to one workflow unless asked for everything."""
    workflow = request.query.get("workflow")
    if request.query.get("all"):
        workflow = None
    return _json({"jobs": [job.to_json() for job in jobs.manager().list(workflow)]})


async def handle_job_action(request):
    job_id = request.match_info["job_id"]
    action = request.match_info["action"]
    manager = jobs.manager()
    handlers = {"cancel": manager.cancel, "pause": manager.pause, "resume": manager.resume}
    handler = handlers.get(action)
    if handler is None:
        return _json({"error": f"unknown action {action}"}, status=400)
    result = handler(job_id)
    return _json({"ok": bool(result)})


async def handle_clear_jobs(request):
    body = await request.json() if request.can_read_body else {}
    workflow = body.get("workflow_key")
    if body.get("all"):
        workflow = None
    return _json({"cleared": jobs.manager().clear_finished(workflow)})


async def handle_manifest(request):
    """Turn the panel's current selection into the JSON the node carries."""
    body = await request.json()
    try:
        entries = manifest_mod.parse({"version": manifest_mod.VERSION, "entries": body.get("items") or []})
    except WmdError as exc:
        return _error(exc)
    document = {"version": manifest_mod.VERSION, "entries": [entry.to_json() for entry in entries]}
    return _json({"manifest": manifest_mod.dumps(document), "node_type": NODE_TYPE})


async def handle_get_rules(_request):
    return _json({"rules": rules.to_json()})


async def handle_post_rules(request):
    body = await request.json()
    pattern = str(body.get("pattern") or "").strip()
    if not pattern:
        return _json({"error": "pattern is required"}, status=400)
    rules.remember(
        rules.Rule(
            pattern=pattern,
            field=str(body.get("field") or "filename"),
            folder=body.get("folder") or None,
            url=body.get("url") or None,
            regex=bool(body.get("regex")),
        )
    )
    return _json({"rules": rules.to_json()})


ROUTES = (
    ("GET", "/config", handle_get_config),
    ("POST", "/config", handle_post_config),
    ("GET", "/folders", handle_folders),
    ("POST", "/scan", handle_scan),
    ("POST", "/resolve", handle_resolve),
    ("POST", "/search", handle_search),
    ("POST", "/download", handle_download),
    ("GET", "/jobs", handle_jobs),
    ("POST", "/jobs/clear", handle_clear_jobs),
    ("POST", "/jobs/{job_id}/{action}", handle_job_action),
    ("POST", "/manifest", handle_manifest),
    ("GET", "/rules", handle_get_rules),
    ("POST", "/rules", handle_post_rules),
)


# -- queue-time hook ----------------------------------------------------------


def on_prompt(json_data: dict[str, Any]) -> dict[str, Any]:
    """Start a queued prompt's pinned downloads as early as we possibly can.

    This runs inside ``POST /prompt``, before validation. Starting here (and
    waiting in the node's VALIDATE_INPUTS) means the transfer overlaps with
    validation instead of following it, and nothing is left to chance about
    whether a loader runs first.
    """
    try:
        prompt = json_data.get("prompt")
        if not isinstance(prompt, dict):
            return json_data
        from .wmd_nodes import start_downloads

        for node in prompt.values():
            if not isinstance(node, dict) or node.get("class_type") != NODE_TYPE:
                continue
            inputs = node.get("inputs") or {}
            if inputs.get("enforce") != "before_execution":
                continue
            start_downloads(str(inputs.get("manifest") or ""), source="prompt")
    except Exception:  # noqa: BLE001 - a hook must never break queueing
        log.exception("Working Model Downloader: on_prompt hook failed")
    return json_data


def _push_progress(job: jobs.Job) -> None:
    server = _prompt_server()
    if server is None:
        return
    try:
        server.send_sync(PROGRESS_EVENT, job.to_json())
    except Exception:  # noqa: BLE001 - the UI is best-effort
        pass


def register() -> bool:
    """Attach routes, the prompt hook and progress pushing to a live ComfyUI."""
    server = _prompt_server()
    if server is None:
        log.info("Working Model Downloader: no PromptServer; panel routes not registered")
        return False

    for method, path, handler in ROUTES:
        server.routes.route(method, PREFIX + path)(handler)

    if hasattr(server, "add_on_prompt_handler"):
        server.add_on_prompt_handler(on_prompt)
    else:
        log.warning(
            "Working Model Downloader: this ComfyUI has no on_prompt hook; "
            "downloads will start when the node is validated instead"
        )

    jobs.manager().subscribe(_push_progress)
    return True
