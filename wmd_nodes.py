"""The one node: it carries the panel's config and executes it.

Scanning, resolving, searching and overriding all happen in the sidebar panel.
What lands in the workflow is the *result* of that: a pinned manifest. The node's
whole job is to make that manifest true before the workflow runs, which is what
makes a headless run reproduce what you sorted out interactively.

Ordering is the subtle part. ComfyUI gives an OUTPUT_NODE no ordering guarantee
against a loader, so "download in my execute()" is not enough -- a CheckpointLoader
may already have run and failed. Two hooks fix it properly:

* ``PromptServer.add_on_prompt_handler`` fires inside ``POST /prompt``, before
  validation and queueing. We start the downloads there, without blocking.
* ``VALIDATE_INPUTS`` is awaited for every node before *any* node executes. We wait
  there for those downloads to finish, and turn a failure into a validation error,
  so the prompt is rejected with a clear message instead of a loader exploding on a
  file that is not there.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import threading
import time

from .wmd import comfy_env, config, jobs
from .wmd import manifest as manifest_mod
from .wmd.errors import WmdError

log = logging.getLogger("working_model_downloader")

EMPTY_MANIFEST = '{\n  "version": 1,\n  "entries": []\n}'

BEFORE_EXECUTION = "before_execution"
ON_NODE_EXECUTION = "on_node_execution"

# Jobs started by the on_prompt handler, keyed by the manifest they came from, so
# VALIDATE_INPUTS can find and wait for the work its own queueing kicked off.
_started: dict[str, list[str]] = {}
_started_lock = threading.Lock()


class AnyType(str):
    """A type that matches anything, for the ordering passthrough."""

    def __ne__(self, _other: object) -> bool:
        return False


ANY = AnyType("*")


def manifest_digest(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def supports_async_validation() -> bool:
    """Whether this ComfyUI awaits a coroutine returned from a node's methods.

    When it does, waiting in VALIDATE_INPUTS leaves the event loop free to keep
    serving progress updates. When it does not, we still wait -- correctness first
    -- but the server is blocked for the duration, so it is worth knowing which.
    """
    try:
        import execution
    except Exception:
        return False
    return inspect.iscoroutinefunction(getattr(execution, "validate_inputs", None))


def start_downloads(manifest_text: str, *, source: str = "prompt") -> list[str]:
    """Queue everything a manifest pins that is not already on disk."""
    try:
        entries = manifest_mod.parse(manifest_text)
    except WmdError as exc:
        log.warning("Working Model Downloader: %s", exc)
        return []

    manager = jobs.manager()
    job_ids: list[str] = []
    for entry in entries:
        if manifest_mod.existing_path(entry):
            continue
        try:
            job_ids.append(manager.submit_entry(entry, source=source).id)
        except Exception as exc:  # noqa: BLE001 - reported through the node's report
            log.warning("Working Model Downloader: could not queue %s: %s", entry.filename, exc)

    with _started_lock:
        _started[manifest_digest(manifest_text)] = job_ids
    return job_ids


def _job_ids_for(manifest_text: str) -> list[str]:
    """Jobs already started for this manifest, if they are all still known.

    Job records are cleared from the manager (by the panel, or by a restart), so a
    remembered id that no longer resolves means the work is gone, not done. Treat
    that as "nothing started" and queue it again rather than waiting on ghosts.
    """
    with _started_lock:
        job_ids = _started.get(manifest_digest(manifest_text), [])
    if not job_ids:
        return []
    manager = jobs.manager()
    return job_ids if all(manager.get(job_id) is not None for job_id in job_ids) else []


def _outstanding(job_ids: list[str]) -> list[jobs.Job]:
    manager = jobs.manager()
    found = [manager.get(job_id) for job_id in job_ids]
    return [job for job in found if job is not None and not job.terminal]


def _failures(job_ids: list[str]) -> list[jobs.Job]:
    manager = jobs.manager()
    found = [manager.get(job_id) for job_id in job_ids]
    return [job for job in found if job is not None and job.status == jobs.FAILED]


def _validation_error(failed: list[jobs.Job]) -> str:
    lines = [f"{job.folder}/{job.filename}: {job.error}" for job in failed]
    return (
        "Working Model Downloader could not fetch "
        f"{len(failed)} pinned model(s):\n  " + "\n  ".join(lines)
    )


def _deadline(timeout: int) -> float | None:
    return time.monotonic() + timeout if timeout > 0 else None


class WMDModelDownloader:
    """Working Model Downloader -- carries the panel's pinned config."""

    CATEGORY = "model_downloader"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", ANY)
    RETURN_NAMES = ("report", "passthrough")
    FUNCTION = "run"
    DESCRIPTION = (
        "Downloads the models pinned by the Model Downloader sidebar panel, before "
        "the rest of the workflow runs. Build the manifest in the panel, then save "
        "the workflow: headless and API runs reproduce it exactly."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": EMPTY_MANIFEST,
                        "tooltip": "Written by the Model Downloader sidebar panel. "
                        "Hand-editable JSON: url, filename, folder, sha256.",
                    },
                ),
                "enforce": (
                    [BEFORE_EXECUTION, ON_NODE_EXECUTION],
                    {
                        "default": BEFORE_EXECUTION,
                        "tooltip": "before_execution waits at queue time, so loaders "
                        "cannot run before the files exist. on_node_execution "
                        "downloads when this node runs, which shows a progress bar "
                        "but only orders nodes downstream of the passthrough.",
                    },
                ),
                "on_failure": (
                    ["error", "warn"],
                    {"default": "error", "tooltip": "Whether a failed download stops the run."},
                ),
                "verify_hash": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Check the pinned sha256 before the file is moved into place."},
                ),
                "queue_timeout": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 86400,
                        "tooltip": "Seconds to wait at queue time; 0 waits as long as it takes.",
                    },
                ),
            },
            "optional": {"passthrough": (ANY, {})},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    # -- queue-time gate --------------------------------------------------

    @classmethod
    def _validate(cls, manifest_text: str, enforce: str, on_failure: str, timeout: int):
        """Shared body of VALIDATE_INPUTS; returns (done, result_or_None)."""
        try:
            entries = manifest_mod.parse(manifest_text)
        except WmdError as exc:
            return True, f"Working Model Downloader: {exc}"

        if not entries or enforce != BEFORE_EXECUTION:
            return True, True

        job_ids = _job_ids_for(manifest_text)
        if not job_ids:
            # No on_prompt handler ran (an older ComfyUI, or a direct API caller).
            job_ids = start_downloads(manifest_text, source="prompt")
        return False, job_ids

    @classmethod
    def _finish(cls, job_ids: list[str], on_failure: str):
        failed = _failures(job_ids)
        if failed and on_failure == "error":
            return _validation_error(failed)
        if failed:
            log.warning("%s", _validation_error(failed))
        return True

    def run(
        self,
        manifest: str = EMPTY_MANIFEST,  # noqa: A002 - the widget name is the public API
        enforce: str = BEFORE_EXECUTION,
        on_failure: str = "error",
        verify_hash: bool = True,
        queue_timeout: int = 0,
        passthrough=None,
        unique_id=None,
    ):
        try:
            entries = manifest_mod.parse(manifest)
        except WmdError as exc:
            if on_failure == "error":
                raise
            return (f"Working Model Downloader: {exc}", passthrough)

        if enforce == BEFORE_EXECUTION:
            # The files were fetched at queue time; report what is actually on disk.
            report = self._verify_on_disk(entries)
        else:
            report = self._download_now(entries, verify_hash=verify_hash)

        summary = report.summary()
        if report.failed and on_failure == "error":
            raise RuntimeError(summary)
        return (summary, passthrough)

    # -- execution paths --------------------------------------------------

    def _verify_on_disk(self, entries) -> manifest_mod.ApplyReport:
        report = manifest_mod.ApplyReport()
        for entry in entries:
            path = manifest_mod.existing_path(entry)
            report.results.append(
                manifest_mod.ApplyResult(
                    entry=entry,
                    status="present" if path else "failed",
                    path=path,
                    error=None if path else "still missing after the queue-time download",
                )
            )
        return report

    def _download_now(self, entries, *, verify_hash: bool) -> manifest_mod.ApplyReport:
        bar = _progress_bar(len(entries))
        state = {"index": 0}

        def on_entry(_entry) -> None:
            state["index"] += 1
            if bar is not None:
                bar.update_absolute(state["index"], len(entries))

        return manifest_mod.apply(
            entries, cfg=config.load(), verify=verify_hash, on_entry=on_entry
        )

    @classmethod
    def IS_CHANGED(cls, manifest: str = "", **kwargs):  # noqa: A002 - widget name
        """Re-run when the manifest changes, or when a pinned file has gone missing."""
        try:
            entries = manifest_mod.parse(manifest)
        except WmdError:
            return manifest_digest(manifest)
        present = "".join(
            "1" if comfy_env.find_existing(entry.folder, entry.filename) else "0"
            for entry in entries
        )
        return f"{manifest_digest(manifest)}:{present}"


def _progress_bar(total: int):
    try:
        from comfy.utils import ProgressBar
    except Exception:
        return None
    try:
        return ProgressBar(total)
    except Exception:
        return None


# -- VALIDATE_INPUTS, in whichever flavour this ComfyUI can await --------------


async def _validate_async(cls, manifest="", enforce=BEFORE_EXECUTION, on_failure="error",
                          queue_timeout=0, **kwargs):
    done, result = cls._validate(manifest, enforce, on_failure, queue_timeout)
    if done:
        return result
    deadline = _deadline(queue_timeout)
    while _outstanding(result):
        if deadline is not None and time.monotonic() > deadline:
            return (
                "Working Model Downloader: still downloading after "
                f"{queue_timeout}s. Raise queue_timeout, or watch the sidebar panel."
            )
        await asyncio.sleep(0.5)
    return cls._finish(result, on_failure)


def _validate_sync(cls, manifest="", enforce=BEFORE_EXECUTION, on_failure="error",
                   queue_timeout=0, **kwargs):
    done, result = cls._validate(manifest, enforce, on_failure, queue_timeout)
    if done:
        return result
    deadline = _deadline(queue_timeout)
    while _outstanding(result):
        if deadline is not None and time.monotonic() > deadline:
            return (
                "Working Model Downloader: still downloading after "
                f"{queue_timeout}s. Raise queue_timeout, or watch the sidebar panel."
            )
        time.sleep(0.5)
    return cls._finish(result, on_failure)


def install_validator(node_cls=WMDModelDownloader) -> str:
    """Attach the waiting VALIDATE_INPUTS this ComfyUI can actually await."""
    if supports_async_validation():
        node_cls.VALIDATE_INPUTS = classmethod(_validate_async)
        return "async"
    node_cls.VALIDATE_INPUTS = classmethod(_validate_sync)
    return "sync"


MODE = install_validator()

NODE_CLASS_MAPPINGS = {"WMD_ModelDownloader": WMDModelDownloader}
NODE_DISPLAY_NAME_MAPPINGS = {"WMD_ModelDownloader": "Working Model Downloader"}
