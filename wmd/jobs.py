"""One download engine, shared by the panel and the node.

Both surfaces submit here, so a download started from the sidebar and one started
by a queued prompt cannot race for the same file, and the panel can show progress
for either. Progress lives in the ``.part`` file on disk, so a job survives a
ComfyUI restart: resuming is just submitting it again.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from . import comfy_env, config, download, http
from .models import ManifestEntry, RemoteFile

QUEUED = "queued"
RUNNING = "downloading"
DONE = "done"
PRESENT = "present"
FAILED = "error"
CANCELLED = "cancelled"
PAUSED = "paused"

_TERMINAL = frozenset({DONE, PRESENT, FAILED, CANCELLED})


@dataclass
class Job:
    id: str
    url: str
    filename: str
    folder: str
    dest: str
    provider: str = "direct"
    size: int | None = None
    sha256: str | None = None
    status: str = QUEUED
    downloaded: int = 0
    total: int | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    speed: float = 0.0
    source: str = "panel"
    workflow: str = ""

    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _pause: bool = field(default=False, repr=False)
    _mark: tuple[float, int] = field(default=(0.0, 0), repr=False)

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL

    def belongs_to(self, workflow: str | None) -> bool:
        """Whether this job should be shown under ``workflow``.

        A job with no workflow was started by a queued prompt rather than by the
        panel, so it belongs to no particular graph and is shown under all of
        them -- hiding it would mean a download nobody could see or cancel.
        """
        return workflow is None or self.workflow in ("", workflow)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "filename": self.filename,
            "folder": self.folder,
            "dest": self.dest,
            "provider": self.provider,
            "status": self.status,
            "downloaded": self.downloaded,
            "total": self.total if self.total is not None else self.size,
            "speed": round(self.speed, 1),
            "error": self.error,
            "source": self.source,
            "workflow": self.workflow,
            "created": self.created,
            "updated": self.updated,
        }


class JobManager:
    """Runs downloads on background threads, bounded by the configured concurrency."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._slots = threading.Semaphore(config.load().prefs.max_concurrent_downloads)
        self._path_locks: dict[str, threading.Lock] = {}
        self._listeners: list[Any] = []

    # -- listeners --------------------------------------------------------

    def subscribe(self, callback: Any) -> None:
        """Register a callback invoked with each job update (used to push to the UI)."""
        with self._lock:
            self._listeners.append(callback)

    def _notify(self, job: Job) -> None:
        for callback in list(self._listeners):
            try:
                callback(job)
            except Exception:  # noqa: BLE001 - a broken listener must not kill a download
                continue

    # -- queries ----------------------------------------------------------

    def list(self, workflow: str | None = None) -> list[Job]:
        """Jobs, optionally only those belonging to one workflow.

        Downloads outlive the graph that started them, so the panel asks for its
        own -- otherwise opening a second workflow shows the first one's queue.
        """
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda job: job.created)
        return [job for job in jobs if job.belongs_to(workflow)]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> list[Job]:
        return [job for job in self.list() if not job.terminal]

    def clear_finished(self, workflow: str | None = None) -> int:
        """Drop finished jobs. Running ones are never dropped from under a user."""
        with self._lock:
            finished = [
                job.id for job in self._jobs.values() if job.terminal and job.belongs_to(workflow)
            ]
            for job_id in finished:
                del self._jobs[job_id]
        return len(finished)

    # -- submission -------------------------------------------------------

    def _path_lock(self, path: str) -> threading.Lock:
        with self._lock:
            lock = self._path_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self._path_locks[path] = lock
            return lock

    def submit_entry(
        self, entry: ManifestEntry, *, source: str = "panel", workflow: str = ""
    ) -> Job:
        dest = os.path.join(comfy_env.destination_dir(entry.folder), entry.filename)
        return self.submit(
            RemoteFile(
                url=entry.url,
                filename=os.path.basename(entry.filename),
                provider=entry.provider,
                size=entry.size,
                sha256=entry.sha256,
            ),
            dest=dest,
            folder=entry.folder,
            source=source,
            workflow=workflow,
        )

    def submit(
        self,
        file: RemoteFile,
        *,
        dest: str,
        folder: str,
        source: str = "panel",
        workflow: str = "",
    ) -> Job:
        """Queue a download, reusing the existing job when one is already running."""
        with self._lock:
            for job in self._jobs.values():
                if job.dest == dest and not job.terminal:
                    # Two workflows can want the same file; the newcomer adopts the
                    # transfer already in flight rather than starting a second one.
                    if workflow and not job.workflow:
                        job.workflow = workflow
                    return job
            job = Job(
                id=f"wmd-{next(self._ids)}",
                url=file.url,
                filename=os.path.basename(dest),
                folder=folder,
                dest=dest,
                provider=file.provider,
                size=file.size,
                sha256=file.sha256,
                total=file.size,
                source=source,
                workflow=workflow,
            )
            self._jobs[job.id] = job

        thread = threading.Thread(target=self._run, args=(job, file), daemon=True)
        thread.start()
        return job

    # -- control ----------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.terminal:
            return False
        job._pause = False
        job._cancel.set()
        if job.status == QUEUED:
            self._set(job, CANCELLED)
        return True

    def pause(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.terminal:
            return False
        job._pause = True
        job._cancel.set()
        return True

    def resume(self, job_id: str) -> Job | None:
        job = self.get(job_id)
        if job is None or job.status not in (PAUSED, FAILED, CANCELLED):
            return None
        return self.submit(
            RemoteFile(
                url=job.url,
                filename=job.filename,
                provider=job.provider,
                size=job.size,
                sha256=job.sha256,
            ),
            dest=job.dest,
            folder=job.folder,
            source=job.source,
            workflow=job.workflow,
        )

    # -- worker -----------------------------------------------------------

    def _set(self, job: Job, status: str, error: str | None = None) -> None:
        job.status = status
        job.error = error
        job.updated = time.time()
        self._notify(job)

    def _run(self, job: Job, file: RemoteFile) -> None:
        cfg = config.load()
        self._slots.acquire()
        try:
            with self._path_lock(job.dest):
                if job._cancel.is_set() and not job._pause:
                    self._set(job, CANCELLED)
                    return
                self._set(job, RUNNING)

                def on_progress(done: int, total: int | None) -> None:
                    now = time.time()
                    last_time, last_done = job._mark
                    job.downloaded = done
                    if total:
                        job.total = total
                    if now - last_time >= 0.5:
                        if last_time:
                            job.speed = (done - last_done) / (now - last_time)
                        job._mark = (now, done)
                        job.updated = now
                        self._notify(job)

                try:
                    outcome = download.download(
                        file,
                        job.dest,
                        cfg=cfg,
                        session=http.new_session(),
                        progress=on_progress,
                        cancel=job._cancel,
                        verify=cfg.prefs.verify_hash,
                        keep_partial_on_cancel=job._pause,
                    )
                except Exception as exc:  # noqa: BLE001 - reported on the job
                    if job._pause:
                        self._set(job, PAUSED)
                    elif job._cancel.is_set():
                        self._set(job, CANCELLED)
                    else:
                        self._set(job, FAILED, config.redact(str(exc), cfg))
                    return

                job.downloaded = outcome.size
                job.total = outcome.size
                self._set(job, PRESENT if outcome.status == "present" else DONE)
        finally:
            self._slots.release()


_manager: JobManager | None = None
_manager_lock = threading.Lock()


def manager() -> JobManager:
    """The process-wide manager. One engine, so both surfaces stay in step."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = JobManager()
        return _manager
