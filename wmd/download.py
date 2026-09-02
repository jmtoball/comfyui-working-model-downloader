"""The transfer itself: resumable, verified, and hostile to plausible-looking junk.

Every guard here exists because it is a failure mode this ecosystem actually has:

* An ``Authorization`` header must not survive the redirect to presigned object
  storage, which rejects requests carrying one (handled in :mod:`wmd.http`).
* An unauthenticated Civitai download redirects to a login page. Saving that HTML
  under a ``.safetensors`` name and reporting success is the single most common
  bug in this space, so the body is sniffed before anything is renamed.
* HuggingFace can hand back a stale presigned Xet URL that 403s; the fix is to ask
  the stable Hub URL again, not to persist the presigned one.
* ``416`` on a resume means the ``.part`` is already complete.
* A ``200`` in reply to a ``Range`` request means the server ignored the range, so
  appending would silently corrupt the file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import requests

from . import http
from .errors import AuthRequired, DownloadFailed
from .models import RemoteFile, has_model_extension

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .config import Config

CHUNK = 1024 * 1024
MAX_ATTEMPTS = 4

# Hosts that serve HuggingFace's presigned content. A 403 from one of these means
# the signature went stale, not that we lack permission.
_XET_HOST_MARKERS = ("xethub", "xet.", "cdn-lfs", "cas-server", "cas-bridge")

# Types that are never a model file. `text/plain` is deliberately absent: real
# storage backends do serve weights with a lazy content type, and the byte sniff
# below catches the error pages that matter regardless of how they are labelled.
_TEXTUAL_TYPES = ("text/html", "application/json", "application/xml")

# No model weights file is this small; a body this size is a message.
_MESSAGE_SIZE = 4096

# How far an inexact advertised size may be out. A figure published in kilobytes
# cannot be wrong by a whole kilobyte or more once converted back.
_SIZE_TOLERANCE = 1024

ProgressCallback = Callable[[int, "int | None"], None]


@dataclass
class Outcome:
    path: str
    status: str  # downloaded | present
    size: int = 0
    sha256: str | None = None
    verified: bool = False


def auth_headers(provider: str, cfg: Config) -> dict[str, str]:
    token = cfg.token_for(provider)
    return {"Authorization": f"Bearer {token}"} if token else {}


def _is_xet_host(url: str) -> bool:
    host = http.host_of(url)
    return any(marker in host for marker in _XET_HOST_MARKERS)


def _looks_textual(response: requests.Response) -> bool:
    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    return any(content_type.startswith(prefix) for prefix in _TEXTUAL_TYPES)


def _is_printable_text(data: bytes) -> bool:
    if not data:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(char in "\r\n\t" or 32 <= ord(char) < 127 or ord(char) > 160 for char in text)


def _reject_textual_body(response: requests.Response, filename: str) -> None:
    """Refuse an error page when a model file was expected.

    This is the failure mode that makes downloaders lie: an auth redirect or a
    rate-limit notice gets written out under the model's name, the download
    "succeeds", and the truth only surfaces when a loader chokes on it much later.

    The body is judged on evidence, not just on its declared type -- a
    misconfigured CDN will happily label an error page ``application/octet-stream``,
    and a correctly-working one sometimes labels weights ``text/plain``.
    """
    if not has_model_extension(filename):
        return
    head = b""
    try:
        for chunk in response.iter_content(chunk_size=512):
            head = chunk or b""
            break
    except requests.RequestException:
        return

    declared = response.headers.get("Content-Length", "")
    tiny = declared.isdigit() and int(declared) < _MESSAGE_SIZE
    markup = head.lstrip()[:1] in (b"<", b"{")

    if markup or _looks_textual(response) or (tiny and _is_printable_text(head)):
        snippet = head[:200].decode("utf-8", "replace").replace("\n", " ").strip()
        raise DownloadFailed(
            f"the server returned a web page instead of {filename}: {snippet[:160]}"
        )
    # The peeked bytes still belong to the file; hand them back to the caller.
    response._wmd_prefetched = head  # type: ignore[attr-defined]


def _iter_body(response: requests.Response):
    prefetched = getattr(response, "_wmd_prefetched", b"")
    if prefetched:
        yield prefetched
    yield from response.iter_content(chunk_size=CHUNK)


def _hash_existing(path: str) -> hashlib._Hash:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest


def _free_space(path: str) -> int:
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or ".").free
    except OSError:
        return 0


def _open_stream(
    session: requests.Session,
    file: RemoteFile,
    headers: dict[str, str],
    timeout: int,
) -> requests.Response:
    """GET the file, refreshing a stale HuggingFace Xet signature once."""
    response = http.follow(session, file.url, headers=headers, stream=True, timeout=timeout)
    if response.status_code == 403 and _is_xet_host(response.url):
        response.close()
        retry_headers = {**headers, "Cache-Control": "no-cache", "Pragma": "no-cache"}
        bust = f"{'&' if '?' in file.url else '?'}wmd_retry={time.time_ns()}"
        response = http.follow(
            session, file.url + bust, headers=retry_headers, stream=True, timeout=timeout
        )
    return response


# --------------------------------------------------------------------------
# Parallel transfer
#
# One connection rarely gets the whole pipe. Measured against the same file, at
# the same moment, from the same rented box: a single stream to HuggingFace held
# 3-29 MB/s while eight ranged connections aggregated 103 MB/s. The link, the
# disk and the CDN were all fine -- one socket simply was not given the
# bandwidth, and which socket gets throttled varies minute to minute, which is
# why the same download can look fast one hour and hopeless the next.
#
# Everything the sequential path guards against still applies here, so the same
# helpers do the guarding: `_open_stream` refreshes a stale Xet signature,
# `_reject_textual_body` catches a login page served under a .safetensors name,
# and `_finalize` verifies the checksum -- reading the finished file back,
# because segments arrive out of order and cannot be hashed as they land.


# Below this, the extra requests cost more than the parallelism returns.
PARALLEL_MIN_SIZE = 32 * 1024 * 1024

# Serving a range means honouring it. A 200 to a Range request means the server
# is about to send the whole file down every socket, which would corrupt the
# assembly, so that answer disqualifies the file from parallel transfer.
_RANGED = 206


# Splitting a span smaller than this is not worth a fresh connection and its
# request round trip.
MIN_STEAL = 16 * 1024 * 1024


class _Segment:
    """A contiguous span of the file, and how much of it is already on disk.

    ``end`` is inclusive and *shrinks* when another worker steals the tail.
    Fixed spans are what make the last stretch of a large download crawl: the
    first connection to finish sits idle while the slowest one carries the rest
    alone, at a quarter of the rate the link was managing a moment earlier.
    """

    __slots__ = ("start", "end", "done")

    def __init__(self, start: int, end: int, done: int = 0) -> None:
        self.start = start
        self.end = end
        self.done = done

    @property
    def cursor(self) -> int:
        """The next byte this segment needs."""
        return self.start + self.done

    @property
    def remaining(self) -> int:
        return self.end - self.cursor + 1


def _covers(segments: list[_Segment], total: int) -> bool:
    """Whether the spans tile the file exactly -- no gap, no overlap."""
    nextbyte = 0
    for segment in sorted(segments, key=lambda s: s.start):
        if segment.start != nextbyte:
            return False
        nextbyte = segment.end + 1
    return nextbyte == total


def _take_work(
    pending: list[_Segment], segments: list[_Segment], lock: threading.Lock
) -> _Segment | None:
    """The next span for a worker that has run out of its own.

    An unclaimed span first -- a resumed plan can hold more of them than we run
    workers -- and otherwise half of whatever span has the most left. Returning
    None means there is no longer enough work to be worth another connection.
    """
    with lock:
        while pending:
            segment = pending.pop()
            if segment.remaining > 0:
                return segment
        donor = max(segments, key=lambda s: s.remaining, default=None)
        if donor is None or donor.remaining < 2 * MIN_STEAL:
            return None
        # The donor keeps everything up to the split; we take the tail. It may
        # write a little past the new end before it notices, which is why the
        # writer clips rather than trusting the bound it started with.
        split = donor.cursor + donor.remaining // 2
        thief = _Segment(split, donor.end)
        donor.end = split - 1
        segments.append(thief)
        return thief


def _plan_path(part: str) -> str:
    return part + ".plan"


def _load_plan(part: str, total: int, connections: int) -> list[_Segment] | None:
    """The spans from a previous run, if they still describe this file.

    Two shapes are accepted: explicit spans, and the equal cut that plans
    written before work stealing implied by their connection count.
    """
    try:
        with open(_plan_path(part), encoding="utf-8") as handle:
            plan = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(plan, dict) or plan.get("total") != total:
        return None

    raw = plan.get("segments")
    if isinstance(raw, list):
        segments = []
        for entry in raw:
            if not isinstance(entry, list) or len(entry) != 3:
                return None
            if not all(isinstance(n, int) and n >= 0 for n in entry):
                return None
            start, end, done = entry
            if start > end or done > end - start + 1:
                return None
            segments.append(_Segment(start, end, done))
        return segments if _covers(segments, total) else None

    # Pre-stealing plan: equal spans, so the cut has to match to line up.
    if plan.get("connections") != connections:
        return None
    done = plan.get("done")
    if not isinstance(done, list) or len(done) != connections:
        return None
    if not all(isinstance(n, int) and n >= 0 for n in done):
        return None
    spans = _segments(total, connections)
    done = done[: len(spans)] + [0] * max(0, len(spans) - len(done))
    segments = [_Segment(s, e, d) for (s, e), d in zip(spans, done, strict=True)]
    if any(s.done > s.end - s.start + 1 for s in segments):
        return None
    return segments


def _save_plan(
    part: str, total: int, connections: int, segments: list[_Segment]
) -> None:
    """Record the spans so an interrupted transfer can pick them up again.

    Only the explicit form is written. A build older than work stealing finds
    no "done" key, treats the plan as unusable and starts the file over --
    which costs a restart rather than assembling the file from wrong offsets.
    """
    try:
        with open(_plan_path(part), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "total": total,
                    "connections": connections,
                    "segments": [[s.start, s.end, s.done] for s in segments],
                },
                handle,
            )
    except OSError:
        pass  # A lost plan costs a restart, not correctness.


def _segments(total: int, connections: int) -> list[tuple[int, int]]:
    span = (total + connections - 1) // connections
    bounds = [(i * span, min(total, (i + 1) * span) - 1) for i in range(connections)]
    return [(start, end) for start, end in bounds if start <= end]


def _probe_ranged(
    session: requests.Session,
    file: RemoteFile,
    headers: dict[str, str],
    timeout: int,
) -> int | None:
    """The server's own size, or None when it will not serve ranges.

    Asked with a one-byte range rather than HEAD: a HEAD through a redirect
    chain reports whichever hop answered last, which is how a 15GB file ends up
    chunked as though it were a 15-byte redirect body.
    """
    response = _open_stream(session, file, {**headers, "Range": "bytes=0-0"}, timeout)
    with response:
        if response.status_code in (401, 403):
            raise AuthRequired(
                file.provider, f"{response.status_code} while downloading {file.filename}."
            )
        if response.status_code != _RANGED:
            return None
        # Deliberately no textual-body check here: the probe asks for one byte,
        # so Content-Length is 1 and any printable byte would read as "a tiny
        # text response", condemning perfectly good weights. A 206 carrying a
        # byte range is already poor cover for an error page, and the first
        # segment below sniffs a real body anyway.
        content_range = response.headers.get("Content-Range") or ""
        _, _, tail = content_range.partition("/")
        return int(tail) if tail.isdigit() else None


def _fetch_segment(
    session: requests.Session,
    file: RemoteFile,
    headers: dict[str, str],
    timeout: int,
    part: str,
    segment: _Segment,
    sniff: bool,
    cancel: threading.Event | None,
    failures: list[BaseException],
) -> bool:
    """Fetch one span to its end. False means the failure has been recorded."""
    for attempt in range(MAX_ATTEMPTS):
        if cancel is not None and cancel.is_set():
            return False
        if segment.remaining <= 0:
            return True
        begin, stop = segment.cursor, segment.end
        try:
            ranged = {**headers, "Range": f"bytes={begin}-{stop}"}
            response = _open_stream(session, file, ranged, timeout)
            with response:
                if response.status_code != _RANGED:
                    raise DownloadFailed(
                        f"{file.filename}: expected 206 for a range, got "
                        f"{response.status_code}"
                    )
                if sniff:
                    # One segment has to carry the sniff the probe could not do.
                    _reject_textual_body(response, file.filename)
                # Every worker holds its own handle and only ever writes inside
                # its own span, so no locking is needed around the file.
                with open(part, "r+b") as handle:
                    handle.seek(begin)
                    cursor = begin
                    for chunk in _iter_body(response):
                        if cancel is not None and cancel.is_set():
                            return False
                        if not chunk:
                            continue
                        # Re-read the bound every chunk: another worker may have
                        # taken this span's tail while the body was streaming,
                        # and writing past it would double-count those bytes.
                        stop = segment.end
                        if cursor > stop:
                            break
                        if cursor + len(chunk) > stop + 1:
                            chunk = chunk[: stop + 1 - cursor]
                        handle.write(chunk)
                        cursor += len(chunk)
                        segment.done += len(chunk)
                        # A steal can land between reading the bound above and
                        # writing here. The bytes are identical either way --
                        # both workers write the same file content -- but
                        # counting them twice would overshoot the total, so the
                        # tally is clamped to the span rather than trusted.
                        span = segment.end - segment.start + 1
                        if segment.done >= span:
                            segment.done = span
                            break
                        if cursor > stop:
                            break
            if segment.remaining <= 0:
                return True
            # A body that stopped short: go round again from the new cursor.
        except (requests.RequestException, DownloadFailed, OSError) as exc:
            if attempt + 1 >= MAX_ATTEMPTS:
                failures.append(exc)
                return False
            time.sleep(min(16.0, 2.0**attempt))
    failures.append(
        DownloadFailed(f"{file.filename}: a segment never finished")
    )
    return False


def _work(
    session: requests.Session,
    file: RemoteFile,
    headers: dict[str, str],
    timeout: int,
    part: str,
    segment: _Segment,
    pending: list[_Segment],
    segments: list[_Segment],
    lock: threading.Lock,
    sniff: bool,
    cancel: threading.Event | None,
    failures: list[BaseException],
) -> None:
    """Keep fetching spans until there is no worthwhile work left to take."""
    current: _Segment | None = segment
    while current is not None:
        if not _fetch_segment(
            session, file, headers, timeout, part, current, sniff, cancel, failures
        ):
            return
        sniff = False
        current = _take_work(pending, segments, lock)


def _download_parallel(
    file: RemoteFile,
    dest_path: str,
    part: str,
    *,
    session: requests.Session,
    headers: dict[str, str],
    timeout: int,
    connections: int,
    progress: ProgressCallback | None,
    cancel: threading.Event | None,
    verify: bool,
    keep_partial_on_cancel: bool,
) -> Outcome | None:
    """Fetch ``file`` over several ranged connections.

    Returns None when the source will not support it -- an unknown size or a
    server that ignores ranges -- so the caller can fall back to one stream.
    """
    total = _probe_ranged(session, file, headers, timeout)
    if not total:
        return None

    segments = _load_plan(part, total, connections)
    if segments is None or not os.path.isfile(part) or os.path.getsize(part) != total:
        # Preallocate so every worker can seek straight to its own slot.
        with open(part, "wb") as handle:
            handle.truncate(total)
        segments = [_Segment(start, end) for start, end in _segments(total, connections)]

    lock = threading.Lock()
    failures: list[BaseException] = []
    # Reversed so pop() hands out the earliest span first; a resumed plan can
    # hold more spans than we run workers, and the extras wait here.
    pending = [s for s in reversed(segments) if s.remaining > 0]

    workers = []
    for _ in range(min(connections, len(pending))):
        segment = pending.pop()
        workers.append(
            threading.Thread(
                target=_work,
                args=(session, file, headers, timeout, part, segment, pending,
                      segments, lock, segment.start == 0, cancel, failures),
                daemon=True,
            )
        )
    for worker in workers:
        worker.start()

    # Progress is reported from here rather than from the workers: callers
    # render it straight onto a UI and have never had to be thread-safe.
    try:
        while any(worker.is_alive() for worker in workers):
            time.sleep(0.25)
            # Snapshot under the lock: a worker may append a stolen span.
            with lock:
                current = list(segments)
            if progress is not None:
                progress(sum(s.done for s in current), total)
            _save_plan(part, total, connections, current)
    finally:
        for worker in workers:
            worker.join()
        _save_plan(part, total, connections, segments)

    if cancel is not None and cancel.is_set():
        if not keep_partial_on_cancel:
            _discard(part)
            _discard(_plan_path(part))
        raise DownloadFailed("cancelled")

    if failures:
        raise DownloadFailed(f"could not download {file.filename}: {failures[0]}")
    fetched = sum(s.done for s in segments)
    if fetched != total:
        raise DownloadFailed(
            f"{file.filename} ended early ({fetched} of {total} bytes)"
        )

    if progress is not None:
        progress(total, total)
    # digest=None on purpose: the segments landed out of order, so the checksum
    # has to come from reading the assembled file back.
    outcome = _finalize(file, part, dest_path, None, verify)
    _discard(_plan_path(part))
    return outcome


def download(
    file: RemoteFile,
    dest_path: str,
    *,
    cfg: Config,
    session: requests.Session | None = None,
    progress: ProgressCallback | None = None,
    cancel: threading.Event | None = None,
    verify: bool = True,
    overwrite: bool = False,
    keep_partial_on_cancel: bool = False,
) -> Outcome:
    """Fetch ``file`` to ``dest_path``, resuming and verifying as it goes.

    ``keep_partial_on_cancel`` is what separates a pause from a cancel: a pause
    leaves the ``.part`` behind so the next attempt resumes, a cancel does not.
    """
    session = session or http.new_session()
    timeout = cfg.prefs.request_timeout
    base_headers = auth_headers(file.provider, cfg)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    if os.path.isfile(dest_path) and not overwrite:
        size = os.path.getsize(dest_path)
        if size_matches(size, file):
            return Outcome(path=dest_path, status="present", size=size)

    if file.size and _free_space(dest_path) < file.size * 1.02:
        raise DownloadFailed(
            f"not enough free space for {file.filename} "
            f"({file.size / 1024**3:.1f} GiB needed)"
        )

    part = dest_path + ".part"
    last_error: Exception | None = None

    # Big files go over several connections; the parallel section above explains
    # why one is usually not enough. Anything the source will not support falls
    # through to the single-stream path below, which remains the only one that
    # can cope with a server that refuses ranges outright.
    connections = max(1, int(getattr(cfg.prefs, "download_connections", 1) or 1))
    if connections > 1 and (file.size or 0) >= PARALLEL_MIN_SIZE:
        outcome = None
        try:
            outcome = _download_parallel(
                file, dest_path, part,
                session=session, headers=dict(base_headers), timeout=timeout,
                connections=connections, progress=progress, cancel=cancel,
                verify=verify, keep_partial_on_cancel=keep_partial_on_cancel,
            )
        except requests.RequestException as exc:
            # Only the probe can fail this way; one stream may still get through.
            last_error = exc
        if outcome is not None:
            return outcome
        # A part preallocated for the parallel plan is already full length, which
        # the sequential path would read as a resume offset and append past.
        _discard(part)
        _discard(_plan_path(part))

    for attempt in range(MAX_ATTEMPTS):
        if cancel is not None and cancel.is_set():
            if not keep_partial_on_cancel:
                _discard(part)
            raise DownloadFailed("cancelled")

        offset = os.path.getsize(part) if os.path.isfile(part) else 0
        headers = dict(base_headers)
        if offset:
            headers["Range"] = f"bytes={offset}-"

        try:
            response = _open_stream(session, file, headers, timeout)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(16.0, 2.0**attempt))
            continue

        with response:
            status = response.status_code

            if status == 416 and offset:
                # The range is past the end: what we have is the whole file.
                break
            if status in (401, 403):
                raise AuthRequired(file.provider, f"{status} while downloading {file.filename}.")
            if status in http.RETRY_STATUSES and attempt + 1 < MAX_ATTEMPTS:
                response.close()
                time.sleep(min(16.0, 2.0**attempt))
                continue
            if status >= 400:
                raise DownloadFailed(f"{status} while downloading {file.filename}")

            if offset and status != 206:
                # Range ignored: appending here would corrupt the file.
                _discard(part)
                offset = 0

            _reject_textual_body(response, file.filename)

            # Content-Length describes this very transfer, so it beats whatever
            # the catalogue said -- and unlike a kilobyte-rounded figure it is
            # exact, which is what deciding "ended early" needs.
            declared = response.headers.get("Content-Length")
            served = int(declared) + offset if declared and declared.isdigit() else None
            total = served if served is not None else file.size

            digest = hashlib.sha256() if verify else None
            if digest is not None and offset:
                digest = _hash_existing(part)

            downloaded = offset
            try:
                with open(part, "ab" if offset else "wb") as handle:
                    for chunk in _iter_body(response):
                        if cancel is not None and cancel.is_set():
                            handle.flush()
                            if not keep_partial_on_cancel:
                                _discard(part)
                            raise DownloadFailed("cancelled")
                        if not chunk:
                            continue
                        handle.write(chunk)
                        if digest is not None:
                            digest.update(chunk)
                        downloaded += len(chunk)
                        if progress is not None:
                            progress(downloaded, total)
            except requests.RequestException as exc:
                # Keep the .part: the next attempt resumes from where this stopped.
                last_error = exc
                time.sleep(min(16.0, 2.0**attempt))
                continue

        # Only the server's own figure is trustworthy enough to call a transfer
        # short; an advertised size that is a kilobyte out would retry forever.
        short = (
            downloaded < served
            if served is not None
            else total is not None and not size_matches(downloaded, file)
        )
        if short:
            last_error = DownloadFailed(
                f"{file.filename} ended early ({downloaded} of {total} bytes)"
            )
            time.sleep(min(16.0, 2.0**attempt))
            continue

        return _finalize(file, part, dest_path, digest, verify)

    if os.path.isfile(part):
        # We got here via 416, or after exhausting retries on a complete file.
        return _finalize(file, part, dest_path, None, verify)
    raise DownloadFailed(f"could not download {file.filename}: {last_error}")


def size_matches(actual: int, file: RemoteFile) -> bool:
    """Whether ``actual`` agrees with the size the source advertised.

    An inexact size is allowed to differ by up to a kilobyte, which is the most a
    kilobyte-rounded figure can be out by. Holding it to the byte discards
    perfectly good files.
    """
    if file.size is None:
        return True
    if file.size_exact:
        return actual == file.size
    return abs(actual - file.size) <= _SIZE_TOLERANCE


def _finalize(
    file: RemoteFile,
    part: str,
    dest_path: str,
    digest: hashlib._Hash | None,
    verify: bool,
) -> Outcome:
    size = os.path.getsize(part)

    actual: str | None = None
    verified = False
    if verify and file.sha256:
        # The checksum settles it. Checking the advertised size first -- and
        # discarding on a mismatch -- would throw away a file the hash proves is
        # byte-for-byte correct, which is exactly what a rounded size causes.
        actual = (digest.hexdigest() if digest is not None else _hash_existing(part).hexdigest())
        if actual.lower() != file.sha256.lower():
            broken = dest_path + ".bad"
            os.replace(part, broken)
            raise DownloadFailed(
                f"{file.filename} failed its checksum; kept the download at {broken}"
            )
        verified = True
    else:
        # No checksum to appeal to, so the size is all we have.
        if not size_matches(size, file):
            _discard(part)
            raise DownloadFailed(
                f"{file.filename} is {size} bytes but should be {file.size}; discarded"
            )
        if digest is not None:
            actual = digest.hexdigest()

    os.replace(part, dest_path)
    return Outcome(path=dest_path, status="downloaded", size=size, sha256=actual, verified=verified)


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
