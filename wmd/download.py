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
