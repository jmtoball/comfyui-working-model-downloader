"""Shared HTTP behaviour: retries, and redirects that do not leak credentials.

The redirect handling is the important part. Both HuggingFace and Civitai hand out
a redirect to presigned object storage, and object storage rejects a request that
also carries an ``Authorization`` header. We therefore drive redirects ourselves
and drop auth on any cross-host hop, rather than trusting a client default.
"""

from __future__ import annotations

import time
from urllib.parse import urlparse

import requests

from .errors import AuthRequired, DownloadFailed, SourceNotFound

USER_AGENT = "comfyui-working-model-downloader/0.1 (+https://github.com/jmtoball/comfyui-working-model-downloader)"

MAX_REDIRECTS = 10
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def same_host(a: str, b: str) -> bool:
    """Treat a subdomain of the origin as the same host (hf.co → cdn-lfs.hf.co)."""
    ha, hb = host_of(a), host_of(b)
    if not ha or not hb:
        return False
    return ha == hb or ha.endswith("." + hb) or hb.endswith("." + ha)


def _retry_after(response: requests.Response, attempt: int) -> float:
    raw = response.headers.get("Retry-After", "")
    try:
        return max(0.0, min(60.0, float(raw)))
    except ValueError:
        return min(30.0, 2.0**attempt)


def follow(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    stream: bool = False,
    timeout: int = 30,
    method: str = "GET",
) -> requests.Response:
    """Perform a request, following redirects by hand.

    Auth headers are dropped the moment we leave the origin host, so a presigned
    storage URL never sees a bearer token.
    """
    current = url
    current_headers = dict(headers or {})
    origin = current

    for _hop in range(MAX_REDIRECTS):
        response = session.request(
            method,
            current,
            headers=current_headers,
            allow_redirects=False,
            stream=stream,
            timeout=timeout,
        )
        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise DownloadFailed(f"redirect without a Location header from {current}")
            target = requests.compat.urljoin(current, location)
            # Civitai bounces unauthenticated downloads to its login page.
            if "/login" in target and "download-auth" in target:
                raise AuthRequired("civitai", "Civitai redirected the download to its login page.")
            if not same_host(target, origin):
                current_headers.pop("Authorization", None)
            current = target
            continue
        return response

    raise DownloadFailed(f"too many redirects starting at {url}")


def request_json(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, object] | None = None,
    provider: str = "",
    timeout: int = 30,
    attempts: int = 3,
) -> object:
    """GET a JSON document, retrying transient failures."""
    if params:
        query = "&".join(f"{k}={requests.utils.quote(str(v), safe='')}" for k, v in params.items())
        url = f"{url}{'&' if '?' in url else '?'}{query}"

    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = follow(session, url, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last = exc
            time.sleep(min(8.0, 2.0**attempt))
            continue

        if response.status_code in (401, 403):
            raise AuthRequired(provider, f"{response.status_code} from {host_of(url)}.")
        if response.status_code == 404:
            raise SourceNotFound(f"{host_of(url)} has nothing at {urlparse(url).path}")
        if response.status_code in RETRY_STATUSES and attempt + 1 < attempts:
            time.sleep(_retry_after(response, attempt))
            continue
        if response.status_code >= 400:
            raise DownloadFailed(f"{response.status_code} from {url}")
        try:
            return response.json()
        except ValueError as exc:
            raise DownloadFailed(f"{host_of(url)} returned a non-JSON response") from exc

    raise DownloadFailed(f"could not reach {host_of(url)}: {last}")
