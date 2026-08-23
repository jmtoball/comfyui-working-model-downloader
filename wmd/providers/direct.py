"""Catch-all for plain https links to a model file.

Only accepts URLs that actually look like a file download, so a link to a blog
post in someone's notes does not turn into a download job.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import requests

from ..models import PROVIDER_DIRECT, ModelRef, RemoteFile, has_model_extension

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config


class DirectProvider:
    name = PROVIDER_DIRECT

    def parse(self, text: str) -> ModelRef | None:
        text = (text or "").strip().strip("<>").rstrip(".,;)")
        parsed = urlparse(text)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return None
        filename = posixpath.basename(unquote(parsed.path))
        if not has_model_extension(filename):
            return None
        return ModelRef(raw=text, provider=self.name, filename_hint=filename)

    def resolve(self, ref: ModelRef, cfg: Config, session: requests.Session) -> list[RemoteFile]:
        filename = ref.filename_hint or posixpath.basename(unquote(urlparse(ref.raw).path))
        return [RemoteFile(url=ref.raw, filename=filename, provider=self.name)]

    def search(self, filename: str, cfg: Config, session: requests.Session) -> list[RemoteFile]:
        return []


PROVIDER = DirectProvider()
