"""Provider interface plus the registry the rest of the package resolves through."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import requests

from ..models import ModelRef, RemoteFile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config


class Provider(Protocol):
    name: str

    def parse(self, text: str) -> ModelRef | None:
        """Turn one URL or identifier into a reference, or None if it isn't ours."""

    def resolve(
        self, ref: ModelRef, cfg: Config, session: requests.Session
    ) -> list[RemoteFile]:
        """Expand a reference into the concrete files it names."""

    def search(
        self, filename: str, cfg: Config, session: requests.Session
    ) -> list[RemoteFile]:
        """Find candidate files whose name matches ``filename`` exactly."""


def registry() -> list[Provider]:
    from . import civitai, direct, huggingface

    return [huggingface.PROVIDER, civitai.PROVIDER, direct.PROVIDER]


def by_name(name: str) -> Provider | None:
    for provider in registry():
        if provider.name == name:
            return provider
    return None


def parse(text: str) -> ModelRef | None:
    """First provider that recognises ``text`` wins; ``direct`` is the catch-all."""
    for provider in registry():
        ref = provider.parse(text)
        if ref is not None:
            return ref
    return None
