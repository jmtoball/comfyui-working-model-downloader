"""Errors that carry enough context to tell the user what to actually do."""

from __future__ import annotations


class WmdError(Exception):
    """Base class for every error this package raises deliberately."""


class AuthRequired(WmdError):
    """The source needs credentials we do not have, or the ones we have were rejected."""

    def __init__(self, provider: str, detail: str = ""):
        self.provider = provider
        hint = {
            "civitai": "Set a Civitai API key (civitai.com → Account → API Keys) "
            "in the Model Downloader panel or CIVITAI_API_KEY.",
            "huggingface": "Set a HuggingFace token (huggingface.co/settings/tokens) "
            "in the Model Downloader panel or HF_TOKEN, and accept the model's "
            "licence on its page if it is gated.",
        }.get(provider, "Credentials are required for this source.")
        super().__init__(f"{detail + ' ' if detail else ''}{hint}".strip())


class SourceNotFound(WmdError):
    """The URL parsed fine but the source says there is nothing there."""


class ResolutionFailed(WmdError):
    """We could not turn a reference into a concrete file."""


class DownloadFailed(WmdError):
    """The transfer failed, or what arrived was not the file we asked for."""
