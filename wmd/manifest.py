"""The pinned config the node carries.

A manifest is the panel's output and the node's input: an explicit list of files,
where each goes and what it should hash to. Applying one performs no resolution,
no search and no guessing -- which is exactly why a headless run reproduces what
you sorted out interactively, months later, without a browser.

Secrets are never written here: a manifest travels inside a workflow JSON, which
people share.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import requests

from . import comfy_env, download, http, resolve
from .config import Config
from .errors import WmdError
from .models import PROVIDER_CIVITAI, ManifestEntry, RemoteFile, Resolution

VERSION = 1


class ManifestError(WmdError):
    """The manifest could not be read, or asks for something impossible."""


@dataclass
class ApplyResult:
    entry: ManifestEntry
    status: str  # present | downloaded | failed | skipped
    path: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in ("present", "downloaded", "skipped")


@dataclass
class ApplyReport:
    results: list[ApplyResult] = field(default_factory=list)

    @property
    def failed(self) -> list[ApplyResult]:
        return [r for r in self.results if r.status == "failed"]

    def summary(self) -> str:
        if not self.results:
            return "Working Model Downloader: nothing pinned in this workflow."
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        head = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
        lines = [f"Working Model Downloader: {head}."]
        for result in self.results:
            mark = {"present": "=", "downloaded": "+", "skipped": "~", "failed": "!"}.get(
                result.status, "?"
            )
            detail = f" -- {result.error}" if result.error else ""
            lines.append(f"  {mark} {result.entry.folder}/{result.entry.filename}{detail}")
        return "\n".join(lines)


def build(resolutions: list[Resolution]) -> dict[str, Any]:
    """Pin resolved downloads into a manifest document."""
    entries: list[ManifestEntry] = []
    for resolution in resolutions:
        if not resolution.resolved or resolution.file is None:
            continue
        slot = resolution.ref.slot
        entries.append(
            ManifestEntry(
                url=resolution.file.url,
                filename=resolve.safe_relpath(resolution.filename),
                folder=resolution.folder or "",
                provider=resolution.file.provider,
                sha256=resolution.file.sha256,
                size=resolution.file.size,
                slot=(
                    {"node": slot.node_id, "input": slot.input_name} if slot else None
                ),
                origin=resolution.ref.origin,
                tier=resolution.tier,
            )
        )
    return {"version": VERSION, "entries": [entry.to_json() for entry in entries]}


def dumps(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=False)


def parse(raw: str | dict[str, Any] | None) -> list[ManifestEntry]:
    """Read a manifest, rejecting anything we cannot honour exactly."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return []
    if isinstance(raw, str):
        try:
            document = json.loads(raw)
        except ValueError as exc:
            raise ManifestError(f"the manifest is not valid JSON: {exc}") from exc
    else:
        document = raw

    if not isinstance(document, dict):
        raise ManifestError("the manifest must be a JSON object")

    version = document.get("version", VERSION)
    if not isinstance(version, int) or version > VERSION:
        raise ManifestError(
            f"this manifest is version {version}; this node understands up to {VERSION}"
        )

    raw_entries = document.get("entries")
    if raw_entries is None:
        return []
    if not isinstance(raw_entries, list):
        raise ManifestError("the manifest's 'entries' must be a list")

    entries: list[ManifestEntry] = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise ManifestError(f"entry {index} is not an object")
        url = str(item.get("url") or "").strip()
        filename = resolve.safe_relpath(str(item.get("filename") or "").strip())
        folder = comfy_env.map_legacy(str(item.get("folder") or "").strip())
        if not url:
            raise ManifestError(f"entry {index} has no url")
        if not filename:
            raise ManifestError(f"entry {index} ({url}) has no filename")
        if not folder:
            raise ManifestError(f"entry {index} ({filename}) has no folder")
        sha = str(item.get("sha256") or "").strip().lower() or None
        if sha is not None and len(sha) != 64:
            raise ManifestError(f"entry {index} ({filename}) has a malformed sha256")
        size = item.get("size")
        entries.append(
            ManifestEntry(
                url=url,
                filename=filename,
                folder=folder,
                provider=str(item.get("provider") or "direct"),
                sha256=sha,
                size=int(size) if isinstance(size, (int, float)) else None,
                slot=item.get("slot") if isinstance(item.get("slot"), dict) else None,
                origin=str(item.get("origin") or "manifest"),
                tier=str(item.get("tier") or "manifest"),
            )
        )
    return entries


def entry_file(entry: ManifestEntry) -> RemoteFile:
    return RemoteFile(
        url=entry.url,
        filename=os.path.basename(entry.filename),
        provider=entry.provider,
        size=entry.size,
        sha256=entry.sha256,
        # A pinned Civitai size came from the same kilobyte-rounded figure, and
        # the manifest does not record that, so infer it from the provider.
        size_exact=entry.provider != PROVIDER_CIVITAI,
    )


def existing_path(entry: ManifestEntry) -> str | None:
    return comfy_env.find_existing(entry.folder, entry.filename)


def apply(
    entries: list[ManifestEntry],
    *,
    cfg: Config | None = None,
    session: requests.Session | None = None,
    verify: bool = True,
    progress: download.ProgressCallback | None = None,
    on_entry: Any = None,
    cancel: threading.Event | None = None,
) -> ApplyReport:
    """Download every pinned entry that is not already on disk."""
    cfg = cfg or Config()
    session = session or http.new_session()
    report = ApplyReport()

    for entry in entries:
        if on_entry is not None:
            on_entry(entry)

        present = existing_path(entry)
        if present:
            report.results.append(ApplyResult(entry=entry, status="present", path=present))
            continue

        try:
            target = os.path.join(comfy_env.destination_dir(entry.folder), entry.filename)
            outcome = download.download(
                entry_file(entry),
                target,
                cfg=cfg,
                session=session,
                progress=progress,
                cancel=cancel,
                verify=verify and bool(entry.sha256),
            )
        except Exception as exc:  # noqa: BLE001 - one bad entry must not stop the rest
            from .config import redact

            report.results.append(
                ApplyResult(entry=entry, status="failed", error=redact(str(exc), cfg))
            )
            continue

        report.results.append(
            ApplyResult(
                entry=entry,
                status="present" if outcome.status == "present" else "downloaded",
                path=outcome.path,
            )
        )

    return report
