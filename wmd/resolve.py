"""Turning references and missing slots into concrete, placed downloads.

The order matters, and it is the whole argument for this package: documented links
are tried first and search last. A link the workflow's author wrote down is a fact;
a filename match found by searching is a guess, and it is labelled as one.
"""

from __future__ import annotations

import os
import posixpath

import requests

from . import classify, comfy_env, http, match, providers
from .config import Config
from .errors import WmdError
from .models import (
    ORIGIN_SEARCH,
    TIER_DOCUMENTED,
    TIER_MANUAL,
    TIER_PROPERTIES,
    TIER_SEARCH,
    TIER_UNRESOLVED,
    ModelRef,
    RemoteFile,
    Resolution,
    Slot,
)


def safe_relpath(name: str) -> str:
    """A destination-relative path that cannot escape its model folder."""
    cleaned = str(name or "").replace("\\", "/")
    parts = [p for p in cleaned.split("/") if p not in ("", ".", "..")]
    return posixpath.join(*parts) if parts else ""


def dest_path(resolution: Resolution) -> str:
    if not resolution.folder:
        raise WmdError(f"{resolution.filename or resolution.ref.raw} has no destination folder")
    relative = safe_relpath(resolution.filename)
    if not relative:
        raise WmdError(f"{resolution.ref.raw} resolved without a filename")
    return os.path.join(comfy_env.destination_dir(resolution.folder), relative)


def _tier_for_origin(ref: ModelRef) -> str:
    return {
        "note": TIER_DOCUMENTED,
        "properties": TIER_PROPERTIES,
        "search": TIER_SEARCH,
    }.get(ref.origin, TIER_MANUAL)


def _finish(resolution: Resolution) -> Resolution:
    """Attach a destination folder and note whether the file is already on disk."""
    verdict = classify.classify(resolution.ref, resolution.file)
    resolution.folder = verdict.folder
    resolution.reason = verdict.reason
    if verdict.folder is None:
        resolution.tier = TIER_UNRESOLVED
        return resolution
    if resolution.tier == TIER_UNRESOLVED:
        resolution.tier = _tier_for_origin(resolution.ref)
    resolution.existing_path = comfy_env.find_existing(verdict.folder, resolution.filename)
    return resolution


def _files_for(ref: ModelRef, cfg: Config, session: requests.Session) -> list[RemoteFile]:
    provider = providers.by_name(ref.provider)
    if provider is None:
        raise WmdError(f"no provider handles {ref.raw}")
    files = provider.resolve(ref, cfg, session)
    # A repo or subtree can hold many files; a bound slot wants exactly one.
    if ref.slot is not None and len(files) > 1:
        wanted = ref.slot.filename
        exact = [f for f in files if match.names_match(f.filename, wanted)]
        if exact:
            return exact[:1]
    return files


def resolve_refs(
    refs: list[ModelRef],
    *,
    cfg: Config | None = None,
    session: requests.Session | None = None,
    slots: list[Slot] | None = None,
    allow_search: bool = False,
) -> list[Resolution]:
    """Resolve everything we know about, then optionally search for what is left."""
    cfg = cfg or Config()
    session = session or http.new_session()
    refs = list(refs)
    pending = list(slots or [])

    # Pass one: pair documented links with missing slots using the names we already
    # have, so resolution can be narrowed to the single file a slot needs.
    if pending:
        _bound, pending = match.match_refs_to_slots(refs, pending)

    resolutions: list[Resolution] = []
    resolved_pairs: list[tuple[ModelRef, RemoteFile]] = []

    for ref in refs:
        try:
            files = _files_for(ref, cfg, session)
        except WmdError as exc:
            resolutions.append(Resolution(ref=ref, error=str(exc), reason="could not be resolved"))
            continue
        for file in files:
            resolutions.append(Resolution(ref=ref, file=file))
            resolved_pairs.append((ref, file))

    # Pass two: some sources only reveal their filename once resolved (a Civitai
    # model page, an unlabelled repo link). Try those against the remaining slots.
    if pending:
        _bound, pending = match.match_files_to_slots(resolved_pairs, pending)

    for resolution in resolutions:
        if resolution.error is None:
            _finish(resolution)

    if pending and allow_search:
        resolutions.extend(search_for_slots(pending, cfg=cfg, session=session))
    elif pending:
        resolutions.extend(unresolved_for_slots(pending))

    return resolutions


def unresolved_for_slots(slots: list[Slot]) -> list[Resolution]:
    """Report a missing model we have no source for, rather than inventing one."""
    out: list[Resolution] = []
    for slot in slots:
        ref = ModelRef(raw=slot.filename, origin="slot", filename_hint=slot.filename, slot=slot)
        out.append(
            Resolution(
                ref=ref,
                folder=slot.folder,
                tier=TIER_UNRESOLVED,
                reason=(
                    f"{slot.node_type or 'a loader'} needs {slot.filename} in {slot.folder}, "
                    "but nothing in this workflow says where to get it"
                ),
            )
        )
    return out


def search_for_slots(
    slots: list[Slot],
    *,
    cfg: Config | None = None,
    session: requests.Session | None = None,
) -> list[Resolution]:
    """Last resort: look for an exact filename match on HuggingFace and Civitai.

    Only exact matches count, and the result is never auto-selected when more than
    one source has the name -- the panel asks instead.
    """
    cfg = cfg or Config()
    session = session or http.new_session()
    out: list[Resolution] = []

    for slot in slots:
        candidates: list[RemoteFile] = []
        for provider in providers.registry():
            try:
                candidates.extend(provider.search(slot.filename, cfg, session))
            except WmdError:
                continue

        ref = ModelRef(
            raw=slot.filename,
            origin=ORIGIN_SEARCH,
            filename_hint=slot.filename,
            slot=slot,
        )
        if not candidates:
            out.append(
                Resolution(
                    ref=ref,
                    folder=slot.folder,
                    tier=TIER_UNRESOLVED,
                    reason=f"no source on HuggingFace or Civitai has a file named {slot.filename}",
                    candidates=[],
                )
            )
            continue

        resolution = Resolution(ref=ref, candidates=candidates, tier=TIER_SEARCH)
        if len(candidates) == 1:
            resolution.file = candidates[0]
            _finish(resolution)
            resolution.tier = TIER_SEARCH
            resolution.reason = f"found by searching for {slot.filename} (a guess: verify it)"
        else:
            resolution.folder = slot.folder
            resolution.reason = (
                f"{len(candidates)} sources have a file named {slot.filename}; pick one"
            )
            resolution.tier = TIER_UNRESOLVED
        out.append(resolution)

    return out


def resolution_json(resolution: Resolution) -> dict[str, object]:
    """The shape the panel and the CLI render."""
    file = resolution.file
    slot = resolution.ref.slot
    try:
        destination = dest_path(resolution) if resolution.resolved else None
    except WmdError:
        destination = None
    return {
        "url": file.url if file else resolution.ref.raw,
        "source_url": resolution.ref.raw,
        "provider": file.provider if file else resolution.ref.provider,
        "filename": resolution.filename,
        "folder": resolution.folder,
        "size": file.size if file else None,
        "sha256": file.sha256 if file else None,
        "tier": resolution.tier,
        "reason": resolution.reason,
        "error": resolution.error,
        "origin": resolution.ref.origin,
        "origin_node": resolution.ref.origin_node,
        "existing_path": resolution.existing_path,
        "dest_path": destination,
        "resolved": resolution.resolved,
        "slot": (
            {"node": slot.node_id, "input": slot.input_name, "node_type": slot.node_type}
            if slot
            else None
        ),
        "candidates": [
            {
                "url": candidate.url,
                "filename": candidate.filename,
                "provider": candidate.provider,
                "size": candidate.size,
                "sha256": candidate.sha256,
                "meta": candidate.meta,
            }
            for candidate in resolution.candidates
        ],
    }
