"""Pairing documented links with the loader slots that are actually missing.

This join is what makes documented links useful. Knowing that a note links to
``flux1-dev.safetensors`` is interesting; knowing that the workflow's UNETLoader
is missing exactly that file, and that its combo draws from ``diffusion_models``,
turns the link into a download with a destination and a filename that are not
guesses.
"""

from __future__ import annotations

import posixpath
from urllib.parse import unquote, urlparse

from .analysis import clean_local_suffix
from .models import ModelRef, RemoteFile, Slot


def _basename(value: str) -> str:
    return posixpath.basename(str(value or "").replace("\\", "/"))


def _url_filename(url: str) -> str:
    return posixpath.basename(unquote(urlparse(url or "").path))


def candidate_names(ref: ModelRef) -> list[str]:
    """Every filename this reference might turn out to be, before resolving it."""
    names: list[str] = []
    for value in (
        ref.filename_override,
        ref.filename_hint,
        _url_filename(ref.raw),
        *ref.nearby_names,
    ):
        name = _basename(value or "")
        if name and name.lower() not in {n.lower() for n in names}:
            names.append(name)
    return names


def names_match(a: str, b: str) -> bool:
    """Exact filename match, tolerating a browser duplicate suffix on either side."""
    a, b = _basename(a), _basename(b)
    if not a or not b:
        return False
    if a.lower() == b.lower():
        return True
    return clean_local_suffix(a).lower() == clean_local_suffix(b).lower()


def attach(ref: ModelRef, slot: Slot) -> ModelRef:
    """Bind a reference to a slot: the slot dictates both folder and filename.

    ``slot.value`` is kept whole rather than reduced to a basename, so a workflow
    referencing ``SDXL/model.safetensors`` gets the subfolder it expects.
    """
    ref.slot = slot
    if slot.folder:
        ref.folder_hint = ref.folder_hint or slot.folder
    if not ref.filename_override:
        ref.filename_override = str(slot.value).replace("\\", "/")
    return ref


def match_refs_to_slots(
    refs: list[ModelRef], slots: list[Slot]
) -> tuple[list[ModelRef], list[Slot]]:
    """Join by filename, refusing to guess when a slot has more than one candidate.

    Returns the refs that were bound and the slots still without a source.
    """
    taken: set[int] = set()
    bound: list[ModelRef] = []
    unmatched: list[Slot] = []

    for slot in slots:
        wanted = slot.filename
        matches = [
            (position, ref)
            for position, ref in enumerate(refs)
            if position not in taken
            and ref.slot is None
            and any(names_match(name, wanted) for name in candidate_names(ref))
        ]
        if len(matches) != 1:
            # Zero candidates, or an ambiguity we have no business resolving.
            unmatched.append(slot)
            continue
        position, ref = matches[0]
        taken.add(position)
        bound.append(attach(ref, slot))

    return bound, unmatched


def match_files_to_slots(
    resolved: list[tuple[ModelRef, RemoteFile]], slots: list[Slot]
) -> tuple[list[ModelRef], list[Slot]]:
    """Second pass, once resolution has revealed the real filenames.

    A Civitai model page says nothing about the file it will produce until the API
    answers; this catches the slots that only become matchable at that point.
    """
    taken: set[int] = set()
    bound: list[ModelRef] = []
    unmatched: list[Slot] = []

    for slot in slots:
        matches = [
            (position, ref)
            for position, (ref, file) in enumerate(resolved)
            if position not in taken and ref.slot is None and names_match(file.filename, slot.filename)
        ]
        if len(matches) != 1:
            unmatched.append(slot)
            continue
        position, ref = matches[0]
        taken.add(position)
        bound.append(attach(ref, slot))

    return bound, unmatched
