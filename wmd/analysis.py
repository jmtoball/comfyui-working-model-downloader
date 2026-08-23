"""Working out which models a workflow really needs, and which are missing.

Runs on ComfyUI's API/prompt format, which is already the execution view: muted
nodes are gone and bypassed ones are wired through. On top of that we keep only
nodes that can reach an output, so a disconnected experiment on the canvas does
not conjure up downloads.

The folder a slot belongs to is derived from the node class itself: a combo input
whose option list matches a ``folder_paths`` file list *is* a model reference, and
the folder it matched is exactly where ComfyUI will look for the file. That works
for third-party loaders too, which a table of known node types never would.
"""

from __future__ import annotations

import os
import re
from typing import Any

from . import comfy_env
from .models import Slot, has_model_extension

# Browser and OS duplicate suffixes. A workflow referencing "model (2).safetensors"
# was authored on a machine that downloaded the file twice; locally we have it
# under the clean name, so this is a rename, not a missing model.
_LOCAL_SUFFIX_RE = re.compile(r"(\s*\(\d+\)|\s*-?\s*(copy|kopie))+$", re.IGNORECASE)

# Fallback when a combo's options cannot be matched -- typically because the local
# folder is empty, so there is no file list to match against. Only unambiguous
# names belong here: `clip_name` is deliberately absent because CLIPLoader and
# CLIPVisionLoader use it for different folders, and option matching resolves that.
INPUT_NAME_HINTS = {
    "ckpt_name": "checkpoints",
    "checkpoint_name": "checkpoints",
    "lora_name": "loras",
    "vae_name": "vae",
    "unet_name": "diffusion_models",
    "model_name": "upscale_models",
    "control_net_name": "controlnet",
    "controlnet_name": "controlnet",
    "style_model_name": "style_models",
    "gligen_name": "gligen",
    "clip_vision_name": "clip_vision",
    "ipadapter_file": "ipadapter",
}


def normalize(value: Any) -> str:
    return str(value).replace("\\", "/") if isinstance(value, str) else str(value or "")


def clean_local_suffix(filename: str) -> str:
    """Strip a browser duplicate suffix from the stem, leaving the extension."""
    base = os.path.basename(normalize(filename))
    stem, ext = os.path.splitext(base)
    return _LOCAL_SUFFIX_RE.sub("", stem).strip() + ext


def _combo_options(spec: Any) -> list[str] | None:
    """Options of a combo input spec, or None when the input is not a combo."""
    if not isinstance(spec, (tuple, list)) or not spec:
        return None
    head = spec[0]
    if isinstance(head, (list, tuple)):
        return [option for option in head if isinstance(option, str)]
    # V3 notation: ("COMBO", {"options": [...]})
    if head == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options")
        if isinstance(options, (list, tuple)):
            return [option for option in options if isinstance(option, str)]
    return None


class _FolderIndex:
    """The current file list of every registered model folder, fetched once."""

    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {}
        for key in comfy_env.folder_keys():
            try:
                self._lists[key] = comfy_env.filename_list(key)
            except Exception:
                continue
        self._sets = {key: {normalize(f) for f in files} for key, files in self._lists.items()}

    def match(self, options: list[str]) -> str | None:
        """Which folder a combo's options came from.

        Exact set equality is decisive. Otherwise the smallest folder that contains
        every option wins, which covers loaders that filter the list they show.
        """
        wanted = {normalize(o) for o in options if isinstance(o, str)}
        if not wanted:
            return None
        best: tuple[int, str] | None = None
        for key, files in self._sets.items():
            if not files:
                continue
            if wanted == files:
                return key
            if wanted <= files and (best is None or len(files) < best[0]):
                best = (len(files), key)
        return best[1] if best else None

    def contains(self, key: str, value: str) -> bool:
        return normalize(value) in self._sets.get(key, set())

    def find_basename(self, key: str, filename: str) -> str | None:
        """Same file under a different path or a duplicate-suffixed name."""
        target = clean_local_suffix(filename).lower()
        for candidate in self._lists.get(key, []):
            if clean_local_suffix(candidate).lower() == target:
                return candidate
        return None


def _reachable(prompt: dict[str, Any]) -> tuple[set[str], bool]:
    """Node ids from which an output node is reachable.

    Without a recognisable output node -- an unknown custom output, or no node
    registry at all -- everything is treated as active rather than dropping work.
    """
    classes = comfy_env.node_class_mappings()
    outputs = {
        node_id
        for node_id, node in prompt.items()
        if isinstance(node, dict)
        and getattr(classes.get(node.get("class_type")), "OUTPUT_NODE", False)
    }
    if not outputs:
        return set(prompt), False

    incoming: dict[str, set[str]] = {node_id: set() for node_id in prompt}
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, list) and value and isinstance(value[0], (str, int)):
                incoming[node_id].add(str(value[0]))

    seen: set[str] = set()
    queue = list(outputs)
    while queue:
        current = queue.pop()
        if current in seen or current not in prompt:
            continue
        seen.add(current)
        queue.extend(incoming.get(current, ()))
    return seen, True


def find_slots(prompt: dict[str, Any]) -> tuple[list[Slot], list[Slot]]:
    """Model slots in the active graph.

    Returns ``(active, not_connected)``. Slots on unreachable nodes are reported
    separately rather than dropped: ComfyUI still paints those nodes red, so
    silently omitting them would look like we missed something.
    """
    if not isinstance(prompt, dict):
        return [], []

    index = _FolderIndex()
    reachable, had_outputs = _reachable(prompt)
    active: list[Slot] = []
    inactive: list[Slot] = []

    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("class_type") or "")
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        spec = comfy_env.node_input_types(node_type)
        required = spec.get("required") if isinstance(spec.get("required"), dict) else {}
        optional = spec.get("optional") if isinstance(spec.get("optional"), dict) else {}
        declared = {**(required or {}), **(optional or {})}

        for input_name, value in (inputs or {}).items():
            if not isinstance(value, str) or not value.strip():
                continue

            options = _combo_options(declared.get(input_name))
            folder = index.match(options) if options else None
            if folder is None:
                # No registry, or an empty folder with nothing to match against.
                if options is None and declared:
                    continue
                folder = INPUT_NAME_HINTS.get(input_name)
                if folder is None and not has_model_extension(value):
                    continue
            if folder is None:
                continue

            present = index.contains(folder, value)
            found_as = None if present else index.find_basename(folder, value)
            slot = Slot(
                node_id=str(node_id),
                input_name=input_name,
                node_type=node_type,
                value=value,
                folder=folder,
                missing=not present and found_as is None,
                found_as=found_as,
            )
            if had_outputs and str(node_id) not in reachable:
                inactive.append(slot)
            else:
                active.append(slot)

    return active, inactive


def missing_slots(prompt: dict[str, Any]) -> list[Slot]:
    active, _inactive = find_slots(prompt)
    return [slot for slot in active if slot.missing]
