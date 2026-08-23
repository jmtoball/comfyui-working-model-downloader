"""Scanning a workflow for everything it tells us about its models.

Three sources, in descending order of precision:

1. ``node.properties.models`` -- ComfyUI's own embedded metadata, with a filename
   and a target directory.
2. Note and MarkdownNote text -- where workflow authors actually paste their
   download links, and the source no other downloader reads.
3. The loader slots themselves -- which files the graph needs and which are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import analysis, providers, sources
from .models import (
    ORIGIN_NOTE,
    ORIGIN_PROPERTIES,
    ModelRef,
    Slot,
)

NOTE_TYPES = {"note", "markdownnote", "note plus (mtb)"}

# Keys ComfyUI and its ecosystem have used for the same embedded metadata.
_MODEL_PROPERTY_KEYS = ("models", "missing_models", "missingModels")


@dataclass
class ScanResult:
    documented: list[ModelRef] = field(default_factory=list)
    missing: list[Slot] = field(default_factory=list)
    not_connected: list[Slot] = field(default_factory=list)
    present: list[Slot] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        def slot_json(slot: Slot) -> dict[str, Any]:
            return {
                "node": slot.node_id,
                "input": slot.input_name,
                "node_type": slot.node_type,
                "value": slot.value,
                "folder": slot.folder,
                "missing": slot.missing,
                "found_as": slot.found_as,
            }

        return {
            "documented": [
                {
                    "url": ref.raw,
                    "provider": ref.provider,
                    "origin": ref.origin,
                    "origin_node": ref.origin_node,
                    "nearby_names": ref.nearby_names,
                    "folder_hint": ref.folder_hint,
                    "filename_hint": ref.filename_hint,
                }
                for ref in self.documented
            ],
            "missing": [slot_json(s) for s in self.missing],
            "not_connected": [slot_json(s) for s in self.not_connected],
            "present": [slot_json(s) for s in self.present],
        }


def _iter_nodes(graph: Any) -> list[dict[str, Any]]:
    """Every node in a UI-format workflow, including nested subgraph definitions."""
    found: list[dict[str, Any]] = []
    stack: list[Any] = [graph]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, list):
            stack.extend(current)
            continue
        if not isinstance(current, dict):
            continue

        nodes = current.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, dict):
                    found.append(node)
                    stack.append(node)
        for key in ("definitions", "subgraphs", "extra"):
            value = current.get(key)
            if isinstance(value, (dict, list)):
                stack.append(value)
    return found


def _note_text(node: dict[str, Any]) -> str:
    values = node.get("widgets_values")
    if isinstance(values, str):
        return values
    if isinstance(values, dict):
        values = list(values.values())
    if not isinstance(values, list):
        return ""
    return "\n".join(str(value) for value in values if isinstance(value, str))


def refs_from_notes(graph: Any) -> list[ModelRef]:
    """URLs documented in Note and MarkdownNote nodes."""
    refs: list[ModelRef] = []
    for node in _iter_nodes(graph):
        node_type = str(node.get("type") or node.get("class_type") or "").strip().lower()
        if node_type not in NOTE_TYPES:
            continue
        text = _note_text(node)
        if not text:
            continue
        refs.extend(
            sources.refs_from_text(
                text, origin=ORIGIN_NOTE, origin_node=str(node.get("id", ""))
            )
        )
    return refs


def refs_from_properties(graph: Any) -> list[ModelRef]:
    """ComfyUI's embedded ``properties.models`` metadata: name, url and directory."""
    refs: list[ModelRef] = []
    for node in _iter_nodes(graph):
        properties = node.get("properties")
        if not isinstance(properties, dict):
            continue
        entries: list[Any] = []
        for key in _MODEL_PROPERTY_KEYS:
            value = properties.get(key)
            if isinstance(value, list):
                entries.extend(value)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url") or "").strip()
            if not url:
                continue
            ref = providers.parse(url)
            if ref is None:
                continue
            ref.origin = ORIGIN_PROPERTIES
            ref.origin_node = str(node.get("id", ""))
            name = str(entry.get("name") or "").strip()
            if name:
                ref.filename_hint = name
                ref.nearby_names = [name]
            directory = str(entry.get("directory") or "").strip()
            if directory:
                ref.folder_hint = directory
            digest = str(entry.get("hash") or "").strip().lower()
            if len(digest) == 64:
                ref.sha256_hint = digest
            refs.append(ref)
    return refs


def scan(workflow: Any = None, prompt: dict[str, Any] | None = None) -> ScanResult:
    """Everything the workflow says about its models.

    ``workflow`` is the UI graph (it carries notes and properties); ``prompt`` is
    the API format (it carries the execution view used for missing-slot analysis).
    Either may be omitted -- you simply get less.
    """
    result = ScanResult()

    if workflow is not None:
        result.documented = sources.dedupe(
            [*refs_from_properties(workflow), *refs_from_notes(workflow)]
        )

    if prompt:
        active, inactive = analysis.find_slots(prompt)
        result.missing = [slot for slot in active if slot.missing]
        result.present = [slot for slot in active if not slot.missing]
        result.not_connected = [slot for slot in inactive if slot.missing]

    return result
