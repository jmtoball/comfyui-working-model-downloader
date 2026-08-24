"""Core data structures shared by every layer.

Nothing in ``wmd`` imports ComfyUI; these types are what the pure resolution and
download code passes around, and what the node, the routes and the CLI all speak.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Extensions ComfyUI recognises as model weights, plus the few extras that show up
# in the wild (``.gguf`` for quantised checkpoints, ``.onnx`` for detectors).
MODEL_EXTENSIONS = frozenset(
    {
        ".safetensors",
        ".sft",
        ".ckpt",
        ".pt",
        ".pt2",
        ".pth",
        ".bin",
        ".pkl",
        ".gguf",
        ".onnx",
        ".vae",
    }
)

PROVIDER_HUGGINGFACE = "huggingface"
PROVIDER_CIVITAI = "civitai"
PROVIDER_DIRECT = "direct"

# Where a reference came from. Ordered loosely by how much we trust it.
ORIGIN_MANIFEST = "manifest"
ORIGIN_NOTE = "note"
ORIGIN_PROPERTIES = "properties"
ORIGIN_SLOT = "slot"
ORIGIN_MANUAL = "manual"
ORIGIN_SEARCH = "search"

# Resolution tiers, in the order ``resolve`` tries them. Recorded on every
# Resolution so the panel can explain itself and the manifest can record provenance.
TIER_MANIFEST = "manifest"
TIER_DOCUMENTED = "documented"
TIER_PROPERTIES = "properties"
TIER_RULE = "rule"
TIER_SEARCH = "search"
TIER_MANUAL = "manual"
TIER_UNRESOLVED = "unresolved"

# Whether the workflow actually asks for a file. Notes routinely document
# alternatives, optional extras and whole directories: across a corpus of real
# workflows, 62% of documented links were never referenced by the graph. Offering
# them is useful; downloading them by default is not.
NEED_REQUIRED = "required"      # a loader needs it and it is not on disk
NEED_SPARE = "spare"            # a loader needs it and it is already there
NEED_OPTIONAL = "optional"      # documented or found, but nothing in the graph uses it
NEED_UNKNOWN = "unknown"        # no graph to check against, so we cannot say


def has_model_extension(name: str) -> bool:
    return os.path.splitext(str(name))[1].lower() in MODEL_EXTENSIONS


@dataclass(frozen=True)
class Slot:
    """A model-shaped input on a node in the workflow.

    ``folder`` is the ``folder_paths`` key the slot draws its options from, which is
    the most reliable destination signal we have: it is what ComfyUI itself will
    search when the loader runs.
    """

    node_id: str
    input_name: str
    node_type: str = ""
    value: str = ""
    folder: str | None = None
    missing: bool = True
    found_as: str | None = None

    @property
    def filename(self) -> str:
        return os.path.basename(str(self.value).replace("\\", "/"))


@dataclass
class ModelRef:
    """Something the user wants downloaded, before we know what files it means."""

    raw: str
    provider: str = PROVIDER_DIRECT
    origin: str = ORIGIN_MANUAL
    origin_node: str = ""

    # HuggingFace coordinates.
    repo: str | None = None
    revision: str = "main"
    path: str | None = None
    is_tree: bool = False

    # Civitai coordinates.
    model_id: str | None = None
    version_id: str | None = None
    query: dict[str, str] = field(default_factory=dict)

    # Hints gathered on the way in.
    filename_hint: str | None = None
    folder_hint: str | None = None
    sha256_hint: str | None = None
    slot: Slot | None = None
    # Filenames mentioned next to this link in the text it came from. Not the
    # download name -- these are what `match` uses to pair a documented link with
    # the loader slot it belongs to.
    nearby_names: list[str] = field(default_factory=list)

    # Explicit user decisions; these beat every heuristic.
    folder_override: str | None = None
    filename_override: str | None = None

    def key(self) -> str:
        """Identity for de-duplication: the canonical coordinates, not the raw text."""
        if self.provider == PROVIDER_HUGGINGFACE:
            return f"hf:{self.repo}@{self.revision}/{self.path or ''}"
        if self.provider == PROVIDER_CIVITAI:
            return f"civitai:{self.model_id or ''}:{self.version_id or ''}"
        return f"url:{self.raw}"


@dataclass
class RemoteFile:
    """One concrete downloadable file."""

    url: str
    filename: str
    provider: str = PROVIDER_DIRECT
    size: int | None = None
    sha256: str | None = None
    # Whether ``size`` is the exact byte count. Civitai publishes sizes in
    # kilobytes with limited precision, so converting back cannot reproduce the
    # byte count and the value is only good enough for progress and disk checks.
    size_exact: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Resolution:
    """A file plus the destination we decided on, and why."""

    ref: ModelRef
    file: RemoteFile | None = None
    folder: str | None = None
    tier: str = TIER_UNRESOLVED
    reason: str = ""
    error: str | None = None
    existing_path: str | None = None
    need: str = NEED_UNKNOWN
    candidates: list[RemoteFile] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.file is not None and self.folder is not None and self.error is None

    @property
    def filename(self) -> str:
        if self.ref.filename_override:
            return self.ref.filename_override
        if self.file is not None:
            return self.file.filename
        return self.ref.filename_hint or ""


@dataclass
class ManifestEntry:
    """One pinned download, as stored in the node's manifest widget."""

    url: str
    filename: str
    folder: str
    provider: str = PROVIDER_DIRECT
    sha256: str | None = None
    size: int | None = None
    slot: dict[str, str] | None = None
    origin: str = ORIGIN_MANUAL
    tier: str = TIER_MANUAL

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "url": self.url,
            "filename": self.filename,
            "folder": self.folder,
            "provider": self.provider,
        }
        if self.sha256:
            out["sha256"] = self.sha256
        if self.size is not None:
            out["size"] = self.size
        if self.slot:
            out["slot"] = self.slot
        if self.origin:
            out["origin"] = self.origin
        if self.tier:
            out["tier"] = self.tier
        return out
