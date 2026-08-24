"""The only module that knows ComfyUI exists.

Everything else in ``wmd`` asks this adapter where models live and what a node
class looks like. When ComfyUI is not importable (tests, the CLI, CI) it falls
back to a directory tree rooted at ``$WMD_MODELS_DIR``, so the whole package
stays runnable standalone.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from .models import MODEL_EXTENSIONS

# ComfyUI renamed two folder keys and keeps mapping the old names; mirror that here
# so a workflow or a hint using the legacy name still lands in the right place.
_LEGACY = {"unet": "diffusion_models", "clip": "text_encoders"}

# Used only by the standalone fallback, to give the CLI a sane folder list.
_FALLBACK_KEYS = (
    "checkpoints",
    "loras",
    "vae",
    "text_encoders",
    "diffusion_models",
    "clip_vision",
    "style_models",
    "embeddings",
    "controlnet",
    "gligen",
    "upscale_models",
    "latent_upscale_models",
    "hypernetworks",
    "photomaker",
    "classifiers",
    "model_patches",
    "audio_encoders",
    "background_removal",
    "frame_interpolation",
    "geometry_estimation",
    "optical_flow",
    "detection",
    "diffusers",
    "configs",
)


def _folder_paths() -> Any | None:
    """The live ``folder_paths`` module, or None when running standalone.

    Deliberately not cached: tests inject a stub into ``sys.modules`` between
    cases, and ComfyUI itself lets custom nodes register folders after import.
    """
    module = sys.modules.get("folder_paths")
    if module is not None:
        return module
    try:
        import folder_paths  # type: ignore[import-not-found]
    except Exception:
        return None
    return folder_paths


def available() -> bool:
    return _folder_paths() is not None


def _fallback_root() -> str:
    return os.environ.get("WMD_MODELS_DIR") or os.path.join(os.getcwd(), "models")


def map_legacy(key: str) -> str:
    key = (key or "").strip().strip("/").replace("\\", "/")
    # Hints from `properties.models` are directory paths ("models/loras"), not keys.
    if key.startswith("models/"):
        key = key[len("models/") :]
    key = key.split("/")[0]
    fp = _folder_paths()
    if fp is not None and hasattr(fp, "map_legacy"):
        try:
            return fp.map_legacy(key)
        except Exception:
            pass
    return _LEGACY.get(key, key)


def folder_keys() -> list[str]:
    fp = _folder_paths()
    if fp is not None:
        try:
            keys = [k for k in fp.folder_names_and_paths if k != "custom_nodes"]
            if keys:
                return sorted(keys)
        except Exception:
            pass
    return list(_FALLBACK_KEYS)


def is_folder_key(key: str) -> bool:
    return bool(key) and map_legacy(key) in set(folder_keys())


def folder_paths_for(key: str) -> list[str]:
    key = map_legacy(key)
    fp = _folder_paths()
    if fp is not None:
        try:
            return list(fp.get_folder_paths(key))
        except Exception:
            pass
    return [os.path.join(_fallback_root(), key)]


def filename_list(key: str) -> list[str]:
    """Relative names of the model files currently registered under ``key``."""
    key = map_legacy(key)
    fp = _folder_paths()
    if fp is not None:
        try:
            return list(fp.get_filename_list(key))
        except Exception:
            return []
    names: list[str] = []
    for root in folder_paths_for(key):
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in MODEL_EXTENSIONS:
                    rel = os.path.relpath(os.path.join(dirpath, name), root)
                    names.append(rel.replace("\\", "/"))
    return sorted(names)


def find_existing(key: str, filename: str) -> str | None:
    """Absolute path of ``filename`` under any registered path for ``key``.

    Searches every registered root rather than just the first, so a model already
    present via ``extra_model_paths.yaml`` is not downloaded a second time.
    """
    base = os.path.basename(str(filename).replace("\\", "/"))
    if not base:
        return None
    for root in folder_paths_for(key):
        direct = os.path.join(root, base)
        if os.path.isfile(direct):
            return direct
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            if base in filenames:
                return os.path.join(dirpath, base)
    return None


def destination_dir(key: str) -> str:
    """Where a new file for ``key`` should be written.

    Prefers the first registered path that already exists and is writable, so we
    follow the user's ``extra_model_paths.yaml`` priority instead of forcing
    everything into the default install.
    """
    roots = folder_paths_for(key)
    if not roots:
        raise ValueError(f"no registered path for model folder {key!r}")
    for root in roots:
        if os.path.isdir(root) and os.access(root, os.W_OK):
            return root
    return roots[0]


def user_dir() -> str:
    """Directory for our own config and rules, inside ComfyUI's user data."""
    fp = _folder_paths()
    if fp is not None and hasattr(fp, "get_user_directory"):
        try:
            return os.path.join(fp.get_user_directory(), "working_model_downloader")
        except Exception:
            pass
    override = os.environ.get("WMD_USER_DIR")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".comfyui-working-model-downloader")


def node_class_mappings() -> dict[str, Any]:
    """ComfyUI's registry of node classes, used to inspect loader inputs."""
    try:
        import nodes  # type: ignore[import-not-found]
    except Exception:
        return {}
    return dict(getattr(nodes, "NODE_CLASS_MAPPINGS", {}) or {})


def node_input_types(node_type: str) -> dict[str, Any]:
    cls = node_class_mappings().get(node_type)
    if cls is None:
        return {}
    try:
        spec = cls.INPUT_TYPES()
    except Exception:
        return {}
    return spec if isinstance(spec, dict) else {}
