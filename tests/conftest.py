"""Test fixtures.

The package never imports ComfyUI, so the tests supply a stand-in: a
``folder_paths`` module backed by a temporary tree, and a ``nodes`` module holding
whatever node classes a test needs. Injecting them into ``sys.modules`` is exactly
how the real thing is found, so nothing under test is aware of the difference.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wmd.download  # noqa: E402,F401 - after the path fix, so the sleep patches resolve
import wmd.http  # noqa: E402,F401

DEFAULT_FOLDERS = (
    "checkpoints",
    "loras",
    "vae",
    "text_encoders",
    "diffusion_models",
    "clip_vision",
    "controlnet",
    "upscale_models",
    "embeddings",
    "style_models",
    "hypernetworks",
    "frame_interpolation",
    "geometry_estimation",
    "audio_encoders",
    "detection",
)


class FakeFolderPaths:
    """A stand-in for ComfyUI's ``folder_paths``, rooted at a temporary directory."""

    def __init__(self, root: str, folders=DEFAULT_FOLDERS):
        self.root = root
        self.models = os.path.join(root, "models")
        self.folder_names_and_paths = {
            name: ([os.path.join(self.models, name)], {".safetensors", ".ckpt", ".pt", ".bin"})
            for name in folders
        }
        self.folder_names_and_paths["custom_nodes"] = ([os.path.join(root, "custom_nodes")], set())
        for paths, _ in self.folder_names_and_paths.values():
            os.makedirs(paths[0], exist_ok=True)

    # -- the subset of the real API that wmd uses -------------------------

    def map_legacy(self, name):
        return {"unet": "diffusion_models", "clip": "text_encoders"}.get(name, name)

    def get_folder_paths(self, name):
        return list(self.folder_names_and_paths[self.map_legacy(name)][0])

    def get_filename_list(self, name):
        out = []
        for root in self.get_folder_paths(name):
            for dirpath, _dirs, files in os.walk(root):
                for filename in files:
                    rel = os.path.relpath(os.path.join(dirpath, filename), root)
                    out.append(rel.replace("\\", "/"))
        return sorted(out)

    def get_user_directory(self):
        path = os.path.join(self.root, "user")
        os.makedirs(path, exist_ok=True)
        return path

    # -- test helpers -----------------------------------------------------

    def add_file(self, folder, name, content=b"weights"):
        path = os.path.join(self.get_folder_paths(folder)[0], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)
        return path

    def add_folder(self, name):
        path = os.path.join(self.models, name)
        os.makedirs(path, exist_ok=True)
        self.folder_names_and_paths[name] = ([path], {".safetensors"})


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Retry backoff is real behaviour worth having; waiting for it in tests is not."""
    monkeypatch.setattr("wmd.http.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("wmd.download.time.sleep", lambda _seconds: None)


@pytest.fixture
def folder_paths(tmp_path, monkeypatch):
    fake = FakeFolderPaths(str(tmp_path))
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    monkeypatch.setenv("WMD_USER_DIR", os.path.join(str(tmp_path), "wmd-user"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("CIVITAI_API_KEY", raising=False)
    monkeypatch.delenv("CIVITAI_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", os.path.join(str(tmp_path), "no-hf-home"))
    return fake


@pytest.fixture
def standalone(tmp_path, monkeypatch):
    """No ComfyUI at all: the fallback path used by the CLI."""
    monkeypatch.delitem(sys.modules, "folder_paths", raising=False)
    monkeypatch.setenv("WMD_MODELS_DIR", os.path.join(str(tmp_path), "models"))
    monkeypatch.setenv("WMD_USER_DIR", os.path.join(str(tmp_path), "wmd-user"))
    return tmp_path


def _make_node_class(inputs, output_node=False):
    """A minimal stand-in for a ComfyUI node class."""

    class Node:
        OUTPUT_NODE = output_node

        @classmethod
        def INPUT_TYPES(cls):
            return inputs

    return Node


@pytest.fixture
def make_node_class():
    return _make_node_class


@pytest.fixture
def node_registry(monkeypatch):
    """Install a ``nodes`` module whose NODE_CLASS_MAPPINGS a test controls."""
    module = types.ModuleType("nodes")
    module.NODE_CLASS_MAPPINGS = {}
    monkeypatch.setitem(sys.modules, "nodes", module)
    return module.NODE_CLASS_MAPPINGS


_extension_module = None


def _load_extension():
    """Import the repo as ComfyUI imports a custom node: as a package."""
    global _extension_module
    if _extension_module is None:
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location(
            "wmd_extension",
            os.path.join(root, "__init__.py"),
            submodule_search_locations=[root],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["wmd_extension"] = module
        spec.loader.exec_module(module)
        _extension_module = module
    return _extension_module


@pytest.fixture
def extension(folder_paths):
    """The custom node package, loaded the way ComfyUI loads it.

    Note that the package imports its core as ``wmd_extension.wmd``, which is a
    different module object from the ``wmd`` the other tests import directly. Tests
    that touch shared state -- the job manager in particular -- must go through
    ``extension.wmd`` to see the same instance the node does.
    """
    module = _load_extension()
    module.wmd.jobs._manager = None
    module.wmd_nodes._started.clear()
    return module


@pytest.fixture
def cfg():
    from wmd.config import Config

    return Config()
