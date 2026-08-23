"""Credentials and preferences.

Secrets come from the environment by default and can be overridden at runtime from
the sidebar panel; the override is stored in ComfyUI's user directory (mode 0600),
never in the extension folder and never in the workflow.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from . import comfy_env

CONFIG_FILENAME = "config.json"

_HF_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
_CIVITAI_ENV_VARS = ("CIVITAI_API_KEY", "CIVITAI_TOKEN")

_lock = threading.Lock()


@dataclass
class Preferences:
    max_concurrent_downloads: int = 2
    request_timeout: int = 30
    search_limit: int = 20
    include_nsfw: bool = False
    verify_hash: bool = True


@dataclass
class Config:
    hf_token: str = ""
    hf_token_source: str = ""
    civitai_api_key: str = ""
    civitai_api_key_source: str = ""
    prefs: Preferences = field(default_factory=Preferences)

    def token_for(self, provider: str) -> str:
        if provider == "huggingface":
            return self.hf_token
        if provider == "civitai":
            return self.civitai_api_key
        return ""


def config_path() -> str:
    return os.path.join(comfy_env.user_dir(), CONFIG_FILENAME)


def _read_stored() -> dict[str, Any]:
    try:
        with open(config_path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _hf_token_file() -> str:
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
        "huggingface",
    )
    return os.path.join(home, "token")


def _env_secret(names: tuple[str, ...]) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def load() -> Config:
    stored = _read_stored()
    prefs_data = stored.get("prefs") if isinstance(stored.get("prefs"), dict) else {}
    prefs = Preferences()
    for key, value in (prefs_data or {}).items():
        if hasattr(prefs, key) and value is not None:
            setattr(prefs, key, type(getattr(prefs, key))(value))
    prefs.max_concurrent_downloads = max(1, min(8, int(prefs.max_concurrent_downloads)))

    cfg = Config(prefs=prefs)

    # Panel-set values win over the environment: they are the more deliberate choice.
    hf = str(stored.get("hf_token") or "").strip()
    if hf:
        cfg.hf_token, cfg.hf_token_source = hf, "user"
    else:
        hf = _env_secret(_HF_ENV_VARS)
        if hf:
            cfg.hf_token, cfg.hf_token_source = hf, "env"
        else:
            try:
                with open(_hf_token_file(), encoding="utf-8") as handle:
                    hf = handle.read().strip()
            except OSError:
                hf = ""
            if hf:
                cfg.hf_token, cfg.hf_token_source = hf, "hf-cli"

    civitai = str(stored.get("civitai_api_key") or "").strip()
    if civitai:
        cfg.civitai_api_key, cfg.civitai_api_key_source = civitai, "user"
    else:
        civitai = _env_secret(_CIVITAI_ENV_VARS)
        if civitai:
            cfg.civitai_api_key, cfg.civitai_api_key_source = civitai, "env"

    return cfg


def save(updates: dict[str, Any]) -> Config:
    """Merge ``updates`` into the stored config. An empty secret clears the override."""
    with _lock:
        stored = _read_stored()
        for key in ("hf_token", "civitai_api_key"):
            if key in updates:
                value = str(updates[key] or "").strip()
                if value:
                    stored[key] = value
                else:
                    stored.pop(key, None)
        if isinstance(updates.get("prefs"), dict):
            prefs = stored.get("prefs")
            stored["prefs"] = {**(prefs if isinstance(prefs, dict) else {}), **updates["prefs"]}

        path = config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(stored, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return load()


def _mask(secret: str) -> str:
    if not secret:
        return ""
    return f"…{secret[-4:]}" if len(secret) > 4 else "…"


def masked(cfg: Config | None = None) -> dict[str, Any]:
    """The shape the panel gets. Plaintext secrets never leave the backend."""
    cfg = cfg or load()
    return {
        "hf_token_set": bool(cfg.hf_token),
        "hf_token_hint": _mask(cfg.hf_token),
        "hf_token_source": cfg.hf_token_source,
        "civitai_api_key_set": bool(cfg.civitai_api_key),
        "civitai_api_key_hint": _mask(cfg.civitai_api_key),
        "civitai_api_key_source": cfg.civitai_api_key_source,
        "prefs": asdict(cfg.prefs),
        "config_path": config_path(),
    }


def redact(text: str, cfg: Config | None = None) -> str:
    """Scrub known secrets out of anything that might be logged or shown."""
    if not text:
        return text
    cfg = cfg or load()
    out = str(text)
    for secret in (cfg.hf_token, cfg.civitai_api_key):
        if secret and len(secret) >= 8:
            out = out.replace(secret, "***")
    return out
