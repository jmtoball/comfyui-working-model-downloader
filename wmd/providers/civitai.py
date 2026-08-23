"""Civitai: URL parsing, version resolution and filename search.

The API key goes in an ``Authorization`` header, never in a ``?token=`` query
parameter, so it cannot leak through a log line or a referrer. That only works
because :func:`wmd.http.follow` drops the header on the redirect to presigned
object storage, which rejects requests carrying one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import requests

from .. import http
from ..errors import ResolutionFailed
from ..models import PROVIDER_CIVITAI, ModelRef, RemoteFile

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config

# civitai.red and civitai.green are the SFW/NSFW aliases; all three share one API.
HOSTS = ("civitai.com", "civitai.red", "civitai.green")
API = "https://civitai.com/api"

# urn:air:sd1:lora:civitai:1234@5678
_AIR_RE = re.compile(
    r"^urn:air:(?P<ecosystem>[^:]*):(?P<type>[^:]*):civitai:(?P<model>\d+)(?:@(?P<version>\d+))?$",
    re.IGNORECASE,
)

# Query parameters Civitai accepts on a download URL, worth preserving verbatim.
_FILE_PARAMS = ("type", "format", "size", "fp")

# Civitai's model.type vocabulary mapped onto ComfyUI folder keys. The values that
# are not model weights map to None so they can be reported rather than downloaded.
MODEL_TYPE_FOLDERS: dict[str, str | None] = {
    "checkpoint": "checkpoints",
    "textualinversion": "embeddings",
    "hypernetwork": "hypernetworks",
    "aestheticgradient": "embeddings",
    "lora": "loras",
    "locon": "loras",
    "dora": "loras",
    "lycoris": "loras",
    "controlnet": "controlnet",
    "upscaler": "upscale_models",
    "vae": "vae",
    "motionmodule": "diffusion_models",
    "poses": None,
    "wildcards": None,
    "workflows": None,
    "other": None,
}


class CivitaiProvider:
    name = PROVIDER_CIVITAI

    def parse(self, text: str) -> ModelRef | None:
        text = (text or "").strip().strip("<>").rstrip(".,;)")
        if not text:
            return None

        air = _AIR_RE.match(text)
        if air:
            ref = ModelRef(raw=text, provider=self.name, model_id=air.group("model"))
            ref.version_id = air.group("version")
            ref.folder_hint = MODEL_TYPE_FOLDERS.get((air.group("type") or "").lower())
            return ref

        if "://" not in text:
            return None
        parsed = urlparse(text)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        if host not in HOSTS:
            return None

        segments = [s for s in parsed.path.split("/") if s]
        params = parse_qs(parsed.query)
        ref = ModelRef(raw=text, provider=self.name)
        for key in _FILE_PARAMS:
            if params.get(key):
                ref.query[key] = params[key][0]

        # /api/download/models/<versionId>
        if segments[:3] == ["api", "download", "models"] and len(segments) > 3:
            ref.version_id = segments[3]
            return ref
        # /api/v1/model-versions/<id> and /api/v1/models/<id>
        if segments[:2] == ["api", "v1"] and len(segments) > 3:
            if segments[2] == "model-versions":
                ref.version_id = segments[3]
                return ref
            if segments[2] == "models":
                ref.model_id = segments[3]
                ref.version_id = (params.get("modelVersionId") or [None])[0]
                return ref
            return None
        # /model-versions/<id>
        if segments[:1] == ["model-versions"] and len(segments) > 1:
            ref.version_id = segments[1]
            return ref
        # /models/<id>[/<slug>][?modelVersionId=]
        if segments[:1] == ["models"] and len(segments) > 1 and segments[1].isdigit():
            ref.model_id = segments[1]
            ref.version_id = (params.get("modelVersionId") or [None])[0]
            return ref
        return None

    # -- resolution -------------------------------------------------------

    def _headers(self, cfg: Config) -> dict[str, str]:
        token = cfg.token_for(self.name)
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _version(self, ref: ModelRef, cfg: Config, session: requests.Session) -> dict[str, Any]:
        headers = self._headers(cfg)
        timeout = cfg.prefs.request_timeout
        if ref.version_id:
            data = http.request_json(
                session,
                f"{API}/v1/model-versions/{ref.version_id}",
                headers=headers,
                provider=self.name,
                timeout=timeout,
            )
            return data if isinstance(data, dict) else {}
        if not ref.model_id:
            raise ResolutionFailed(f"{ref.raw} names neither a Civitai model nor a version")
        model = http.request_json(
            session,
            f"{API}/v1/models/{ref.model_id}",
            headers=headers,
            provider=self.name,
            timeout=timeout,
        )
        versions = (model or {}).get("modelVersions") if isinstance(model, dict) else None
        if not versions:
            raise ResolutionFailed(f"Civitai model {ref.model_id} has no published versions")
        version = dict(versions[0])
        # /models/<id> omits the parent metadata that /model-versions/<id> carries.
        version.setdefault("model", {"name": model.get("name"), "type": model.get("type")})
        version.setdefault("modelId", model.get("id"))
        return version

    def _pick_file(self, version: dict[str, Any], ref: ModelRef) -> dict[str, Any] | None:
        files = [f for f in (version.get("files") or []) if isinstance(f, dict)]
        if not files:
            return None
        wanted = {k: v.lower() for k, v in ref.query.items() if k in _FILE_PARAMS}
        if wanted:
            def matches(entry: dict[str, Any]) -> bool:
                meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
                for key, value in wanted.items():
                    have = str(entry.get(key) if key == "type" else (meta or {}).get(key) or "")
                    if have.lower() != value:
                        return False
                return True

            filtered = [f for f in files if matches(f)]
            if filtered:
                files = filtered
        return next((f for f in files if f.get("primary")), files[0])

    def _to_file(self, version: dict[str, Any], entry: dict[str, Any], ref: ModelRef) -> RemoteFile:
        model = version.get("model") if isinstance(version.get("model"), dict) else {}
        version_id = version.get("id") or ref.version_id
        url = str(entry.get("downloadUrl") or f"{API}/download/models/{version_id}")
        if ref.query:
            extra = "&".join(f"{k}={v}" for k, v in ref.query.items() if k in _FILE_PARAMS)
            if extra:
                url = f"{url}{'&' if '?' in url else '?'}{extra}"
        hashes = entry.get("hashes") if isinstance(entry.get("hashes"), dict) else {}
        size_kb = entry.get("sizeKB")
        return RemoteFile(
            url=url,
            filename=str(entry.get("name") or f"civitai-{version_id}.safetensors"),
            provider=self.name,
            size=int(float(size_kb) * 1024) if isinstance(size_kb, (int, float)) else None,
            sha256=str((hashes or {}).get("SHA256") or "").lower() or None,
            meta={
                "model_id": version.get("modelId") or ref.model_id,
                "version_id": version_id,
                "model_name": (model or {}).get("name") or "",
                "model_type": (model or {}).get("type") or "",
                "base_model": version.get("baseModel") or "",
                "file_type": entry.get("type") or "",
            },
        )

    def resolve(self, ref: ModelRef, cfg: Config, session: requests.Session) -> list[RemoteFile]:
        version = self._version(ref, cfg, session)
        entry = self._pick_file(version, ref)
        if entry is None:
            raise ResolutionFailed(f"Civitai version {version.get('id') or ref.raw} has no files")
        return [self._to_file(version, entry, ref)]

    # -- search -----------------------------------------------------------

    def search(self, filename: str, cfg: Config, session: requests.Session) -> list[RemoteFile]:
        import posixpath

        base = posixpath.basename(filename)
        stem = posixpath.splitext(base)[0]
        if not stem:
            return []
        params: dict[str, object] = {"query": stem, "limit": cfg.prefs.search_limit}
        if cfg.prefs.include_nsfw:
            params["nsfw"] = "true"
        data = http.request_json(
            session,
            f"{API}/v1/models",
            headers=self._headers(cfg),
            params=params,
            provider=self.name,
            timeout=cfg.prefs.request_timeout,
        )
        items = (data or {}).get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []

        wanted = base.lower()
        out: list[RemoteFile] = []
        for model in items:
            if not isinstance(model, dict):
                continue
            for version in model.get("modelVersions") or []:
                if not isinstance(version, dict):
                    continue
                version.setdefault("model", {"name": model.get("name"), "type": model.get("type")})
                version.setdefault("modelId", model.get("id"))
                for entry in version.get("files") or []:
                    if isinstance(entry, dict) and str(entry.get("name") or "").lower() == wanted:
                        out.append(self._to_file(version, entry, ModelRef(raw="", provider=self.name)))
        return out


PROVIDER = CivitaiProvider()
