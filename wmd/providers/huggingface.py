"""HuggingFace: URL parsing, file resolution and filename search.

Deliberately built on ``requests`` against the public API rather than
``huggingface_hub``: custom nodes share one Python environment, so a heavy pinned
dependency is a liability, and we need our own resumable downloader either way.
"""

from __future__ import annotations

import posixpath
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import requests

from .. import http
from ..errors import ResolutionFailed, SourceNotFound
from ..models import PROVIDER_HUGGINGFACE, ModelRef, RemoteFile, has_model_extension

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config

HOSTS = ("huggingface.co", "hf.co")
API = "https://huggingface.co/api"
BASE = "https://huggingface.co"

# owner/repo, optionally followed by a git-ish view. Owners and repo names allow
# dots and dashes, which is why this is not just a naive split.
_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_BARE_RE = re.compile(rf"^(?:hf://)?(?P<repo>{_NAME}/{_NAME})(?P<rest>/.*)?$")
_VIEW_RE = re.compile(r"^/(resolve|blob|tree)/(?P<rest>.+)$")


def _split_revision(rest: str) -> tuple[str, str]:
    """Split ``<revision>/<path>``; revisions may themselves contain slashes.

    ``refs/pr/3`` and ``refs/heads/main`` are three segments, everything else is one.
    """
    parts = rest.split("/")
    if parts[0] == "refs" and len(parts) >= 3:
        return "/".join(parts[:3]), "/".join(parts[3:])
    return parts[0], "/".join(parts[1:])


class HuggingFaceProvider:
    name = PROVIDER_HUGGINGFACE

    def parse(self, text: str) -> ModelRef | None:
        text = (text or "").strip().strip("<>").rstrip(".,;)")
        if not text:
            return None

        repo_type = "model"
        rest = ""
        repo = ""

        # `hf://org/repo/...` is the Hub's own URI scheme; rewrite it to the web
        # form so one parser handles both.
        if text.lower().startswith("hf://"):
            text = BASE + "/" + text[len("hf://") :].lstrip("/")

        if "://" in text:
            parsed = urlparse(text)
            host = (parsed.hostname or "").lower().removeprefix("www.")
            if host not in HOSTS:
                return None
            path = parsed.path
            segments = [s for s in path.split("/") if s]
            if segments and segments[0] in ("datasets", "spaces"):
                repo_type = segments[0].rstrip("s")
                segments = segments[1:]
            if len(segments) < 2:
                return None
            repo = "/".join(segments[:2])
            rest = "/" + "/".join(segments[2:]) if segments[2:] else ""
        else:
            match = _BARE_RE.match(text)
            if not match:
                return None
            repo = match.group("repo")
            rest = match.group("rest") or ""
            # A bare "some/path.safetensors" is a filename, not a repo id.
            if not rest and has_model_extension(repo):
                return None

        ref = ModelRef(raw=text, provider=self.name, repo=repo)
        if repo_type != "model":
            ref.query["repo_type"] = repo_type

        view = _VIEW_RE.match(rest)
        if view:
            revision, path = _split_revision(view.group("rest"))
            ref.revision = unquote(revision) or "main"
            ref.path = unquote(path).strip("/") or None
            ref.is_tree = view.group(1) == "tree" or ref.path is None
        else:
            ref.is_tree = True

        if ref.path and not ref.is_tree:
            ref.filename_hint = posixpath.basename(ref.path)
        return ref

    # -- resolution -------------------------------------------------------

    def _headers(self, cfg: Config) -> dict[str, str]:
        token = cfg.token_for(self.name)
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _api_prefix(self, ref: ModelRef) -> str:
        repo_type = ref.query.get("repo_type", "model")
        return {"model": "models", "dataset": "datasets", "space": "spaces"}.get(
            repo_type, "models"
        )

    def _download_url(self, ref: ModelRef, path: str) -> str:
        repo_type = ref.query.get("repo_type", "model")
        prefix = {"model": "", "dataset": "datasets/", "space": "spaces/"}.get(repo_type, "")
        return f"{BASE}/{prefix}{ref.repo}/resolve/{ref.revision}/{path}"

    def _tree(
        self, ref: ModelRef, cfg: Config, session: requests.Session, path: str, recursive: bool
    ) -> list[dict[str, Any]]:
        url = f"{API}/{self._api_prefix(ref)}/{ref.repo}/tree/{ref.revision}"
        if path:
            url = f"{url}/{path}"
        params: dict[str, object] = {"recursive": "1"} if recursive else {}
        data = http.request_json(
            session,
            url,
            headers=self._headers(cfg),
            params=params,
            provider=self.name,
            timeout=cfg.prefs.request_timeout,
        )
        return [entry for entry in data if isinstance(entry, dict)] if isinstance(data, list) else []

    def repo_info(self, ref: ModelRef, cfg: Config, session: requests.Session) -> dict[str, Any]:
        """Repo metadata (tags, pipeline_tag, library_name) used to classify files."""
        try:
            data = http.request_json(
                session,
                f"{API}/{self._api_prefix(ref)}/{ref.repo}",
                headers=self._headers(cfg),
                provider=self.name,
                timeout=cfg.prefs.request_timeout,
            )
        except (SourceNotFound, ResolutionFailed):
            return {}
        return data if isinstance(data, dict) else {}

    def _to_file(self, ref: ModelRef, entry: dict[str, Any], info: dict[str, Any]) -> RemoteFile:
        path = str(entry.get("path") or "")
        lfs = entry.get("lfs") if isinstance(entry.get("lfs"), dict) else {}
        return RemoteFile(
            url=self._download_url(ref, path),
            filename=posixpath.basename(path),
            provider=self.name,
            size=(lfs or {}).get("size") or entry.get("size"),
            sha256=(lfs or {}).get("oid") or None,
            meta={
                "repo": ref.repo,
                "revision": ref.revision,
                "repo_path": path,
                "tags": info.get("tags") or [],
                "pipeline_tag": info.get("pipeline_tag") or "",
                "library_name": info.get("library_name") or "",
            },
        )

    def resolve(self, ref: ModelRef, cfg: Config, session: requests.Session) -> list[RemoteFile]:
        if not ref.repo:
            raise ResolutionFailed(f"{ref.raw} is not a HuggingFace repository")

        info = self.repo_info(ref, cfg, session)

        if ref.path and not ref.is_tree:
            directory = posixpath.dirname(ref.path)
            entries = self._tree(ref, cfg, session, directory, recursive=False)
            match = next((e for e in entries if e.get("path") == ref.path), None)
            if match is None:
                # An empty or unlistable directory still has a working resolve URL.
                match = {"path": ref.path, "type": "file"}
            return [self._to_file(ref, match, info)]

        entries = self._tree(ref, cfg, session, ref.path or "", recursive=True)
        files = [
            self._to_file(ref, entry, info)
            for entry in entries
            if entry.get("type") == "file" and has_model_extension(str(entry.get("path") or ""))
        ]
        if not files:
            raise ResolutionFailed(
                f"no model files found in {ref.repo}"
                + (f"/{ref.path}" if ref.path else "")
            )
        return files

    # -- search -----------------------------------------------------------

    def search(self, filename: str, cfg: Config, session: requests.Session) -> list[RemoteFile]:
        """Find repos whose file list contains ``filename`` verbatim.

        Search is a fallback tier: we only ever return exact filename matches, so a
        candidate is either the file the workflow asked for or nothing.
        """
        stem = posixpath.splitext(posixpath.basename(filename))[0]
        if not stem:
            return []
        data = http.request_json(
            session,
            f"{API}/models",
            headers=self._headers(cfg),
            params={"search": stem, "limit": cfg.prefs.search_limit, "full": "true"},
            provider=self.name,
            timeout=cfg.prefs.request_timeout,
        )
        if not isinstance(data, list):
            return []

        wanted = posixpath.basename(filename).lower()
        out: list[RemoteFile] = []
        for repo_data in data:
            if not isinstance(repo_data, dict):
                continue
            repo_id = str(repo_data.get("id") or repo_data.get("modelId") or "")
            siblings = repo_data.get("siblings")
            if not repo_id or not isinstance(siblings, list):
                continue
            ref = ModelRef(raw=f"{BASE}/{repo_id}", provider=self.name, repo=repo_id)
            for sibling in siblings:
                if not isinstance(sibling, dict):
                    continue
                path = str(sibling.get("rfilename") or "")
                if posixpath.basename(path).lower() != wanted:
                    continue
                out.append(self._to_file(ref, {"path": path, "size": sibling.get("size")}, repo_data))
        return out


PROVIDER = HuggingFaceProvider()
