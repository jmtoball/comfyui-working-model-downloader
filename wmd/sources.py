"""Turning human-written text into model references.

This is the part no other downloader does: workflow authors document their models
in Note and MarkdownNote nodes, usually as a filename next to a link. Reading those
links gives an exact answer where searching by filename can only guess.
"""

from __future__ import annotations

import posixpath
import re
from urllib.parse import unquote, urlparse

from . import providers
from .models import ORIGIN_MANUAL, ModelRef, has_model_extension

# Bare URLs. Trailing punctuation is stripped afterwards because prose puts commas
# and full stops directly against links, and markdown wraps them in <> or ().
_URL_RE = re.compile(r"""(?:https?://|hf://)[^\s<>"'`\]\)]+""", re.IGNORECASE)
_AIR_RE = re.compile(r"urn:air:[A-Za-z0-9]*:[A-Za-z0-9]*:civitai:\d+(?:@\d+)?", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[(?P<text>[^\]]*)\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HREF_RE = re.compile(r"""href\s*=\s*["'](?P<url>[^"']+)["']""", re.IGNORECASE)

# A filename with a model extension. Deliberately excludes spaces and brackets:
# model files essentially never contain them, and allowing them swallows the
# surrounding prose ("Put [flux1-dev.safetensors").
_FILENAME_RE = re.compile(
    r"[\w.+\-]+\.(?:safetensors|sft|ckpt|pt|pt2|pth|bin|pkl|gguf|onnx|vae)\b",
    re.IGNORECASE,
)

_TRAILING = ".,;:!?'\"`*_)]}>"

# How far back a filename mentioned on its own line still counts as labelling a link.
_NEARBY_LINES = 2


def _clean(url: str) -> str:
    url = url.strip().strip("<>")
    while url and url[-1] in _TRAILING:
        # Keep a closing paren that belongs to the URL itself, e.g. a wiki path.
        if url[-1] == ")" and url.count("(") > url.count(")"):
            break
        url = url[:-1]
    return url


def extract_urls(text: str) -> list[str]:
    """Every link in ``text``, in order, de-duplicated.

    Markdown and HTML links are read structurally first so their label text cannot
    be mistaken for part of the URL.
    """
    found: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        candidate = _clean(candidate)
        if candidate and candidate not in seen:
            seen.add(candidate)
            found.append(candidate)

    remainder = text or ""
    for pattern in (_MD_LINK_RE, _HREF_RE):
        for match in pattern.finditer(remainder):
            add(match.group("url"))
        remainder = pattern.sub(" ", remainder)

    for match in _URL_RE.finditer(remainder):
        add(match.group(0))
    for match in _AIR_RE.finditer(text or ""):
        add(match.group(0))
    return found


def extract_filenames(text: str) -> list[str]:
    """Model filenames mentioned in ``text``, e.g. ``flux1-dev.safetensors``."""
    out: list[str] = []
    for match in _FILENAME_RE.finditer(text or ""):
        name = match.group(0).strip().strip("`'\"").strip()
        name = posixpath.basename(name.replace("\\", "/"))
        if name and has_model_extension(name) and name not in out:
            out.append(name)
    return out


def _url_filename(url: str) -> str:
    return posixpath.basename(unquote(urlparse(url).path))


def refs_from_text(text: str, *, origin: str, origin_node: str = "") -> list[ModelRef]:
    """Parse documentation prose into references, keeping nearby filenames.

    A link is read together with the filename written around it: as the text of its
    markdown link, elsewhere on its own line, or on a nearby line. Authors write
    both ``VAE: `ae.safetensors` `` above the link and ``(save it as x.safetensors)``
    below it.

    Each filename is given to the *closest* link only. A note listing three models
    on consecutive lines would otherwise hand every name to every link, and a wrong
    label is worse than none: these names are what pairs a link with the loader slot
    that is missing.
    """
    if not text:
        return []

    lines = text.splitlines()

    # A markdown label belongs to its own link, so it is claimed here and removed
    # from the line before the proximity pass sees it.
    labels: dict[int, list[str]] = {}
    stripped_lines: list[str] = []
    line_urls: list[list[str]] = []
    for index, line in enumerate(lines):
        remainder = line
        for match in _MD_LINK_RE.finditer(line):
            labels.setdefault(index, []).extend(extract_filenames(match.group("text")))
            remainder = remainder.replace(match.group(0), " " + match.group("url") + " ")
        line_urls.append(extract_urls(line))
        # A filename inside a URL is that link's own file, not a label for anything.
        for url in line_urls[index]:
            remainder = remainder.replace(url, " ")
        stripped_lines.append(remainder)

    url_lines = [index for index, urls in enumerate(line_urls) if urls]
    nearby: dict[int, list[str]] = {}
    for index, remainder in enumerate(stripped_lines):
        names = extract_filenames(remainder)
        if not names or not url_lines:
            continue
        distance = min(abs(index - candidate) for candidate in url_lines)
        if distance > _NEARBY_LINES:
            continue
        for candidate in url_lines:
            if abs(index - candidate) == distance:
                nearby.setdefault(candidate, []).extend(names)

    refs: list[ModelRef] = []
    for index, urls in enumerate(line_urls):
        for url in urls:
            ref = providers.parse(url)
            if ref is None:
                continue
            ref.origin = origin
            ref.origin_node = origin_node

            own = _url_filename(url).lower()
            seen: set[str] = set()
            names: list[str] = []
            for name in [*labels.get(index, []), *nearby.get(index, [])]:
                lowered = name.lower()
                if lowered != own and lowered not in seen:
                    seen.add(lowered)
                    names.append(name)
            ref.nearby_names = names
            refs.append(ref)

    return refs


def refs_from_input(text: str, *, origin: str = ORIGIN_MANUAL) -> list[ModelRef]:
    """Parse the panel's paste box: one entry per line, ``#`` starts a comment.

    More permissive than note parsing, because here every line is meant to be a
    model: a bare ``owner/repo`` is accepted as a HuggingFace repository.
    """
    refs: list[ModelRef] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        candidates = extract_urls(line) or [line]
        for candidate in candidates:
            ref = providers.parse(candidate)
            if ref is not None:
                ref.origin = origin
                refs.append(ref)
    return dedupe(refs)


def dedupe(refs: list[ModelRef]) -> list[ModelRef]:
    """Collapse references to the same thing, merging the hints they carry."""
    out: list[ModelRef] = []
    index: dict[str, ModelRef] = {}
    for ref in refs:
        key = ref.key()
        existing = index.get(key)
        if existing is None:
            index[key] = ref
            out.append(ref)
            continue
        for name in ref.nearby_names:
            if name not in existing.nearby_names:
                existing.nearby_names.append(name)
        existing.folder_hint = existing.folder_hint or ref.folder_hint
        existing.filename_hint = existing.filename_hint or ref.filename_hint
        existing.slot = existing.slot or ref.slot
    return out
