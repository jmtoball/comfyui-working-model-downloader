"""Fetch real ComfyUI workflows from Civitai and measure the scanner against them.

Unit tests prove the logic does what it is supposed to; this proves the logic
survives contact with what people actually publish. The corpus itself is other
people's work and is not committed -- it is fetched on demand into a gitignored
directory.

    python tools/corpus.py fetch            # download workflows into .corpus/
    python tools/corpus.py scan             # what the scanner finds in them
    python tools/corpus.py coverage         # how often a destination can be named

Set CIVITAI_API_KEY to reach the workflows whose authors require a login --
roughly two thirds of them.
"""

from __future__ import annotations

import collections
import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wmd import classify, match, scan, sources  # noqa: E402
from wmd.models import ModelRef, Slot, has_model_extension  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".corpus")
MAX_BYTES = 12 * 1024 * 1024
PAGES_PER_SORT = 8
SORTS = ("Most%20Downloaded", "Newest", "Highest%20Rated", "Most%20Liked")


def _get(url: str, timeout: int = 90) -> bytes:
    headers = {"User-Agent": "wmd-corpus/1.0"}
    token = os.environ.get("CIVITAI_API_KEY", "")
    if token and "civitai.com" in url:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=timeout
    ) as response:
        return response.read(MAX_BYTES + 1)


def _is_workflow(data: object) -> bool:
    """A ComfyUI UI-format workflow: a node list whose entries have types."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("nodes"), list)
        and any(isinstance(node, dict) and "type" in node for node in data["nodes"])
    )


def _save(name: str, raw: bytes) -> int:
    try:
        if not _is_workflow(json.loads(raw)):
            return 0
    except Exception:
        return 0
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:110]
    path = os.path.join(CORPUS, safe)
    if os.path.exists(path):
        return 0
    with open(path, "wb") as handle:
        handle.write(raw)
    return 1


def fetch() -> None:
    os.makedirs(CORPUS, exist_ok=True)
    seen: set[int] = set()
    saved = gated = oversize = 0

    for sort in SORTS:
        api = f"https://civitai.com/api/v1/models?types=Workflows&limit=40&sort={sort}"
        for _page in range(PAGES_PER_SORT):
            if not api:
                break
            try:
                listing = json.loads(_get(api))
            except Exception as exc:
                print(f"  listing failed ({sort}): {exc}", file=sys.stderr)
                break
            api = (listing.get("metadata") or {}).get("nextPage")

            for model in listing.get("items", []):
                if model["id"] in seen:
                    continue
                seen.add(model["id"])
                for version in (model.get("modelVersions") or [])[:1]:
                    for entry in (version.get("files") or [])[:2]:
                        url, name = entry.get("downloadUrl"), entry.get("name") or ""
                        if not url:
                            continue
                        if (entry.get("sizeKB") or 0) * 1024 > MAX_BYTES:
                            oversize += 1
                            continue
                        try:
                            raw = _get(url)
                        except urllib.error.HTTPError as exc:
                            gated += exc.code in (401, 403)
                            continue
                        except Exception:
                            continue

                        if name.lower().endswith(".json"):
                            saved += _save(f"{model['id']}_{name}", raw)
                        elif name.lower().endswith(".zip"):
                            try:
                                archive = zipfile.ZipFile(io.BytesIO(raw))
                            except Exception:
                                continue
                            for member in archive.namelist():
                                # Only JSON is of interest; bundled images are not.
                                if not member.lower().endswith(".json"):
                                    continue
                                try:
                                    inner = archive.read(member)
                                except Exception:
                                    continue
                                saved += _save(f"{model['id']}_{os.path.basename(member)}", inner)

            print(f"  {sort}: {len(seen)} models seen, {saved} workflows saved", file=sys.stderr)

    print(f"models={len(seen)} gated={gated} oversize={oversize} workflows={saved} -> {CORPUS}")


def _workflows():
    if not os.path.isdir(CORPUS):
        sys.exit(f"no corpus at {CORPUS}; run `python tools/corpus.py fetch` first")
    for name in sorted(os.listdir(CORPUS)):
        try:
            with open(os.path.join(CORPUS, name), encoding="utf-8") as handle:
                yield name, json.load(handle)
        except Exception:
            continue


def _referenced_files(graph) -> dict[str, dict]:
    """Widget values that look like model filenames.

    A stand-in for real missing-slot analysis, which needs a live ComfyUI to read
    each node class's INPUT_TYPES. It is also exactly the signal the competing
    downloaders rely on, which makes it the right baseline to compare against.
    """
    found: dict[str, dict] = {}
    for node in scan._iter_nodes(graph):
        values = node.get("widgets_values")
        if isinstance(values, dict):
            values = list(values.values())
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and has_model_extension(value) and len(value) < 200:
                found.setdefault(os.path.basename(value.replace("\\", "/")), node)
    return found


def scan_corpus() -> None:
    stats = collections.Counter()
    for _name, graph in _workflows():
        notes = scan.refs_from_notes(graph)
        properties = scan.refs_from_properties(graph)
        documented = sources.dedupe([*properties, *notes])
        referenced = _referenced_files(graph)
        slots = [
            Slot(node_id="", input_name="?", value=filename, folder="checkpoints")
            for filename in referenced
        ]
        bound, _unmatched = match.match_refs_to_slots(list(documented), slots)

        stats["workflows"] += 1
        stats["with_notes"] += any(
            str(n.get("type", "")).lower() in scan.NOTE_TYPES for n in scan._iter_nodes(graph)
        )
        stats["with_documented_links"] += bool(documented)
        stats["with_properties_models"] += bool(properties)
        stats["with_a_join"] += bool(bound)
        stats["referenced_files"] += len(referenced)
        stats["documented_links"] += len(documented)
        stats["joined"] += len(bound)

    total = max(stats["workflows"], 1)
    print(f"workflows                    {stats['workflows']}")
    print(f"  with Note/MarkdownNote     {stats['with_notes']} ({stats['with_notes'] * 100 // total}%)")
    print(f"  with documented links      {stats['with_documented_links']} ({stats['with_documented_links'] * 100 // total}%)")
    print(f"  with properties.models     {stats['with_properties_models']} ({stats['with_properties_models'] * 100 // total}%)")
    print(f"  where a link joined a file {stats['with_a_join']} ({stats['with_a_join'] * 100 // total}%)")
    print()
    print(f"model files referenced       {stats['referenced_files']}")
    print(f"documented links found       {stats['documented_links']}")
    print(f"links joined to a file       {stats['joined']}")


def coverage() -> None:
    """How often the filename alone names a destination.

    This is the weakest tier on purpose: in a live ComfyUI the loader's own combo
    answers first for anything wired into the graph. What is measured here is the
    fallback used for pasted links and unwired files.
    """
    counts: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    unresolved: list[str] = []
    seen: set[str] = set()

    for _name, graph in _workflows():
        for filename in _referenced_files(graph):
            if filename in seen:
                continue
            seen.add(filename)
            verdict = classify.classify(ModelRef(raw=filename, filename_hint=filename))
            counts[verdict.folder or "UNRESOLVED"] += 1
            if verdict.folder:
                examples[verdict.folder].append(filename)
            else:
                unresolved.append(filename)

    total = sum(counts.values())
    named = total - counts["UNRESOLVED"]
    print(f"distinct model filenames     {total}")
    print(f"destination from name alone  {named} ({named * 100 // max(total, 1)}%)")
    print()
    for folder, count in counts.most_common():
        sample = ", ".join(examples[folder][:2])[:64]
        print(f"  {count:5}  {folder:22} {sample}")
    print("\nunresolved (a sample -- most are checkpoints and LoRAs with arbitrary names):")
    for filename in unresolved[:30]:
        print(f"    {filename[:76]}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "scan"
    {"fetch": fetch, "scan": scan_corpus, "coverage": coverage}.get(command, scan_corpus)()
