"""The pinned config: what the panel writes and the node replays."""

from __future__ import annotations

import hashlib
import json

import pytest
import responses

from wmd import manifest, resolve
from wmd.models import PROVIDER_HUGGINGFACE, ModelRef, RemoteFile, Resolution, Slot

URL = "https://huggingface.co/org/repo/resolve/main/style.safetensors"
BODY = b"weights" * 100
DIGEST = hashlib.sha256(BODY).hexdigest()


def a_resolution():
    ref = ModelRef(raw=URL, provider=PROVIDER_HUGGINGFACE, origin="note", origin_node="7")
    ref.slot = Slot(node_id="4", input_name="lora_name", node_type="LoraLoader", folder="loras")
    return Resolution(
        ref=ref,
        file=RemoteFile(
            url=URL,
            filename="style.safetensors",
            provider=PROVIDER_HUGGINGFACE,
            size=len(BODY),
            sha256=DIGEST,
        ),
        folder="loras",
        tier="documented",
    )


def test_a_manifest_round_trips(folder_paths):
    document = manifest.build([a_resolution()])
    entries = manifest.parse(manifest.dumps(document))
    assert len(entries) == 1
    entry = entries[0]
    assert (entry.url, entry.filename, entry.folder) == (URL, "style.safetensors", "loras")
    assert entry.sha256 == DIGEST
    assert entry.slot == {"node": "4", "input": "lora_name"}
    assert entry.origin == "note"


def test_unresolved_items_are_never_pinned(folder_paths):
    unresolved = Resolution(ref=ModelRef(raw="x"), reason="nope")
    assert manifest.build([unresolved, a_resolution()])["entries"] != []
    assert len(manifest.build([unresolved])["entries"]) == 0


def test_an_empty_manifest_is_not_an_error(folder_paths):
    assert manifest.parse("") == []
    assert manifest.parse(None) == []
    assert manifest.parse('{"version": 1}') == []


def test_a_newer_manifest_version_is_refused_rather_than_misread(folder_paths):
    with pytest.raises(manifest.ManifestError, match="understands up to"):
        manifest.parse(json.dumps({"version": 99, "entries": []}))


@pytest.mark.parametrize(
    "entry,message",
    [
        ({"filename": "a.safetensors", "folder": "loras"}, "has no url"),
        ({"url": URL, "folder": "loras"}, "has no filename"),
        ({"url": URL, "filename": "a.safetensors"}, "has no folder"),
        (
            {"url": URL, "filename": "a.safetensors", "folder": "loras", "sha256": "abc"},
            "malformed sha256",
        ),
    ],
)
def test_a_malformed_entry_is_rejected_with_a_reason(folder_paths, entry, message):
    with pytest.raises(manifest.ManifestError, match=message):
        manifest.parse(json.dumps({"version": 1, "entries": [entry]}))


def test_invalid_json_says_so(folder_paths):
    with pytest.raises(manifest.ManifestError, match="not valid JSON"):
        manifest.parse("{nope")


def test_a_pinned_filename_cannot_escape_its_folder(folder_paths):
    entries = manifest.parse(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {"url": URL, "filename": "../../etc/passwd", "folder": "loras"}
                ],
            }
        )
    )
    assert entries[0].filename == "etc/passwd"


def test_legacy_folder_names_are_mapped(folder_paths):
    entries = manifest.parse(
        json.dumps(
            {"version": 1, "entries": [{"url": URL, "filename": "a.safetensors", "folder": "unet"}]}
        )
    )
    assert entries[0].folder == "diffusion_models"


@responses.activate
def test_applying_a_manifest_downloads_only_what_is_absent(folder_paths, cfg):
    folder_paths.add_file("vae", "present.safetensors")
    responses.add(responses.GET, URL, body=BODY, status=200)
    entries = manifest.parse(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "url": "https://example.com/present.safetensors",
                        "filename": "present.safetensors",
                        "folder": "vae",
                    },
                    {
                        "url": URL,
                        "filename": "style.safetensors",
                        "folder": "loras",
                        "sha256": DIGEST,
                        "size": len(BODY),
                    },
                ],
            }
        )
    )
    report = manifest.apply(entries, cfg=cfg)
    assert [r.status for r in report.results] == ["present", "downloaded"]
    assert len(responses.calls) == 1


@responses.activate
def test_applying_a_manifest_resolves_nothing(folder_paths, cfg):
    """The whole point of pinning: no API calls, no search, no guessing."""
    responses.add(responses.GET, URL, body=BODY, status=200)
    entries = manifest.parse(manifest.dumps(manifest.build([a_resolution()])))
    manifest.apply(entries, cfg=cfg)
    assert [call.request.url for call in responses.calls] == [URL]


@responses.activate
def test_one_failed_entry_does_not_stop_the_others(folder_paths, cfg):
    responses.add(responses.GET, "https://example.com/gone.safetensors", status=404)
    responses.add(responses.GET, URL, body=BODY, status=200)
    entries = manifest.parse(
        json.dumps(
            {
                "version": 1,
                "entries": [
                    {
                        "url": "https://example.com/gone.safetensors",
                        "filename": "gone.safetensors",
                        "folder": "loras",
                    },
                    {"url": URL, "filename": "style.safetensors", "folder": "loras"},
                ],
            }
        )
    )
    report = manifest.apply(entries, cfg=cfg)
    assert [r.status for r in report.results] == ["failed", "downloaded"]
    assert len(report.failed) == 1
    assert "gone.safetensors" in report.summary()


def test_the_summary_names_every_entry_and_its_fate(folder_paths, cfg):
    report = manifest.ApplyReport(
        results=[
            manifest.ApplyResult(
                entry=manifest.ManifestEntry(url=URL, filename="a.safetensors", folder="loras"),
                status="downloaded",
            )
        ]
    )
    assert "loras/a.safetensors" in report.summary()
    assert "1 downloaded" in report.summary()


def test_a_manifest_never_carries_credentials(folder_paths):
    document = manifest.build([a_resolution()])
    text = manifest.dumps(document).lower()
    for forbidden in ("token", "api_key", "apikey", "authorization", "password"):
        assert forbidden not in text


def test_destination_paths_go_where_folder_paths_says(folder_paths):
    resolution = a_resolution()
    expected = folder_paths.get_folder_paths("loras")[0]
    assert resolve.dest_path(resolution).startswith(expected)
