"""Scanning a workflow: documentation notes, embedded metadata, and de-duplication."""

from __future__ import annotations

from wmd import scan, sources
from wmd.models import ORIGIN_NOTE, ORIGIN_PROPERTIES

NOTE_TEXT = """\
# FLUX workflow

Download these before running:

- [flux1-dev.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors) -> models/unet
- VAE: <https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors>

The style LoRA is on civitai: https://civitai.com/models/1234?modelVersionId=5678
(save it as `paper-cut.safetensors`)

See https://example.com/tutorial for a walkthrough.
"""

WORKFLOW = {
    "nodes": [
        {"id": 1, "type": "MarkdownNote", "widgets_values": [NOTE_TEXT]},
        {"id": 2, "type": "Note", "widgets_values": ["Upscaler: https://example.com/x4.safetensors"]},
        {
            "id": 3,
            "type": "UNETLoader",
            "widgets_values": ["flux1-dev.safetensors"],
            "properties": {
                "models": [
                    {
                        "name": "flux1-dev.safetensors",
                        "url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
                        "directory": "diffusion_models",
                    }
                ]
            },
        },
        {"id": 4, "type": "KSampler", "widgets_values": [1, 2, 3]},
    ]
}


def test_notes_yield_their_documented_links():
    refs = scan.refs_from_notes(WORKFLOW)
    urls = [ref.raw for ref in refs]
    assert "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors" in urls
    assert "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors" in urls
    assert "https://civitai.com/models/1234?modelVersionId=5678" in urls
    assert "https://example.com/x4.safetensors" in urls
    # A tutorial link is not a model.
    assert "https://example.com/tutorial" not in urls
    assert all(ref.origin == ORIGIN_NOTE for ref in refs)


def test_a_filename_written_next_to_a_link_is_kept_as_a_label():
    refs = {ref.raw: ref for ref in scan.refs_from_notes(WORKFLOW)}
    civitai = refs["https://civitai.com/models/1234?modelVersionId=5678"]
    assert civitai.nearby_names == ["paper-cut.safetensors"]
    # The filename is in the HuggingFace URL itself, so it is the file, not a label.
    hf = refs["https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"]
    assert hf.nearby_names == []


def test_properties_models_metadata_is_read():
    refs = scan.refs_from_properties(WORKFLOW)
    assert len(refs) == 1
    assert refs[0].origin == ORIGIN_PROPERTIES
    assert refs[0].filename_hint == "flux1-dev.safetensors"
    assert refs[0].folder_hint == "diffusion_models"


def test_scan_dedupes_across_sources_and_keeps_the_stronger_hint():
    result = scan.scan(WORKFLOW)
    flux = [
        ref
        for ref in result.documented
        if ref.key() == "hf:black-forest-labs/FLUX.1-dev@main/flux1-dev.safetensors"
    ]
    assert len(flux) == 1
    # properties.models is collected first, so its directory survives the merge.
    assert flux[0].folder_hint == "diffusion_models"


def test_subgraph_definitions_are_walked():
    workflow = {
        "nodes": [{"id": 1, "type": "KSampler"}],
        "definitions": {
            "subgraphs": [
                {
                    "id": "sub",
                    "nodes": [
                        {
                            "id": 9,
                            "type": "Note",
                            "widgets_values": ["https://example.com/inner.safetensors"],
                        }
                    ],
                }
            ]
        },
    }
    assert [ref.raw for ref in scan.refs_from_notes(workflow)] == [
        "https://example.com/inner.safetensors"
    ]


def test_html_links_in_notes_are_read_structurally():
    text = '<a href="https://example.com/model.safetensors">grab it</a>'
    assert sources.extract_urls(text) == ["https://example.com/model.safetensors"]


def test_trailing_punctuation_is_not_part_of_the_url():
    text = "Get https://example.com/model.safetensors, then restart."
    assert sources.extract_urls(text) == ["https://example.com/model.safetensors"]
