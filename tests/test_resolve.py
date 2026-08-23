"""The whole pipeline: a documented workflow in, placed downloads out."""

from __future__ import annotations

import json
import os

import pytest
import responses

from wmd import resolve, scan
from wmd.http import new_session
from wmd.models import TIER_DOCUMENTED, TIER_SEARCH, TIER_UNRESOLVED

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
HF_API = "https://huggingface.co/api/models/black-forest-labs/FLUX.1-dev"


def load_workflow():
    with open(os.path.join(FIXTURES, "documented_workflow.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    return data["workflow"], data["prompt"]


def register_flux_loaders(registry, folder_paths, make_node_class):
    registry["UNETLoader"] = make_node_class(
        {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default"],),
            }
        }
    )
    registry["VAELoader"] = make_node_class(
        {"required": {"vae_name": (folder_paths.get_filename_list("vae"),)}}
    )
    registry["LoraLoaderModelOnly"] = make_node_class(
        {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {}),
            }
        }
    )
    registry["KSampler"] = make_node_class({"required": {"seed": ("INT", {})}})
    registry["SaveImage"] = make_node_class(
        {"required": {"images": ("IMAGE",)}}, output_node=True
    )


def mock_huggingface():
    responses.add(responses.GET, HF_API, json={"tags": []})
    for directory, name, oid in (
        ("split_files/diffusion_models", "flux1-dev.safetensors", "2" * 64),
        ("split_files/vae", "ae.safetensors", "1" * 64),
    ):
        responses.add(
            responses.GET,
            f"{HF_API}/tree/main/{directory}",
            json=[
                {
                    "type": "file",
                    "path": f"{directory}/{name}",
                    "size": 1024,
                    "lfs": {"oid": oid, "size": 1024},
                }
            ],
        )


def mock_civitai():
    with open(os.path.join(FIXTURES, "civitai_version.json"), encoding="utf-8") as handle:
        responses.add(
            responses.GET,
            "https://civitai.com/api/v1/model-versions/5678",
            json=json.load(handle),
        )


@pytest.fixture
def flux(folder_paths, node_registry, make_node_class):
    register_flux_loaders(node_registry, folder_paths, make_node_class)
    return folder_paths


@responses.activate
def test_documented_links_resolve_the_workflows_missing_models(flux, cfg):
    mock_huggingface()
    mock_civitai()
    workflow, prompt = load_workflow()
    result = scan.scan(workflow, prompt)
    assert len(result.missing) == 3

    resolutions = resolve.resolve_refs(
        result.documented, cfg=cfg, session=new_session(), slots=result.missing
    )
    placed = {r.filename: r for r in resolutions if r.resolved}
    assert placed["flux1-dev.safetensors"].folder == "diffusion_models"
    assert placed["ae.safetensors"].folder == "vae"
    assert placed["paper-cut.safetensors"].folder == "loras"
    assert all(r.tier == TIER_DOCUMENTED for r in placed.values())


@responses.activate
def test_the_folder_comes_from_the_loader_not_the_repo_layout(flux, cfg):
    """The UNET lives under split_files/diffusion_models either way -- but the
    reason should be the loader, because that is the authoritative one."""
    mock_huggingface()
    mock_civitai()
    workflow, prompt = load_workflow()
    result = scan.scan(workflow, prompt)
    resolutions = resolve.resolve_refs(
        result.documented, cfg=cfg, session=new_session(), slots=result.missing
    )
    unet = next(r for r in resolutions if r.filename == "flux1-dev.safetensors")
    assert unet.reason == "UNETLoader reads unet_name from diffusion_models"


@responses.activate
def test_a_civitai_page_gets_the_filename_the_workflow_expects(flux, cfg):
    """The note documents the link and the name; the loader confirms the name."""
    mock_huggingface()
    mock_civitai()
    workflow, prompt = load_workflow()
    result = scan.scan(workflow, prompt)
    resolutions = resolve.resolve_refs(
        result.documented, cfg=cfg, session=new_session(), slots=result.missing
    )
    lora = next(r for r in resolutions if r.folder == "loras")
    assert lora.filename == "paper-cut.safetensors"
    assert lora.ref.slot is not None and lora.ref.slot.node_type == "LoraLoaderModelOnly"


@responses.activate
def test_a_model_already_on_disk_is_reported_as_present(flux, cfg):
    flux.add_file("vae", "ae.safetensors")
    mock_huggingface()
    mock_civitai()
    workflow, prompt = load_workflow()
    result = scan.scan(workflow, prompt)
    assert [s.filename for s in result.missing] == [
        "flux1-dev.safetensors",
        "paper-cut.safetensors",
    ]


@responses.activate
def test_a_slot_with_no_documented_link_is_reported_not_guessed(flux, cfg):
    workflow, prompt = load_workflow()
    result = scan.scan(workflow, prompt)
    resolutions = resolve.resolve_refs([], cfg=cfg, session=new_session(), slots=result.missing)
    assert all(r.tier == TIER_UNRESOLVED for r in resolutions)
    assert "nothing in this workflow says where to get it" in resolutions[0].reason


@responses.activate
def test_the_search_fallback_labels_itself_as_a_guess(flux, cfg):
    responses.add(
        responses.GET,
        "https://huggingface.co/api/models",
        json=[{"id": "someone/mirror", "siblings": [{"rfilename": "paper-cut.safetensors"}]}],
    )
    responses.add(responses.GET, "https://civitai.com/api/v1/models", json={"items": []})
    workflow, prompt = load_workflow()
    result = scan.scan(workflow, prompt)
    lora_slot = [s for s in result.missing if s.filename == "paper-cut.safetensors"]

    resolutions = resolve.search_for_slots(lora_slot, cfg=cfg, session=new_session())
    assert resolutions[0].tier == TIER_SEARCH
    assert "a guess" in resolutions[0].reason
    assert resolutions[0].folder == "loras"


@responses.activate
def test_search_with_several_hits_asks_instead_of_choosing(flux, cfg):
    responses.add(
        responses.GET,
        "https://huggingface.co/api/models",
        json=[
            {"id": "one/mirror", "siblings": [{"rfilename": "paper-cut.safetensors"}]},
            {"id": "two/mirror", "siblings": [{"rfilename": "paper-cut.safetensors"}]},
        ],
    )
    responses.add(responses.GET, "https://civitai.com/api/v1/models", json={"items": []})
    workflow, prompt = load_workflow()
    result = scan.scan(workflow, prompt)
    lora_slot = [s for s in result.missing if s.filename == "paper-cut.safetensors"]

    resolution = resolve.search_for_slots(lora_slot, cfg=cfg, session=new_session())[0]
    assert resolution.tier == TIER_UNRESOLVED
    assert len(resolution.candidates) == 2
    assert "pick one" in resolution.reason


@responses.activate
def test_a_broken_link_reports_the_error_without_stopping_the_others(flux, cfg):
    mock_civitai()
    responses.add(responses.GET, HF_API, status=404)
    workflow, prompt = load_workflow()
    result = scan.scan(workflow, prompt)
    resolutions = resolve.resolve_refs(
        result.documented, cfg=cfg, session=new_session(), slots=result.missing
    )
    assert any(r.error for r in resolutions)
    assert any(r.resolved for r in resolutions)


def test_a_destination_cannot_escape_its_model_folder():
    assert resolve.safe_relpath("../../etc/passwd") == "etc/passwd"
    assert resolve.safe_relpath("/abs/model.safetensors") == "abs/model.safetensors"
    assert resolve.safe_relpath("sub/model.safetensors") == "sub/model.safetensors"
