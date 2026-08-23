"""Provider resolution against recorded API shapes."""

from __future__ import annotations

import json
import os

import pytest
import responses

from wmd import providers
from wmd.errors import AuthRequired
from wmd.http import new_session

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture
def session():
    return new_session()


# -- HuggingFace ---------------------------------------------------------


@responses.activate
def test_a_single_file_link_resolves_with_its_size_and_checksum(cfg, session):
    responses.add(responses.GET, "https://huggingface.co/api/models/org/repo", json={"tags": []})
    responses.add(
        responses.GET,
        "https://huggingface.co/api/models/org/repo/tree/main/split_files/vae",
        json=fixture("hf_tree.json"),
    )
    ref = providers.parse("https://huggingface.co/org/repo/blob/main/split_files/vae/ae.safetensors")
    files = providers.by_name("huggingface").resolve(ref, cfg, session)
    assert len(files) == 1
    assert files[0].filename == "ae.safetensors"
    assert files[0].size == 335304388
    assert files[0].sha256 == "1" * 64
    assert files[0].url == (
        "https://huggingface.co/org/repo/resolve/main/split_files/vae/ae.safetensors"
    )


@responses.activate
def test_a_repo_link_resolves_to_its_weight_files_only(cfg, session):
    responses.add(responses.GET, "https://huggingface.co/api/models/org/repo", json={"tags": []})
    responses.add(
        responses.GET,
        "https://huggingface.co/api/models/org/repo/tree/main",
        json=fixture("hf_tree.json"),
    )
    ref = providers.parse("https://huggingface.co/org/repo")
    files = providers.by_name("huggingface").resolve(ref, cfg, session)
    assert sorted(f.filename for f in files) == ["ae.safetensors", "flux1-dev.safetensors"]


@responses.activate
def test_an_unlistable_directory_still_yields_a_working_download_url(cfg, session):
    responses.add(responses.GET, "https://huggingface.co/api/models/org/repo", json={})
    responses.add(
        responses.GET, "https://huggingface.co/api/models/org/repo/tree/main", json=[]
    )
    ref = providers.parse("https://huggingface.co/org/repo/resolve/main/model.safetensors")
    files = providers.by_name("huggingface").resolve(ref, cfg, session)
    assert files[0].url.endswith("/resolve/main/model.safetensors")


@responses.activate
def test_a_gated_repo_says_how_to_get_access(cfg, session):
    responses.add(responses.GET, "https://huggingface.co/api/models/org/repo", status=401)
    ref = providers.parse("https://huggingface.co/org/repo/resolve/main/model.safetensors")
    with pytest.raises(AuthRequired, match="accept the model's licence"):
        providers.by_name("huggingface").resolve(ref, cfg, session)


@responses.activate
def test_search_returns_exact_filename_matches_only(cfg, session):
    responses.add(
        responses.GET,
        "https://huggingface.co/api/models",
        json=[
            {
                "id": "org/repo",
                "siblings": [
                    {"rfilename": "wanted.safetensors"},
                    {"rfilename": "unwanted.safetensors"},
                ],
            }
        ],
    )
    files = providers.by_name("huggingface").search("wanted.safetensors", cfg, session)
    assert [f.filename for f in files] == ["wanted.safetensors"]


# -- Civitai -------------------------------------------------------------


@responses.activate
def test_a_civitai_version_resolves_to_its_primary_file(cfg, session):
    responses.add(
        responses.GET,
        "https://civitai.com/api/v1/model-versions/5678",
        json=fixture("civitai_version.json"),
    )
    ref = providers.parse("https://civitai.com/models/1234?modelVersionId=5678")
    files = providers.by_name("civitai").resolve(ref, cfg, session)
    assert len(files) == 1
    file = files[0]
    assert file.filename == "paper-cut.safetensors"
    assert file.size == int(149504.0 * 1024)
    assert file.sha256 == "aa11bb22cc33dd44ee55ff6677889900aa11bb22cc33dd44ee55ff6677889900"
    assert file.meta["model_type"] == "LORA"


@responses.activate
def test_a_model_page_without_a_version_takes_the_newest(cfg, session):
    responses.add(
        responses.GET,
        "https://civitai.com/api/v1/models/1234",
        json={
            "id": 1234,
            "name": "Paper Cut",
            "type": "LORA",
            "modelVersions": [fixture("civitai_version.json")],
        },
    )
    ref = providers.parse("https://civitai.com/models/1234")
    files = providers.by_name("civitai").resolve(ref, cfg, session)
    assert files[0].filename == "paper-cut.safetensors"
    assert files[0].meta["model_type"] == "LORA"


@responses.activate
def test_download_url_query_filters_are_preserved(cfg, session):
    responses.add(
        responses.GET,
        "https://civitai.com/api/v1/model-versions/5678",
        json=fixture("civitai_version.json"),
    )
    ref = providers.parse(
        "https://civitai.com/api/download/models/5678?type=Model&format=SafeTensor&fp=fp16"
    )
    file = providers.by_name("civitai").resolve(ref, cfg, session)[0]
    assert "format=SafeTensor" in file.url and "fp=fp16" in file.url


@responses.activate
def test_the_api_key_goes_in_a_header_not_the_url(cfg, session):
    cfg.civitai_api_key = "secret-key"
    responses.add(
        responses.GET,
        "https://civitai.com/api/v1/model-versions/5678",
        json=fixture("civitai_version.json"),
    )
    ref = providers.parse("https://civitai.com/model-versions/5678")
    providers.by_name("civitai").resolve(ref, cfg, session)
    request = responses.calls[0].request
    assert request.headers["Authorization"] == "Bearer secret-key"
    assert "secret-key" not in request.url
