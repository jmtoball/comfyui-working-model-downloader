"""URL parsing: every shape these two sites hand people, plus the junk."""

from __future__ import annotations

import pytest

from wmd import providers, sources
from wmd.models import PROVIDER_CIVITAI, PROVIDER_DIRECT, PROVIDER_HUGGINGFACE

HF = "https://huggingface.co"


@pytest.mark.parametrize(
    "url,repo,revision,path,is_tree",
    [
        (f"{HF}/org/repo/resolve/main/model.safetensors", "org/repo", "main", "model.safetensors", False),
        (f"{HF}/org/repo/resolve/main/model.safetensors?download=true", "org/repo", "main", "model.safetensors", False),
        (f"{HF}/org/repo/blob/main/sub/model.safetensors", "org/repo", "main", "sub/model.safetensors", False),
        (f"{HF}/org/repo/resolve/v1.0/model.safetensors", "org/repo", "v1.0", "model.safetensors", False),
        (f"{HF}/org/repo/resolve/refs%2Fpr%2F3/model.safetensors", "org/repo", "refs/pr/3", "model.safetensors", False),
        (f"{HF}/org/repo/tree/main/split_files/vae", "org/repo", "main", "split_files/vae", True),
        (f"{HF}/org/repo", "org/repo", "main", None, True),
        ("https://hf.co/org/repo/blob/main/model.safetensors", "org/repo", "main", "model.safetensors", False),
        ("org/repo", "org/repo", "main", None, True),
        ("org/repo/blob/main/model.safetensors", "org/repo", "main", "model.safetensors", False),
        ("hf://org/repo/resolve/main/model.safetensors", "org/repo", "main", "model.safetensors", False),
    ],
)
def test_huggingface_urls(url, repo, revision, path, is_tree):
    ref = providers.parse(url)
    assert ref is not None and ref.provider == PROVIDER_HUGGINGFACE
    assert (ref.repo, ref.revision, ref.path, ref.is_tree) == (repo, revision, path, is_tree)


def test_huggingface_dataset_repo_type():
    ref = providers.parse(f"{HF}/datasets/org/repo/resolve/main/model.safetensors")
    assert ref is not None and ref.query["repo_type"] == "dataset"


@pytest.mark.parametrize(
    "url,model_id,version_id,query",
    [
        ("https://civitai.com/models/1234", "1234", None, {}),
        ("https://civitai.com/models/1234/some-slug", "1234", None, {}),
        ("https://civitai.com/models/1234?modelVersionId=5678", "1234", "5678", {}),
        ("https://civitai.red/models/1234?modelVersionId=5678", "1234", "5678", {}),
        ("https://civitai.green/model-versions/5678", None, "5678", {}),
        ("https://civitai.com/api/download/models/5678", None, "5678", {}),
        (
            "https://civitai.com/api/download/models/5678?type=Model&format=SafeTensor&fp=fp16",
            None,
            "5678",
            {"type": "Model", "format": "SafeTensor", "fp": "fp16"},
        ),
        ("https://civitai.com/api/v1/model-versions/5678", None, "5678", {}),
        ("urn:air:sd1:lora:civitai:1234@5678", "1234", "5678", {}),
    ],
)
def test_civitai_urls(url, model_id, version_id, query):
    ref = providers.parse(url)
    assert ref is not None and ref.provider == PROVIDER_CIVITAI
    assert (ref.model_id, ref.version_id) == (model_id, version_id)
    assert ref.query == query


def test_air_urn_carries_a_folder_hint():
    ref = providers.parse("urn:air:sd1:lora:civitai:1234@5678")
    assert ref is not None and ref.folder_hint == "loras"


def test_direct_url_only_when_it_names_a_model_file():
    assert providers.parse("https://example.com/files/model.safetensors").provider == PROVIDER_DIRECT
    assert providers.parse("https://example.com/blog/post") is None


@pytest.mark.parametrize(
    "junk",
    ["", "   ", "not a url", "ftp://example.com/model.safetensors", "model.safetensors"],
)
def test_junk_is_rejected(junk):
    assert providers.parse(junk) is None


def test_paste_box_accepts_comments_and_bare_repo_ids():
    refs = sources.refs_from_input(
        """
        # required models
        org/repo  # the whole repo
        https://civitai.com/models/1234?modelVersionId=5678
        """
    )
    assert [ref.provider for ref in refs] == [PROVIDER_HUGGINGFACE, PROVIDER_CIVITAI]


def test_duplicate_references_collapse_and_merge_hints():
    first = providers.parse("https://civitai.com/models/1234?modelVersionId=5678")
    first.nearby_names = ["a.safetensors"]
    second = providers.parse("https://civitai.red/models/1234?modelVersionId=5678")
    second.nearby_names = ["b.safetensors"]
    merged = sources.dedupe([first, second])
    assert len(merged) == 1
    assert merged[0].nearby_names == ["a.safetensors", "b.safetensors"]
