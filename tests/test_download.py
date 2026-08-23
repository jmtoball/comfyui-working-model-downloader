"""The transfer, and every way it is known to go wrong.

Each test here corresponds to a real failure mode observed in the existing ComfyUI
downloader ecosystem, not a hypothetical one.
"""

from __future__ import annotations

import hashlib
import os
import threading

import pytest
import responses

from wmd import download, http
from wmd.errors import AuthRequired, DownloadFailed
from wmd.models import RemoteFile

URL = "https://huggingface.co/org/repo/resolve/main/model.safetensors"
BODY = b"weights" * 1000


def make_file(**kwargs):
    defaults = {
        "url": URL,
        "filename": "model.safetensors",
        "provider": "huggingface",
        "size": len(BODY),
        "sha256": hashlib.sha256(BODY).hexdigest(),
    }
    defaults.update(kwargs)
    return RemoteFile(**defaults)


@pytest.fixture
def dest(tmp_path):
    return str(tmp_path / "models" / "model.safetensors")



@responses.activate
def test_a_plain_download_verifies_and_lands_atomically(dest, cfg):
    responses.add(responses.GET, URL, body=BODY, status=200)
    outcome = download.download(make_file(), dest, cfg=cfg)
    assert outcome.status == "downloaded" and outcome.verified
    assert open(dest, "rb").read() == BODY
    assert not os.path.exists(dest + ".part")


@responses.activate
def test_an_interrupted_download_resumes_from_the_part_file(dest, cfg):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".part", "wb") as handle:
        handle.write(BODY[:100])
    responses.add(
        responses.GET,
        URL,
        body=BODY[100:],
        status=206,
        headers={"Content-Range": f"bytes 100-{len(BODY) - 1}/{len(BODY)}"},
    )
    outcome = download.download(make_file(), dest, cfg=cfg)
    assert open(dest, "rb").read() == BODY
    assert outcome.verified


@responses.activate
def test_a_server_that_ignores_the_range_restarts_instead_of_corrupting(dest, cfg):
    """A 200 answering a Range request means the body is the *whole* file."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".part", "wb") as handle:
        handle.write(BODY[:100])
    responses.add(responses.GET, URL, body=BODY, status=200)
    download.download(make_file(), dest, cfg=cfg)
    assert open(dest, "rb").read() == BODY


@responses.activate
def test_416_means_the_part_file_is_already_complete(dest, cfg):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".part", "wb") as handle:
        handle.write(BODY)
    responses.add(responses.GET, URL, status=416)
    outcome = download.download(make_file(), dest, cfg=cfg)
    assert outcome.status == "downloaded"
    assert open(dest, "rb").read() == BODY


@responses.activate
def test_a_web_page_is_never_saved_as_a_model(dest, cfg):
    """The classic failure: an error page stored under a .safetensors name."""
    responses.add(
        responses.GET,
        URL,
        body=b"<!DOCTYPE html><html><body>Sign in to continue</body></html>",
        status=200,
        content_type="text/html",
    )
    with pytest.raises(DownloadFailed, match="returned a web page"):
        download.download(make_file(size=None, sha256=None), dest, cfg=cfg)
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


@responses.activate
def test_a_json_error_body_is_rejected_even_when_mislabelled(dest, cfg):
    responses.add(
        responses.GET,
        URL,
        body=b'{"error": "unauthorized"}',
        status=200,
        content_type="application/octet-stream",
    )
    with pytest.raises(DownloadFailed, match="returned a web page"):
        download.download(make_file(size=None, sha256=None), dest, cfg=cfg)


@responses.activate
def test_the_civitai_login_redirect_is_reported_as_a_missing_api_key(dest, cfg):
    civitai_url = "https://civitai.com/api/download/models/5678"
    responses.add(
        responses.GET,
        civitai_url,
        status=307,
        headers={"Location": "https://civitai.com/login?returnUrl=%2F&reason=download-auth"},
    )
    with pytest.raises(AuthRequired, match="Civitai API key"):
        download.download(
            make_file(url=civitai_url, provider="civitai", size=None, sha256=None),
            dest,
            cfg=cfg,
        )


@responses.activate
def test_credentials_are_dropped_on_the_hop_to_object_storage(dest, cfg):
    """Presigned storage rejects a request that also carries an Authorization header."""
    cfg.hf_token = "hf_secret_token_value"
    storage = "https://cdn.example-storage.com/presigned/model.safetensors"
    responses.add(responses.GET, URL, status=302, headers={"Location": storage})
    responses.add(responses.GET, storage, body=BODY, status=200)

    download.download(make_file(), dest, cfg=cfg)

    first, second = responses.calls[0].request, responses.calls[1].request
    assert first.headers["Authorization"] == "Bearer hf_secret_token_value"
    assert "Authorization" not in second.headers


@responses.activate
def test_credentials_survive_a_redirect_within_the_same_host(dest, cfg):
    cfg.hf_token = "hf_secret_token_value"
    other = "https://cdn-lfs.huggingface.co/repo/model.safetensors"
    responses.add(responses.GET, URL, status=302, headers={"Location": other})
    responses.add(responses.GET, other, body=BODY, status=200)
    download.download(make_file(), dest, cfg=cfg)
    assert responses.calls[1].request.headers["Authorization"] == "Bearer hf_secret_token_value"


@responses.activate
def test_a_stale_huggingface_xet_signature_is_refreshed_once(dest, cfg):
    xet = "https://transfer.xethub.hf.co/xorbs/default/abc"
    responses.add(responses.GET, URL, status=302, headers={"Location": xet})
    responses.add(responses.GET, xet, status=403)
    responses.add(responses.GET, URL, body=BODY, status=200)

    outcome = download.download(make_file(), dest, cfg=cfg)
    assert outcome.status == "downloaded"
    assert "wmd_retry=" in responses.calls[2].request.url
    assert responses.calls[2].request.headers["Cache-Control"] == "no-cache"


@responses.activate
def test_a_checksum_mismatch_keeps_the_evidence_and_refuses_the_file(dest, cfg):
    responses.add(responses.GET, URL, body=BODY, status=200)
    file = make_file(sha256="0" * 64)
    with pytest.raises(DownloadFailed, match="failed its checksum"):
        download.download(file, dest, cfg=cfg)
    assert not os.path.exists(dest)
    assert os.path.exists(dest + ".bad")


@responses.activate
def test_a_short_body_is_discarded_rather_than_kept(dest, cfg):
    responses.add(responses.GET, URL, body=BODY, status=200)
    with pytest.raises(DownloadFailed, match="should be"):
        download.download(make_file(size=len(BODY) + 500, sha256=None), dest, cfg=cfg)
    assert not os.path.exists(dest)


@responses.activate
def test_cancelling_removes_the_partial_file(dest, cfg):
    responses.add(responses.GET, URL, body=BODY, status=200)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DownloadFailed, match="cancelled"):
        download.download(make_file(), dest, cfg=cfg, cancel=cancel)
    assert not os.path.exists(dest + ".part")


@responses.activate
def test_pausing_keeps_the_partial_file_so_it_can_resume(dest, cfg):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".part", "wb") as handle:
        handle.write(BODY[:100])
    responses.add(responses.GET, URL, body=BODY[100:], status=206)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DownloadFailed, match="cancelled"):
        download.download(
            make_file(), dest, cfg=cfg, cancel=cancel, keep_partial_on_cancel=True
        )
    assert os.path.getsize(dest + ".part") == 100


@responses.activate
def test_a_file_already_on_disk_is_not_fetched_again(dest, cfg):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as handle:
        handle.write(BODY)
    outcome = download.download(make_file(), dest, cfg=cfg)
    assert outcome.status == "present"
    assert len(responses.calls) == 0


@responses.activate
def test_a_401_says_what_to_do_about_it(dest, cfg):
    responses.add(responses.GET, URL, status=401)
    with pytest.raises(AuthRequired, match="huggingface.co/settings/tokens"):
        download.download(make_file(), dest, cfg=cfg)


@responses.activate
def test_a_transient_failure_is_retried(dest, cfg):
    responses.add(responses.GET, URL, status=503)
    responses.add(responses.GET, URL, body=BODY, status=200)
    assert download.download(make_file(), dest, cfg=cfg).status == "downloaded"


@responses.activate
def test_a_full_disk_fails_before_anything_is_written(dest, cfg, monkeypatch):
    class Usage:
        free = 1024

    monkeypatch.setattr("wmd.download.shutil.disk_usage", lambda _path: Usage)
    responses.add(responses.GET, URL, body=BODY, status=200)
    with pytest.raises(DownloadFailed, match="not enough free space"):
        download.download(make_file(size=10 * 1024**3), dest, cfg=cfg)
    assert len(responses.calls) == 0


def test_the_same_host_check_treats_subdomains_as_the_origin():
    assert http.same_host("https://cdn-lfs.huggingface.co/x", "https://huggingface.co/y")
    assert not http.same_host("https://evil.com/x", "https://huggingface.co/y")


@responses.activate
def test_progress_is_reported_as_bytes_arrive(dest, cfg):
    responses.add(responses.GET, URL, body=BODY, status=200)
    seen: list[int] = []
    download.download(make_file(), dest, cfg=cfg, progress=lambda done, _total: seen.append(done))
    assert seen and seen[-1] == len(BODY)


def test_tokens_never_appear_in_a_download_url(cfg):
    """Civitai accepts ?token= but we use a header, so keys cannot leak via a URL."""
    cfg.civitai_api_key = "civitai_secret"
    headers = download.auth_headers("civitai", cfg)
    assert headers == {"Authorization": "Bearer civitai_secret"}
