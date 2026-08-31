"""Transferring one file over several connections.

The reason this exists: measured against the same file, at the same moment,
from the same rented GPU box, a single stream to HuggingFace held 3-29 MB/s
while eight ranged connections aggregated 103 MB/s. Splitting the transfer is
worth real hours on a 60GB model set -- but only if every guard the sequential
path earned still holds, which is what these tests pin down.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading

import pytest
import responses

from wmd import download
from wmd.errors import AuthRequired, DownloadFailed
from wmd.models import RemoteFile

URL = "https://huggingface.co/org/repo/resolve/main/big.safetensors"
# Comfortably over PARALLEL_MIN_SIZE so the parallel path is actually taken.
BIG = bytes(range(256)) * (download.PARALLEL_MIN_SIZE // 256 + 4096)


def make_file(**kwargs):
    defaults = {
        "url": URL,
        "filename": "big.safetensors",
        "provider": "huggingface",
        "size": len(BIG),
        "sha256": hashlib.sha256(BIG).hexdigest(),
    }
    defaults.update(kwargs)
    return RemoteFile(**defaults)


@pytest.fixture
def dest(tmp_path):
    return str(tmp_path / "models" / "big.safetensors")


@pytest.fixture
def parallel_cfg(cfg):
    cfg.prefs.download_connections = 4
    return cfg


def serve_ranges(body: bytes = BIG, *, honour: bool = True, status: int = 206):
    """A server that answers Range requests the way object storage does."""
    seen: list[str] = []

    def handler(request):
        header = request.headers.get("Range", "")
        seen.append(header)
        if not honour:
            return 200, {"Content-Length": str(len(body))}, body
        start, _, end = header.removeprefix("bytes=").partition("-")
        first = int(start or 0)
        last = int(end) if end else len(body) - 1
        chunk = body[first : last + 1]
        headers = {
            "Content-Range": f"bytes {first}-{last}/{len(body)}",
            "Content-Length": str(len(chunk)),
        }
        return status, headers, chunk

    responses.add_callback(responses.GET, URL, callback=handler)
    return seen


@responses.activate
def test_a_large_file_is_fetched_over_several_connections(dest, parallel_cfg):
    seen = serve_ranges()
    outcome = download.download(make_file(), dest, cfg=parallel_cfg)

    assert open(dest, "rb").read() == BIG
    assert outcome.verified
    # One probe plus one request per segment; a single-stream run would be 1.
    assert len(seen) == 1 + parallel_cfg.prefs.download_connections


@responses.activate
def test_the_assembled_file_is_verified_against_its_checksum(dest, parallel_cfg):
    """Segments land out of order, so the hash has to come from the file."""
    serve_ranges(body=BIG[:-1] + b"\x00")
    with pytest.raises(DownloadFailed, match="checksum"):
        download.download(make_file(), dest, cfg=parallel_cfg)
    assert not os.path.exists(dest)
    assert os.path.exists(dest + ".bad")


@responses.activate
def test_a_server_that_ignores_ranges_falls_back_to_one_stream(dest, parallel_cfg):
    """A 200 to a Range request disqualifies the file from parallel transfer."""
    serve_ranges(honour=False)
    outcome = download.download(make_file(), dest, cfg=parallel_cfg)
    assert open(dest, "rb").read() == BIG
    assert outcome.verified
    assert not os.path.exists(dest + ".part.plan")


@responses.activate
def test_an_interrupted_parallel_download_resumes_from_its_plan(dest, parallel_cfg):
    """Each segment records its own progress, so a rerun refetches only the rest."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    connections = parallel_cfg.prefs.download_connections
    spans = download._segments(len(BIG), connections)

    # Simulate a run that finished the first segment and nothing else.
    with open(part, "wb") as handle:
        handle.truncate(len(BIG))
        handle.seek(0)
        handle.write(BIG[spans[0][0] : spans[0][1] + 1])
    done = [spans[0][1] - spans[0][0] + 1] + [0] * (len(spans) - 1)
    download._save_plan(part, len(BIG), connections, done)

    seen = serve_ranges()
    outcome = download.download(make_file(), dest, cfg=parallel_cfg)

    assert open(dest, "rb").read() == BIG
    assert outcome.verified
    # The completed segment is not requested again: probe + remaining segments.
    assert len(seen) == 1 + (len(spans) - 1)
    assert not os.path.exists(part + ".plan")


@responses.activate
def test_a_stale_plan_is_ignored_rather_than_trusted(dest, parallel_cfg):
    """A plan describing a different cut of the file would place bytes wrongly."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    with open(part, "wb") as handle:
        handle.truncate(len(BIG))
    download._save_plan(part, len(BIG) + 999, 2, [10, 20])

    serve_ranges()
    outcome = download.download(make_file(), dest, cfg=parallel_cfg)
    assert open(dest, "rb").read() == BIG
    assert outcome.verified


@responses.activate
def test_credentials_are_reported_from_the_probe_not_a_worker(dest, parallel_cfg):
    """A 401 must surface as AuthRequired, not as a generic transfer failure."""
    responses.add(responses.GET, URL, status=401)
    with pytest.raises(AuthRequired):
        download.download(make_file(), dest, cfg=parallel_cfg)


@responses.activate
def test_cancelling_stops_the_workers_and_removes_the_partial(dest, parallel_cfg):
    serve_ranges()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DownloadFailed, match="cancelled"):
        download.download(make_file(), dest, cfg=parallel_cfg, cancel=cancel)
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")
    assert not os.path.exists(dest + ".part.plan")


@responses.activate
def test_pausing_keeps_the_partial_and_its_plan(dest, parallel_cfg):
    serve_ranges()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(DownloadFailed, match="cancelled"):
        download.download(
            make_file(), dest, cfg=parallel_cfg, cancel=cancel, keep_partial_on_cancel=True
        )
    assert os.path.exists(dest + ".part")
    plan = json.load(open(dest + ".part.plan", encoding="utf-8"))
    assert plan["total"] == len(BIG)


@responses.activate
def test_one_connection_keeps_the_sequential_path(dest, cfg):
    """The default must stay a single stream unless asked otherwise."""
    cfg.prefs.download_connections = 1
    responses.add(responses.GET, URL, body=BIG, status=200)
    outcome = download.download(make_file(), dest, cfg=cfg)
    assert open(dest, "rb").read() == BIG
    assert outcome.verified
    assert len(responses.calls) == 1


@responses.activate
def test_a_small_file_is_not_split(dest, parallel_cfg):
    """Below the threshold the extra requests cost more than they return."""
    body = b"small" * 100
    responses.add(responses.GET, URL, body=body, status=200)
    small = make_file(size=len(body), sha256=hashlib.sha256(body).hexdigest())
    outcome = download.download(small, dest, cfg=parallel_cfg)
    assert outcome.verified
    assert len(responses.calls) == 1


def test_segments_cover_the_file_exactly():
    for total in (1, 2, 1023, 1024, 1_000_003):
        for connections in (1, 2, 3, 8):
            spans = download._segments(total, connections)
            assert spans[0][0] == 0
            assert spans[-1][1] == total - 1
            for (_, end), (start, _) in zip(spans, spans[1:]):
                assert start == end + 1
            assert sum(end - start + 1 for start, end in spans) == total

@responses.activate
def test_a_one_byte_probe_is_not_mistaken_for_an_error_page(dest, parallel_cfg):
    """The probe asks for a single byte, so it always looks "tiny".

    Sniffing that body for text condemned every file whose first byte happened
    to be printable -- which is most of them.
    """
    serve_ranges(body=b"A" + BIG[1:])
    outcome = download.download(
        make_file(sha256=hashlib.sha256(b"A" + BIG[1:]).hexdigest()), dest, cfg=parallel_cfg
    )
    assert outcome.verified


@responses.activate
def test_an_error_page_served_to_a_segment_is_still_caught(dest, parallel_cfg):
    """Moving the sniff off the probe must not lose it altogether."""
    page = b"<html><body>login required</body></html>"

    def handler(request):
        first = int(request.headers["Range"].removeprefix("bytes=").split("-")[0])
        if first == 0 and request.headers["Range"] != "bytes=0-0":
            return 206, {"Content-Range": f"bytes 0-{len(page) - 1}/{len(BIG)}",
                         "Content-Type": "text/html"}, page
        start, _, end = request.headers["Range"].removeprefix("bytes=").partition("-")
        a, b = int(start), int(end or len(BIG) - 1)
        return 206, {"Content-Range": f"bytes {a}-{b}/{len(BIG)}"}, BIG[a : b + 1]

    responses.add_callback(responses.GET, URL, callback=handler)
    with pytest.raises(DownloadFailed, match="web page"):
        download.download(make_file(), dest, cfg=parallel_cfg)
