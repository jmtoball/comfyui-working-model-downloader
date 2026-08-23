"""The panel's HTTP surface.

Handlers are exercised directly with a stub request: the routing itself is
aiohttp's job, what matters here is that the panel gets the right data and that
credentials never travel back to it.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import responses

pytest.importorskip("aiohttp", reason="ComfyUI ships aiohttp; the routes need it")


class FakeRequest:
    def __init__(self, body=None, match_info=None):
        self._body = body or {}
        self.match_info = match_info or {}

    async def json(self):
        return self._body


def run(handler, *args, **kwargs):
    return asyncio.run(handler(*args, **kwargs))


def payload(response):
    return json.loads(response.body.decode("utf-8"))


@pytest.fixture
def routes(extension):
    return extension.routes


def test_the_route_table_covers_what_the_panel_needs(routes):
    paths = {(method, path) for method, path, _ in routes.ROUTES}
    for expected in [
        ("GET", "/config"),
        ("POST", "/config"),
        ("GET", "/folders"),
        ("POST", "/scan"),
        ("POST", "/resolve"),
        ("POST", "/search"),
        ("POST", "/download"),
        ("GET", "/jobs"),
        ("POST", "/manifest"),
    ]:
        assert expected in paths


def test_the_config_endpoint_never_returns_a_plaintext_key(routes, folder_paths):
    run(routes.handle_post_config, FakeRequest({"hf_token": "hf_supersecretvalue"}))
    body = payload(run(routes.handle_get_config, FakeRequest()))
    assert body["hf_token_set"] is True
    assert "hf_supersecretvalue" not in json.dumps(body)
    assert body["hf_token_hint"] == "…alue"


def test_the_config_endpoint_ignores_fields_it_does_not_own(routes, folder_paths):
    run(routes.handle_post_config, FakeRequest({"hf_token": "x", "admin": True}))
    body = payload(run(routes.handle_get_config, FakeRequest()))
    assert "admin" not in body


def test_folders_come_from_the_live_registry(routes, folder_paths):
    body = payload(run(routes.handle_folders, FakeRequest()))
    keys = [folder["key"] for folder in body["folders"]]
    assert "loras" in keys and "custom_nodes" not in keys


def test_scanning_returns_documented_links_and_missing_slots(routes, folder_paths, node_registry, make_node_class):
    node_registry["LoraLoader"] = make_node_class(
        {"required": {"lora_name": (folder_paths.get_filename_list("loras"),)}}
    )
    request = FakeRequest(
        {
            "workflow": {
                "nodes": [
                    {
                        "id": 1,
                        "type": "Note",
                        "widgets_values": ["https://example.com/style.safetensors"],
                    }
                ]
            },
            "prompt": {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "style.safetensors"}}},
        }
    )
    body = payload(run(routes.handle_scan, request))
    assert [item["url"] for item in body["documented"]] == ["https://example.com/style.safetensors"]
    assert [item["value"] for item in body["missing"]] == ["style.safetensors"]


@responses.activate
def test_resolving_reports_the_destination_and_the_reason(routes, folder_paths):
    body = payload(
        run(routes.handle_resolve, FakeRequest({"urls": "https://example.com/some_lora.safetensors"}))
    )
    item = body["items"][0]
    assert item["folder"] == "loras"
    assert item["tier"] == "manual"
    assert "filename looks like" in item["reason"]
    assert item["dest_path"].endswith("some_lora.safetensors")


@responses.activate
def test_a_panel_override_wins_over_the_heuristic(routes, folder_paths):
    url = "https://example.com/some_lora.safetensors"
    body = payload(
        run(
            routes.handle_resolve,
            FakeRequest({"urls": url, "overrides": [{"source_url": url, "folder": "embeddings"}]}),
        )
    )
    assert body["items"][0]["folder"] == "embeddings"
    assert body["items"][0]["tier"] == "manual"


def test_search_needs_something_to_search_for(routes, folder_paths):
    response = run(routes.handle_search, FakeRequest({"filename": ""}))
    assert response.status == 400


@responses.activate
def test_downloading_queues_the_selected_items(routes, extension, folder_paths):
    url = "https://example.com/style.safetensors"
    responses.add(responses.GET, url, body=b"weights", status=200)
    body = payload(
        run(
            routes.handle_download,
            FakeRequest({"items": [{"url": url, "filename": "style.safetensors", "folder": "loras"}]}),
        )
    )
    assert len(body["jobs"]) == 1
    assert body["jobs"][0]["folder"] == "loras"


def test_downloading_rejects_an_item_it_cannot_place(routes, folder_paths):
    response = run(routes.handle_download, FakeRequest({"items": [{"url": "https://x/y.safetensors"}]}))
    assert response.status == 400
    assert "filename" in payload(response)["error"]


def test_the_manifest_endpoint_returns_what_the_node_should_carry(routes, folder_paths):
    body = payload(
        run(
            routes.handle_manifest,
            FakeRequest(
                {
                    "items": [
                        {
                            "url": "https://example.com/style.safetensors",
                            "filename": "style.safetensors",
                            "folder": "loras",
                        }
                    ]
                }
            ),
        )
    )
    assert body["node_type"] == "WMD_ModelDownloader"
    document = json.loads(body["manifest"])
    assert document["entries"][0]["folder"] == "loras"


def test_a_rule_learned_in_the_panel_is_remembered(routes, folder_paths):
    run(
        routes.handle_post_rules,
        FakeRequest({"pattern": "odd_name", "field": "filename", "folder": "embeddings"}),
    )
    body = payload(run(routes.handle_get_rules, FakeRequest()))
    assert body["rules"][0]["folder"] == "embeddings"


def test_an_unknown_job_action_is_refused(routes, folder_paths):
    request = FakeRequest(match_info={"job_id": "wmd-1", "action": "detonate"})
    response = run(routes.handle_job_action, request)
    assert response.status == 400


def test_job_actions_on_an_unknown_job_are_harmless(routes, folder_paths):
    request = FakeRequest(match_info={"job_id": "nope", "action": "cancel"})
    assert payload(run(routes.handle_job_action, request)) == {"ok": False}
