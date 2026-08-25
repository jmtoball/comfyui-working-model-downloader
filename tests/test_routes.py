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

from wmd.models import RemoteFile

pytest.importorskip("aiohttp", reason="ComfyUI ships aiohttp; the routes need it")


class FakeRequest:
    def __init__(self, body=None, match_info=None, query=None):
        self._body = body or {}
        self.match_info = match_info or {}
        self.query = query or {}
        self.can_read_body = body is not None

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


# -- state that belongs to one workflow, not to the panel ---------------------


def a_workflow(loaded=False):
    """A note documenting one model, and a loader that wants it."""
    return {
        "workflow": {
            "nodes": [
                {
                    "id": 1,
                    "type": "Note",
                    "widgets_values": [
                        "https://huggingface.co/org/repo/resolve/main/style.safetensors"
                    ],
                },
                {"id": 2, "type": "LoraLoader", "widgets_values": ["style.safetensors"]},
            ]
        },
        "prompt": {
            "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "style.safetensors"}},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
        },
    }


def mock_huggingface():
    """The two calls resolving a single-file HuggingFace link makes."""
    responses.add(responses.GET, "https://huggingface.co/api/models/org/repo", json={"tags": []})
    responses.add(
        responses.GET,
        "https://huggingface.co/api/models/org/repo/tree/main",
        json=[
            {
                "type": "file",
                "path": "style.safetensors",
                "size": 4096,
                "lfs": {"oid": "b" * 64, "size": 4096},
            }
        ],
    )


@pytest.fixture
def lora_workflow(folder_paths, node_registry, make_node_class):
    node_registry["LoraLoader"] = make_node_class(
        {"required": {"lora_name": (folder_paths.get_filename_list("loras"),)}}
    )
    node_registry["SaveImage"] = make_node_class(
        {"required": {"images": ("IMAGE",)}}, output_node=True
    )
    return folder_paths


@responses.activate
def test_a_model_already_on_disk_is_still_offered(routes, lora_workflow):
    """Once downloaded its slot is no longer missing, but you must still be able
    to pin it -- otherwise finishing a download loses the ability to record it."""
    mock_huggingface()
    lora_workflow.add_file("loras", "style.safetensors")
    body = payload(run(routes.handle_resolve, FakeRequest(a_workflow())))
    assert [item["filename"] for item in body["items"]] == ["style.safetensors"]
    item = body["items"][0]
    assert item["resolved"] is True
    assert item["existing_path"] is not None


@responses.activate
def test_a_downloaded_model_keeps_the_folder_its_loader_asked_for(routes, lora_workflow):
    """Without the present slots the folder would fall back to a filename guess."""
    mock_huggingface()
    lora_workflow.add_file("loras", "style.safetensors")
    body = payload(run(routes.handle_resolve, FakeRequest(a_workflow())))
    assert body["items"][0]["folder"] == "loras"
    assert body["items"][0]["reason"] == "LoraLoader reads lora_name from loras"


@responses.activate
def test_present_models_can_be_excluded_when_asked(routes, lora_workflow):
    mock_huggingface()
    lora_workflow.add_file("loras", "style.safetensors")
    request = FakeRequest({**a_workflow(), "include_present": False})
    body = payload(run(routes.handle_resolve, request))
    # The documented link still resolves; it simply has no slot informing it.
    assert body["items"][0]["reason"] != "LoraLoader reads lora_name from loras"


def test_the_queue_is_scoped_to_the_workflow_that_started_it(routes, extension, folder_paths):
    manager = extension.wmd.jobs.manager()
    for workflow in ("workflows/one.json", "workflows/two.json"):
        run(
            routes.handle_download,
            FakeRequest(
                {
                    "workflow_key": workflow,
                    "items": [
                        {
                            "url": f"https://example.com/{workflow.split('/')[-1]}.safetensors",
                            "filename": f"{workflow.split('/')[-1]}.safetensors",
                            "folder": "loras",
                        }
                    ],
                }
            ),
        )

    first = payload(run(routes.handle_jobs, FakeRequest(query={"workflow": "workflows/one.json"})))
    assert [job["filename"] for job in first["jobs"]] == ["one.json.safetensors"]

    everything = payload(run(routes.handle_jobs, FakeRequest(query={"all": "1"})))
    assert len(everything["jobs"]) == 2
    assert len(manager.list()) == 2


def test_clearing_only_touches_the_asking_workflow(routes, extension, folder_paths):
    manager = extension.wmd.jobs.manager()
    for workflow in ("workflows/one.json", "workflows/two.json"):
        run(
            routes.handle_download,
            FakeRequest(
                {
                    "workflow_key": workflow,
                    "items": [
                        {
                            "url": f"https://example.com/{workflow.split('/')[-1]}.safetensors",
                            "filename": f"{workflow.split('/')[-1]}.safetensors",
                            "folder": "loras",
                        }
                    ],
                }
            ),
        )
    for job in manager.list():
        job.status = extension.wmd.jobs.FAILED

    cleared = payload(
        run(routes.handle_clear_jobs, FakeRequest({"workflow_key": "workflows/one.json"}))
    )
    assert cleared["cleared"] == 1
    assert [job.workflow for job in manager.list()] == ["workflows/two.json"]


def test_a_running_job_is_never_cleared(routes, extension, folder_paths):
    manager = extension.wmd.jobs.manager()
    manager.submit(
        RemoteFile(url="https://example.com/a.safetensors", filename="a.safetensors"),
        dest="/tmp/a.safetensors",
        folder="loras",
        workflow="workflows/one.json",
    )
    for job in manager.list():
        job.status = extension.wmd.jobs.RUNNING
    assert manager.clear_finished("workflows/one.json") == 0


def test_a_download_with_no_workflow_is_visible_under_every_one(routes, extension, folder_paths):
    """A queued prompt starts downloads without a workflow key. Hiding those would
    leave a transfer nobody can see or cancel."""
    manager = extension.wmd.jobs.manager()
    manager.submit(
        RemoteFile(url="https://example.com/node.safetensors", filename="node.safetensors"),
        dest="/tmp/node.safetensors",
        folder="loras",
        source="prompt",
    )
    run(
        routes.handle_download,
        FakeRequest(
            {
                "workflow_key": "workflows/one.json",
                "items": [
                    {
                        "url": "https://example.com/panel.safetensors",
                        "filename": "panel.safetensors",
                        "folder": "loras",
                    }
                ],
            }
        ),
    )

    for workflow in ("workflows/one.json", "workflows/two.json"):
        names = {
            job["filename"]
            for job in payload(run(routes.handle_jobs, FakeRequest(query={"workflow": workflow})))[
                "jobs"
            ]
        }
        assert "node.safetensors" in names, workflow
    # The panel's own download stays with the workflow that asked for it.
    other = payload(run(routes.handle_jobs, FakeRequest(query={"workflow": "workflows/two.json"})))
    assert "panel.safetensors" not in {job["filename"] for job in other["jobs"]}
