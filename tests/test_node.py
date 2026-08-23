"""The node: the queue-time gate, and replaying a pinned manifest.

The node itself is deliberately dumb -- it downloads what the panel pinned. What
is worth testing is the ordering guarantee: that the files exist before any other
node in the graph runs, and that a failure stops the run with a readable message
instead of surfacing later as a confused loader error.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time

import pytest
import responses

URL = "https://huggingface.co/org/repo/resolve/main/style.safetensors"
BODY = b"weights" * 100
DIGEST = hashlib.sha256(BODY).hexdigest()


def entry(**overrides):
    base = {
        "url": URL,
        "filename": "style.safetensors",
        "folder": "loras",
        "provider": "huggingface",
        "sha256": DIGEST,
        "size": len(BODY),
    }
    base.update(overrides)
    return base


def a_manifest(*entries):
    return json.dumps({"version": 1, "entries": list(entries) or [entry()]})


@pytest.fixture
def node(extension):
    return extension.NODE_CLASS_MAPPINGS["WMD_ModelDownloader"]


@pytest.fixture
def jobs(extension):
    """The job manager the node actually uses (see the `extension` fixture)."""
    return extension.wmd.jobs


def wait_for(jobs_module, job_ids, timeout=5.0):
    manager = jobs_module.manager()
    deadline = time.monotonic() + timeout
    found = []
    while time.monotonic() < deadline:
        found = [manager.get(job_id) for job_id in job_ids]
        if all(job is not None and job.terminal for job in found):
            return found
        time.sleep(0.02)
    raise AssertionError(f"jobs did not finish: {[job and job.status for job in found]}")


def validate(node, **kwargs):
    """Call VALIDATE_INPUTS in whichever flavour this ComfyUI installed."""
    result = node.VALIDATE_INPUTS(**kwargs)
    return asyncio.run(result) if inspect.isawaitable(result) else result


# -- registration --------------------------------------------------------


def test_the_node_is_registered_under_a_readable_name(extension):
    assert extension.NODE_DISPLAY_NAME_MAPPINGS["WMD_ModelDownloader"] == "Working Model Downloader"
    assert extension.WEB_DIRECTORY == "web"


def test_the_manifest_widget_is_the_nodes_only_model_configuration(node):
    required = node.INPUT_TYPES()["required"]
    # Everything besides the manifest is policy: the panel decides *what* to
    # download, the node only decides how strictly to enforce it.
    assert set(required) == {"manifest", "enforce", "on_failure", "verify_hash", "queue_timeout"}


def test_the_passthrough_type_accepts_any_connection(extension):
    """So the node can be spliced into a link when explicit ordering is wanted."""
    assert (extension.wmd_nodes.ANY != "IMAGE") is False
    assert (extension.wmd_nodes.ANY != "LATENT") is False


# -- queueing ------------------------------------------------------------


@responses.activate
def test_queueing_starts_the_downloads_a_manifest_pins(extension, jobs, folder_paths):
    responses.add(responses.GET, URL, body=BODY, status=200)
    job_ids = extension.wmd_nodes.start_downloads(a_manifest())
    assert len(job_ids) == 1
    assert wait_for(jobs, job_ids)[0].status == jobs.DONE
    assert folder_paths.get_filename_list("loras") == ["style.safetensors"]


@responses.activate
def test_a_model_already_on_disk_is_not_queued(extension, folder_paths):
    folder_paths.add_file("loras", "style.safetensors")
    assert extension.wmd_nodes.start_downloads(a_manifest()) == []
    assert len(responses.calls) == 0


def test_a_broken_manifest_does_not_break_queueing(extension, folder_paths):
    assert extension.wmd_nodes.start_downloads("{ not json") == []


@responses.activate
def test_remembered_jobs_are_requeued_once_the_manager_forgets_them(extension, jobs, folder_paths):
    responses.add(responses.GET, URL, body=BODY, status=200)
    first = extension.wmd_nodes.start_downloads(a_manifest())
    wait_for(jobs, first)
    jobs.manager().clear_finished()
    # The ids are still remembered but no longer resolve, so waiting on them would
    # wait on nothing. They must be treated as "not started".
    assert extension.wmd_nodes._job_ids_for(a_manifest()) == []


@responses.activate
def test_the_prompt_hook_starts_downloads_for_this_node_only(extension, jobs, folder_paths):
    responses.add(responses.GET, URL, body=BODY, status=200)
    extension.routes.on_prompt(
        {
            "prompt": {
                "1": {"class_type": "KSampler", "inputs": {"seed": 1}},
                "2": {
                    "class_type": "WMD_ModelDownloader",
                    "inputs": {"manifest": a_manifest(), "enforce": "before_execution"},
                },
            }
        }
    )
    queued = jobs.manager().list()
    assert [job.filename for job in queued] == ["style.safetensors"]
    wait_for(jobs, [job.id for job in queued])


def test_the_hook_leaves_the_other_mode_to_the_node(extension, jobs, folder_paths):
    extension.routes.on_prompt(
        {
            "prompt": {
                "2": {
                    "class_type": "WMD_ModelDownloader",
                    "inputs": {"manifest": a_manifest(), "enforce": "on_node_execution"},
                }
            }
        }
    )
    assert jobs.manager().list() == []


def test_a_malformed_prompt_never_breaks_queueing(extension, jobs, folder_paths):
    extension.routes.on_prompt({"prompt": "not a graph"})
    extension.routes.on_prompt({})
    assert jobs.manager().list() == []


# -- the queue-time gate -------------------------------------------------


@responses.activate
def test_validation_waits_for_the_downloads_before_anything_executes(node, folder_paths):
    responses.add(responses.GET, URL, body=BODY, status=200)
    assert validate(node, manifest=a_manifest(), enforce="before_execution") is True
    # The file is on disk by the time validation returns, so no loader further down
    # the graph can run before it exists.
    assert folder_paths.get_filename_list("loras") == ["style.safetensors"]


@responses.activate
def test_a_failed_download_is_reported_as_a_validation_error(node, folder_paths):
    responses.add(responses.GET, URL, status=404)
    result = validate(node, manifest=a_manifest(), enforce="before_execution", on_failure="error")
    assert isinstance(result, str)
    assert "loras/style.safetensors" in result


@responses.activate
def test_a_failed_download_can_be_downgraded_to_a_warning(node, folder_paths):
    responses.add(responses.GET, URL, status=404)
    assert validate(node, manifest=a_manifest(), enforce="before_execution", on_failure="warn") is True


def test_a_malformed_manifest_fails_validation_with_a_readable_message(node, folder_paths):
    result = validate(node, manifest="{ not json", enforce="before_execution")
    assert isinstance(result, str) and "not valid JSON" in result


def test_an_empty_manifest_validates_without_doing_anything(node, folder_paths):
    assert validate(node, manifest='{"version": 1, "entries": []}') is True


def test_the_other_mode_does_not_gate_at_queue_time(node, jobs, folder_paths):
    assert validate(node, manifest=a_manifest(), enforce="on_node_execution") is True
    assert jobs.manager().list() == []


def test_the_gate_is_installed_in_whichever_flavour_comfyui_can_await(extension, monkeypatch):
    """An async wait keeps the event loop serving; a sync one still waits."""
    import sys
    import types

    module = types.ModuleType("execution")

    async def validate_inputs(*args, **kwargs):  # pragma: no cover - shape only
        return None

    module.validate_inputs = validate_inputs
    monkeypatch.setitem(sys.modules, "execution", module)
    assert extension.wmd_nodes.supports_async_validation() is True

    module.validate_inputs = lambda *args, **kwargs: None
    assert extension.wmd_nodes.supports_async_validation() is False


@responses.activate
def test_the_gate_gives_up_after_the_configured_timeout(node, folder_paths, monkeypatch):
    responses.add(responses.GET, URL, body=BODY, status=200)
    monkeypatch.setattr(node, "_finish", classmethod(lambda cls, ids, on_failure: True))
    monkeypatch.setattr("wmd_extension.wmd_nodes._outstanding", lambda job_ids: ["busy"])
    result = validate(node, manifest=a_manifest(), enforce="before_execution", queue_timeout=1)
    assert isinstance(result, str) and "queue_timeout" in result


# -- execution -----------------------------------------------------------


@responses.activate
def test_executing_in_node_mode_downloads_and_reports(node, folder_paths):
    responses.add(responses.GET, URL, body=BODY, status=200)
    report, passthrough = node().run(
        manifest=a_manifest(), enforce="on_node_execution", passthrough="x"
    )
    assert "1 downloaded" in report
    assert "loras/style.safetensors" in report
    assert passthrough == "x"
    assert folder_paths.get_filename_list("loras") == ["style.safetensors"]


def test_execution_stops_the_run_when_a_pinned_file_never_arrived(node, folder_paths):
    with pytest.raises(RuntimeError, match="style.safetensors"):
        node().run(manifest=a_manifest(), enforce="before_execution", on_failure="error")


def test_execution_reports_success_when_the_files_are_there(node, folder_paths):
    folder_paths.add_file("loras", "style.safetensors")
    report, _ = node().run(manifest=a_manifest(), enforce="before_execution")
    assert "1 present" in report


def test_is_changed_tracks_the_manifest_and_the_files_on_disk(node, folder_paths):
    before = node.IS_CHANGED(manifest=a_manifest())
    folder_paths.add_file("loras", "style.safetensors")
    assert node.IS_CHANGED(manifest=a_manifest()) != before
    assert node.IS_CHANGED(manifest=a_manifest(entry(url="https://other/x"))) != before
