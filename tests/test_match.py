"""Joining documented links to the slots that are actually missing."""

from __future__ import annotations

from wmd import match, providers
from wmd.models import RemoteFile, Slot


def slot(value, folder="loras", node_id="1", input_name="lora_name"):
    return Slot(
        node_id=node_id,
        input_name=input_name,
        node_type="LoraLoader",
        value=value,
        folder=folder,
    )


def test_a_link_whose_url_names_the_missing_file_is_bound_to_it():
    ref = providers.parse(
        "https://huggingface.co/org/repo/resolve/main/style.safetensors"
    )
    bound, unmatched = match.match_refs_to_slots([ref], [slot("style.safetensors")])
    assert unmatched == []
    assert bound[0].slot is not None
    assert bound[0].filename_override == "style.safetensors"
    assert bound[0].folder_hint == "loras"


def test_a_filename_documented_next_to_a_link_binds_it():
    ref = providers.parse("https://civitai.com/models/1234")
    ref.nearby_names = ["paper-cut.safetensors"]
    bound, unmatched = match.match_refs_to_slots([ref], [slot("paper-cut.safetensors")])
    assert unmatched == []
    assert bound[0].filename_override == "paper-cut.safetensors"


def test_the_slot_dictates_the_subfolder_it_expects():
    ref = providers.parse("https://huggingface.co/org/repo/resolve/main/style.safetensors")
    bound, _ = match.match_refs_to_slots([ref], [slot("SDXL/style.safetensors")])
    assert bound[0].filename_override == "SDXL/style.safetensors"


def test_two_candidates_for_one_slot_are_left_alone():
    """An ambiguity is the user's call, not ours."""
    first = providers.parse("https://huggingface.co/a/one/resolve/main/style.safetensors")
    second = providers.parse("https://huggingface.co/b/two/resolve/main/style.safetensors")
    bound, unmatched = match.match_refs_to_slots([first, second], [slot("style.safetensors")])
    assert bound == []
    assert [s.filename for s in unmatched] == ["style.safetensors"]


def test_a_slot_with_no_documented_source_stays_unmatched():
    ref = providers.parse("https://huggingface.co/org/repo/resolve/main/other.safetensors")
    bound, unmatched = match.match_refs_to_slots([ref], [slot("style.safetensors")])
    assert bound == []
    assert len(unmatched) == 1


def test_each_link_is_used_for_at_most_one_slot():
    ref = providers.parse("https://huggingface.co/org/repo/resolve/main/style.safetensors")
    bound, unmatched = match.match_refs_to_slots(
        [ref], [slot("style.safetensors", node_id="1"), slot("style.safetensors", node_id="2")]
    )
    assert len(bound) == 1
    assert len(unmatched) == 1


def test_the_second_pass_uses_filenames_only_resolution_could_know():
    """A Civitai model page reveals nothing until the API answers."""
    ref = providers.parse("https://civitai.com/models/1234")
    file = RemoteFile(url="https://civitai.com/api/download/models/5678", filename="found.safetensors")
    bound, unmatched = match.match_files_to_slots([(ref, file)], [slot("found.safetensors")])
    assert unmatched == []
    assert bound[0].filename_override == "found.safetensors"


def test_duplicate_suffixes_still_match():
    assert match.names_match("style (2).safetensors", "style.safetensors")
    assert not match.names_match("style.safetensors", "other.safetensors")
