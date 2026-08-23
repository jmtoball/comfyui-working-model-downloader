"""Missing-slot analysis: reachability, folder derivation, and rename detection."""

from __future__ import annotations

from wmd import analysis, comfy_env


def install_loaders(registry, folder_paths, make_node_class):
    """Register loaders whose combos read from the fake folder_paths, as ComfyUI does."""
    registry["CheckpointLoaderSimple"] = make_node_class(
        {"required": {"ckpt_name": (folder_paths.get_filename_list("checkpoints"),)}}
    )
    registry["LoraLoader"] = make_node_class(
        {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0}),
            }
        }
    )
    registry["VAELoader"] = make_node_class(
        {"required": {"vae_name": (folder_paths.get_filename_list("vae"),)}}
    )
    registry["SaveImage"] = make_node_class(
        {"required": {"images": ("IMAGE",), "filename_prefix": ("STRING", {})}},
        output_node=True,
    )


def test_folder_comes_from_the_loaders_own_option_list(folder_paths, node_registry, make_node_class):
    folder_paths.add_file("loras", "present.safetensors")
    install_loaders(node_registry, folder_paths, make_node_class)
    prompt = {
        "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "missing.safetensors"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
    }
    missing = analysis.missing_slots(prompt)
    assert len(missing) == 1
    assert missing[0].folder == "loras"
    assert missing[0].filename == "missing.safetensors"


def test_a_file_already_on_disk_is_not_missing(folder_paths, node_registry, make_node_class):
    folder_paths.add_file("checkpoints", "sd15.safetensors")
    install_loaders(node_registry, folder_paths, make_node_class)
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd15.safetensors"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
    }
    assert analysis.missing_slots(prompt) == []


def test_nodes_that_cannot_reach_an_output_are_reported_separately(folder_paths, node_registry, make_node_class):
    install_loaders(node_registry, folder_paths, make_node_class)
    prompt = {
        "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "connected.safetensors"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
        "9": {"class_type": "LoraLoader", "inputs": {"lora_name": "orphan.safetensors"}},
    }
    active, inactive = analysis.find_slots(prompt)
    assert [slot.filename for slot in active] == ["connected.safetensors"]
    assert [slot.filename for slot in inactive] == ["orphan.safetensors"]


def test_everything_counts_when_no_output_node_is_recognised(folder_paths, node_registry, make_node_class):
    install_loaders(node_registry, folder_paths, make_node_class)
    node_registry.pop("SaveImage")
    prompt = {"9": {"class_type": "LoraLoader", "inputs": {"lora_name": "orphan.safetensors"}}}
    active, inactive = analysis.find_slots(prompt)
    assert [slot.filename for slot in active] == ["orphan.safetensors"]
    assert inactive == []


def test_a_browser_duplicate_suffix_is_a_rename_not_a_download(folder_paths, node_registry, make_node_class):
    folder_paths.add_file("loras", "style.safetensors")
    install_loaders(node_registry, folder_paths, make_node_class)
    prompt = {
        "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "style (2).safetensors"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
    }
    active, _ = analysis.find_slots(prompt)
    assert active[0].missing is False
    assert active[0].found_as == "style.safetensors"


def test_an_empty_folder_falls_back_to_the_input_name(folder_paths, node_registry, make_node_class):
    # No files anywhere, so no option list can be matched against a folder.
    node_registry["UNETLoader"] = make_node_class(
        {"required": {"unet_name": ([],), "weight_dtype": (["default"],)}}
    )
    prompt = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors"}}}
    missing = analysis.missing_slots(prompt)
    assert [(slot.folder, slot.filename) for slot in missing] == [
        ("diffusion_models", "flux.safetensors")
    ]


def test_ambiguous_input_names_are_left_to_option_matching(folder_paths, node_registry, make_node_class):
    """`clip_name` means two different folders, so a name table must not decide it."""
    folder_paths.add_file("clip_vision", "vit-h.safetensors")
    node_registry["CLIPVisionLoader"] = make_node_class(
        {"required": {"clip_name": (folder_paths.get_filename_list("clip_vision"),)}}
    )
    prompt = {"1": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "other.safetensors"}}}
    assert analysis.missing_slots(prompt)[0].folder == "clip_vision"


def test_non_model_widgets_are_ignored(folder_paths, node_registry, make_node_class):
    install_loaders(node_registry, folder_paths, make_node_class)
    prompt = {
        "1": {
            "class_type": "SaveImage",
            "inputs": {"images": ["2", 0], "filename_prefix": "ComfyUI"},
        }
    }
    assert analysis.missing_slots(prompt) == []


def test_windows_paths_are_normalised(folder_paths, node_registry, make_node_class):
    folder_paths.add_file("loras", "sub/style.safetensors")
    install_loaders(node_registry, folder_paths, make_node_class)
    prompt = {
        "1": {"class_type": "LoraLoader", "inputs": {"lora_name": "sub\\style.safetensors"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
    }
    active, _ = analysis.find_slots(prompt)
    assert active[0].missing is False


def test_extra_model_paths_style_second_root_is_searched(folder_paths, tmp_path):
    """A model present via a second registered root must not be re-downloaded."""
    other = tmp_path / "elsewhere" / "loras"
    other.mkdir(parents=True)
    (other / "shared.safetensors").write_bytes(b"x")
    folder_paths.folder_names_and_paths["loras"][0].append(str(other))
    assert comfy_env.find_existing("loras", "shared.safetensors") == str(
        other / "shared.safetensors"
    )
