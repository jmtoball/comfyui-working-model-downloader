"""Destination heuristics: what decides a folder, and what refuses to."""

from __future__ import annotations

import pytest

from wmd import classify, rules
from wmd.models import (
    PROVIDER_CIVITAI,
    PROVIDER_HUGGINGFACE,
    ModelRef,
    RemoteFile,
    Slot,
)


def civitai_file(model_type, filename="thing.safetensors", size=None):
    return RemoteFile(
        url="https://civitai.com/api/download/models/1",
        filename=filename,
        provider=PROVIDER_CIVITAI,
        size=size,
        meta={"model_type": model_type},
    )


def hf_file(repo_path, filename=None, tags=None, size=None):
    return RemoteFile(
        url=f"https://huggingface.co/org/repo/resolve/main/{repo_path}",
        filename=filename or repo_path.rsplit("/", 1)[-1],
        provider=PROVIDER_HUGGINGFACE,
        size=size,
        meta={"repo_path": repo_path, "tags": tags or []},
    )


@pytest.mark.parametrize(
    "model_type,folder",
    [
        ("Checkpoint", "checkpoints"),
        ("LORA", "loras"),
        ("LoCon", "loras"),
        ("DoRA", "loras"),
        ("LyCORIS", "loras"),
        ("TextualInversion", "embeddings"),
        ("VAE", "vae"),
        ("Controlnet", "controlnet"),
        ("Upscaler", "upscale_models"),
        ("Hypernetwork", "hypernetworks"),
    ],
)
def test_civitai_model_type_decides_the_folder(folder_paths, model_type, folder):
    verdict = classify.classify(ModelRef(raw="x"), civitai_file(model_type))
    assert (verdict.folder, verdict.tier) == (folder, "metadata")


def test_civitai_types_that_are_not_weights_do_not_get_a_folder(folder_paths):
    verdict = classify.classify(ModelRef(raw="x"), civitai_file("Poses", "poses.zip"))
    assert verdict.folder is None
    assert verdict.tier == "unresolved"


@pytest.mark.parametrize(
    "repo_path,folder",
    [
        ("split_files/vae/ae.safetensors", "vae"),
        ("split_files/text_encoders/t5.safetensors", "text_encoders"),
        ("split_files/diffusion_models/flux.safetensors", "diffusion_models"),
        ("vae/diffusion_pytorch_model.safetensors", "vae"),
        ("unet/diffusion_pytorch_model.safetensors", "diffusion_models"),
        ("transformer/model.safetensors", "diffusion_models"),
        ("controlnet/model.safetensors", "controlnet"),
    ],
)
def test_huggingface_repo_layout_decides_the_folder(folder_paths, repo_path, folder):
    verdict = classify.classify(ModelRef(raw="x"), hf_file(repo_path))
    assert (verdict.folder, verdict.tier) == (folder, "metadata")


def test_a_lora_tag_on_the_repo_is_enough(folder_paths):
    verdict = classify.classify(
        ModelRef(raw="x"), hf_file("pytorch_lora_weights.bin", tags=["lora"])
    )
    assert verdict.folder == "loras"


@pytest.mark.parametrize(
    "filename,folder",
    [
        ("model.vae.safetensors", "vae"),
        ("sdxl_vae.safetensors", "vae"),
        ("clip_vision_h.safetensors", "clip_vision"),
        ("t5xxl_fp16.safetensors", "text_encoders"),
        ("clip_l.safetensors", "text_encoders"),
        ("control_v11p_sd15_canny.safetensors", "controlnet"),
        ("some_lora.safetensors", "loras"),
        ("4x-UltraSharp.pth", "upscale_models"),
        ("RealESRGAN_x4plus.pth", "upscale_models"),
        ("my_embedding.pt", "embeddings"),
        # Canonical diffusers filenames, which carry no other signal.
        ("learned_embeds.bin", "embeddings"),
        ("learned_embeds_atkn.safetensors", "embeddings"),
        ("pytorch_lora_weights.safetensors", "loras"),
        # Patterns taken from a corpus of 500+ real Civitai workflows.
        ("ae.safetensors", "vae"),
        ("rife49.pth", "frame_interpolation"),
        ("rife_v4.26.safetensors", "frame_interpolation"),
        ("film_net_fp32.pt", "frame_interpolation"),
        ("depth_anything_v2_vits.pth", "geometry_estimation"),
        ("llava_llama3_fp8_scaled.safetensors", "text_encoders"),
        ("t5-v1_1-xxl-encoder-Q8_0.gguf", "text_encoders"),
        ("umt5-xxl-encoder-Q4_K_S.gguf", "text_encoders"),
        ("hunyuan-video-t2v-720p-Q5_K_M.gguf", "diffusion_models"),
    ],
)
def test_the_filename_is_the_last_resort_before_giving_up(folder_paths, filename, folder):
    verdict = classify.classify(ModelRef(raw="x", filename_hint=filename))
    assert (verdict.folder, verdict.tier) == (folder, "filename")


def test_a_quantised_encoder_is_an_encoder_before_it_is_a_gguf(folder_paths):
    """Rule order matters: the catch-all .gguf rule must not swallow encoders."""
    assert classify.classify(ModelRef(raw="x", filename_hint="umt5-xxl-encoder-Q4_K_S.gguf")).folder == "text_encoders"
    assert classify.classify(ModelRef(raw="x", filename_hint="flux1-dev-Q8_0.gguf")).folder == "diffusion_models"


def test_a_folder_only_some_installs_have_is_skipped_when_absent(folder_paths):
    """`sams` comes from the Impact Pack, so the rule must not fire without it."""
    assert classify.classify(ModelRef(raw="x", filename_hint="sam_vit_b_01ec64.pth")).folder is None
    folder_paths.add_folder("sams")
    assert classify.classify(ModelRef(raw="x", filename_hint="sam_vit_b_01ec64.pth")).folder == "sams"


def test_clip_vision_is_tested_before_clip(folder_paths):
    verdict = classify.classify(ModelRef(raw="x", filename_hint="clip_vision_g.safetensors"))
    assert verdict.folder == "clip_vision"


def test_a_nondescript_name_is_left_unresolved(folder_paths):
    verdict = classify.classify(ModelRef(raw="x", filename_hint="model_v2.safetensors"))
    assert verdict.folder is None
    assert "no rule, metadata or filename signal" in verdict.reason


def test_a_very_large_unlabelled_file_is_guessed_as_a_checkpoint_and_says_so(folder_paths):
    file = RemoteFile(url="u", filename="model_v2.safetensors", size=6 * 1024**3)
    verdict = classify.classify(ModelRef(raw="x"), file)
    assert (verdict.folder, verdict.tier) == ("checkpoints", "size")
    assert "a guess" in verdict.reason


def test_the_slot_beats_every_heuristic(folder_paths):
    """The loader's own combo is a fact; the filename is an inference."""
    ref = ModelRef(raw="x")
    ref.slot = Slot(node_id="1", input_name="unet_name", node_type="UNETLoader", folder="diffusion_models")
    verdict = classify.classify(ref, civitai_file("LORA", "some_lora.safetensors"))
    assert (verdict.folder, verdict.tier) == ("diffusion_models", "slot")


def test_an_explicit_override_beats_the_slot(folder_paths):
    ref = ModelRef(raw="x", folder_override="embeddings")
    ref.slot = Slot(node_id="1", input_name="lora_name", folder="loras")
    assert classify.classify(ref).tier == "override"


def test_a_saved_rule_beats_metadata(folder_paths):
    rules.remember(rules.Rule(pattern="odd_name", field="filename", folder="embeddings"))
    verdict = classify.classify(
        ModelRef(raw="x"), civitai_file("LORA", "odd_name.safetensors")
    )
    assert (verdict.folder, verdict.tier) == ("embeddings", "rule")


def test_a_folder_this_install_does_not_have_is_not_used(folder_paths):
    """`ipadapter` only exists when a custom node registers it."""
    verdict = classify.classify(ModelRef(raw="x", filename_hint="ip-adapter_sd15.safetensors"))
    assert verdict.folder != "ipadapter"

    folder_paths.add_folder("ipadapter")
    verdict = classify.classify(ModelRef(raw="x", filename_hint="ip-adapter_sd15.safetensors"))
    assert verdict.folder == "ipadapter"


def test_the_repository_name_is_used_when_the_filename_says_nothing(folder_paths):
    """`sd-vae-ft-mse/diffusion_pytorch_model.safetensors` is a VAE, and the repo
    name is the only place that is written down."""
    file = RemoteFile(
        url="https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/x.safetensors",
        filename="diffusion_pytorch_model.safetensors",
        provider=PROVIDER_HUGGINGFACE,
        meta={"repo": "stabilityai/sd-vae-ft-mse", "repo_path": "diffusion_pytorch_model.safetensors"},
    )
    verdict = classify.classify(ModelRef(raw="x"), file)
    assert verdict.folder == "vae"
    assert "sd-vae-ft-mse" in verdict.reason


def test_the_filename_still_beats_the_repository_name(folder_paths):
    file = RemoteFile(
        url="u",
        filename="control_v11p_canny.safetensors",
        provider=PROVIDER_HUGGINGFACE,
        meta={"repo": "someone/lora-collection", "repo_path": "control_v11p_canny.safetensors"},
    )
    assert classify.classify(ModelRef(raw="x"), file).folder == "controlnet"


def test_the_workflows_own_directory_hint_is_honoured(folder_paths):
    ref = ModelRef(raw="x", folder_hint="models/unet", filename_hint="thing.safetensors")
    verdict = classify.classify(ref)
    assert (verdict.folder, verdict.tier) == ("diffusion_models", "properties")
