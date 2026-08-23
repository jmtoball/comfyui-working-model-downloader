"""Deciding which ``models/`` folder a file belongs in, and saying why.

Ordered tiers, most authoritative first. Every answer carries the reason that
produced it, so the panel can show its working and the user can tell a fact
("the loader's own combo reads from diffusion_models") from a guess ("the
filename contains 'lora'"). When nothing is authoritative enough, the answer is
*unresolved* -- a wrong folder is worse than an honest question.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass

from . import comfy_env, rules
from .models import PROVIDER_CIVITAI, PROVIDER_HUGGINGFACE, ModelRef, RemoteFile
from .providers.civitai import MODEL_TYPE_FOLDERS

# Directory names used inside HuggingFace repos, mapped onto ComfyUI folder keys.
# The Comfy-Org mirrors publish under `split_files/<key>/`, which is as close to a
# declaration of intent as a repo layout gets.
HF_PATH_FOLDERS = {
    "vae": "vae",
    "vaes": "vae",
    "unet": "diffusion_models",
    "unets": "diffusion_models",
    "transformer": "diffusion_models",
    "diffusion_models": "diffusion_models",
    "text_encoder": "text_encoders",
    "text_encoders": "text_encoders",
    "text_encoder_2": "text_encoders",
    "clip": "text_encoders",
    "clip_vision": "clip_vision",
    "lora": "loras",
    "loras": "loras",
    "controlnet": "controlnet",
    "controlnets": "controlnet",
    "style_models": "style_models",
    "upscale_models": "upscale_models",
    "embeddings": "embeddings",
    "checkpoints": "checkpoints",
    "audio_encoders": "audio_encoders",
    "model_patches": "model_patches",
}

# Filename signals, most specific first: the first hit wins, so `clip_vision`
# must be tested before `clip`, and `control_lora` before the bare `lora`.
FILENAME_RULES: tuple[tuple[str, str], ...] = (
    (r"\.vae\.(safetensors|pt|ckpt|sft)$", "vae"),
    (r"(^|[_\-.])vae([_\-.]|$)", "vae"),
    (r"clip[_\-]?vision", "clip_vision"),
    (r"(^|[_\-.])(t5xxl|umt5|clip_l|clip_g|clip_h|text[_\-]?encoder)", "text_encoders"),
    (r"(control[_\-]?lora)", "controlnet"),
    (r"(control[_\-]?net|^control[_\-]|(^|[_\-.])cnet([_\-.]|$)|t2i[_\-]?adapter)", "controlnet"),
    (r"ip[_\-]?adapter", "ipadapter"),
    (r"(^|[_\-.])(lora|locon|lycoris|dora)([_\-.]|$)", "loras"),
    (r"(esrgan|realesr|swinir|(^|[_\-.])[0-9]+x[_\-]|upscal)", "upscale_models"),
    # Canonical diffusers filenames: these name the kind of weights exactly.
    (r"^learned_embeds(_?[\w.\-]*)?\.", "embeddings"),
    (r"^pytorch_lora_weights\.", "loras"),
    (r"(textual[_\-]?inversion|embedding)", "embeddings"),
    (r"(^|[_\-.])(unet|diffusion[_\-]?model|transformer)([_\-.]|$)", "diffusion_models"),
    (r"(^|[_\-.])(hypernetwork)", "hypernetworks"),
)

# Above this, a file with no other signal is almost certainly a full checkpoint.
_CHECKPOINT_SIZE = 3 * 1024**3


@dataclass
class Verdict:
    folder: str | None
    tier: str
    reason: str


def _valid(folder: str | None) -> str | None:
    """Normalise a folder key and drop it if this ComfyUI has no such folder."""
    if not folder:
        return None
    key = comfy_env.map_legacy(folder)
    return key if comfy_env.is_folder_key(key) else None


def _by_slot(ref: ModelRef) -> Verdict | None:
    if ref.slot is None or not ref.slot.folder:
        return None
    folder = _valid(ref.slot.folder)
    if folder is None:
        return None
    return Verdict(
        folder,
        "slot",
        f"{ref.slot.node_type or 'the loader'} reads {ref.slot.input_name} from {folder}",
    )


def _by_rule(ref: ModelRef, file: RemoteFile | None) -> Verdict | None:
    filename = (file.filename if file else "") or ref.filename_hint or ""
    rule = rules.lookup(filename=filename, url=ref.raw)
    if rule is None or not rule.folder:
        return None
    folder = _valid(rule.folder)
    if folder is None:
        return None
    return Verdict(folder, "rule", f"your saved rule for {rule.pattern!r}")


def _by_hint(ref: ModelRef) -> Verdict | None:
    folder = _valid(ref.folder_hint)
    if folder is None:
        return None
    return Verdict(folder, "properties", f"the workflow declares {folder}")


def _by_civitai(file: RemoteFile) -> Verdict | None:
    model_type = str(file.meta.get("model_type") or "").strip().lower()
    if not model_type:
        return None
    mapped = MODEL_TYPE_FOLDERS.get(model_type.replace(" ", ""))
    folder = _valid(mapped)
    if folder is None:
        return None
    return Verdict(folder, "metadata", f"Civitai calls this a {model_type}")


def _by_hf_path(file: RemoteFile) -> Verdict | None:
    repo_path = str(file.meta.get("repo_path") or "")
    segments = [s.lower() for s in posixpath.dirname(repo_path).split("/") if s]
    for segment in reversed(segments):
        folder = _valid(HF_PATH_FOLDERS.get(segment))
        if folder is not None:
            return Verdict(folder, "metadata", f"the repo stores it under {segment}/")
    return None


def _by_hf_tags(file: RemoteFile) -> Verdict | None:
    tags = {str(tag).lower() for tag in (file.meta.get("tags") or [])}
    library = str(file.meta.get("library_name") or "").lower()
    if "lora" in tags or library == "peft":
        folder = _valid("loras")
        if folder:
            return Verdict(folder, "metadata", "the repo is tagged as a LoRA")
    if "controlnet" in tags:
        folder = _valid("controlnet")
        if folder:
            return Verdict(folder, "metadata", "the repo is tagged as a ControlNet")
    return None


def _by_filename(filename: str) -> Verdict | None:
    name = posixpath.basename(filename or "").lower()
    if not name:
        return None
    for pattern, folder in FILENAME_RULES:
        if re.search(pattern, name):
            valid = _valid(folder)
            if valid is not None:
                return Verdict(valid, "filename", f"the filename looks like a {folder} entry")
    return None


def _by_hf_repo_name(file: RemoteFile) -> Verdict | None:
    """The repository name itself, when nothing more specific said anything.

    Plenty of single-purpose repos publish diffusers-style filenames that carry no
    signal at all -- `sd-vae-ft-mse/diffusion_pytorch_model.safetensors` is a VAE,
    and the only place that is written down is the repo name.
    """
    repo = str(file.meta.get("repo") or "").split("/")[-1].lower()
    if not repo:
        return None
    for pattern, folder in FILENAME_RULES:
        if re.search(pattern, repo):
            valid = _valid(folder)
            if valid is not None:
                return Verdict(valid, "metadata", f"the repository is named {repo!r}")
    return None


def _by_size(file: RemoteFile | None) -> Verdict | None:
    if file is None or not file.size or file.size < _CHECKPOINT_SIZE:
        return None
    folder = _valid("checkpoints")
    if folder is None:
        return None
    gib = file.size / 1024**3
    return Verdict(folder, "size", f"nothing else matched and it is {gib:.1f} GiB (a guess)")


def classify(ref: ModelRef, file: RemoteFile | None = None) -> Verdict:
    """Where this file should go, and why."""
    override = _valid(ref.folder_override)
    if override is not None:
        return Verdict(override, "override", "you chose this folder")

    checks = [_by_slot(ref), _by_rule(ref, file), _by_hint(ref)]
    if file is not None:
        if file.provider == PROVIDER_CIVITAI:
            checks.append(_by_civitai(file))
        if file.provider == PROVIDER_HUGGINGFACE:
            checks.append(_by_hf_path(file))
            checks.append(_by_hf_tags(file))
    checks.append(_by_filename((file.filename if file else "") or ref.filename_hint or ""))
    if file is not None and file.provider == PROVIDER_HUGGINGFACE:
        checks.append(_by_hf_repo_name(file))
    checks.append(_by_size(file))

    for verdict in checks:
        if verdict is not None:
            return verdict

    return Verdict(None, "unresolved", "no rule, metadata or filename signal identified this")
