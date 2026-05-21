from __future__ import annotations

from typing import Any


MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "pi0": {
        "modelFamily": "pi0",
        "kind": "pretrained_checkpoint",
        "sourceType": "public_model_repo",
        "uri": "gs://openpi-assets/checkpoints/pi0_base",
        "checkpointFormat": "openpi_checkpoint",
        "provider": "openpi",
    },
    "pi05": {
        "modelFamily": "pi05",
        "kind": "pretrained_checkpoint",
        "sourceType": "public_model_repo",
        "uri": "gs://openpi-assets/checkpoints/pi05_base",
        "checkpointFormat": "openpi_checkpoint",
        "provider": "openpi",
    },
    "smolvla": {
        "modelFamily": "smolvla",
        "kind": "pretrained_checkpoint",
        "sourceType": "public_model_repo",
        "uri": "hf://lerobot/smolvla_base",
        "checkpointFormat": "lerobot_pretrained_model",
        "provider": "huggingface",
    },
    "openvla": {
        "modelFamily": "openvla",
        "kind": "pretrained_checkpoint",
        "sourceType": "public_model_repo",
        "uri": "hf://openvla/openvla-7b",
        "checkpointFormat": "huggingface_transformers",
        "provider": "huggingface",
    },
    "rynnvla": {
        "modelFamily": "rynnvla",
        "kind": "pretrained_checkpoint",
        "sourceType": "public_model_repo",
        "uri": "hf://Alibaba-DAMO-Academy/RynnVLA-001-7B-Base",
        "checkpointFormat": "huggingface_transformers",
        "provider": "huggingface",
    },
    "gr00t": {
        "modelFamily": "gr00t",
        "kind": "pretrained_checkpoint",
        "sourceType": "public_model_repo",
        "uri": "hf://nvidia/GR00T-N1.7-3B",
        "checkpointFormat": "huggingface_transformers",
        "provider": "huggingface",
    },
    "act": {
        "modelFamily": "act",
        "kind": "training_template",
        "checkpointFormat": "lerobot_pretrained_model",
        "provider": "lerobot",
        "warning": "ACT is treated as a training template unless a checkpoint is selected.",
    },
    "diffusion": {
        "modelFamily": "diffusion",
        "kind": "training_template",
        "checkpointFormat": "lerobot_pretrained_model",
        "provider": "lerobot",
        "warning": "Diffusion Policy is treated as a training template unless a checkpoint is selected.",
    },
    "robomimic": {
        "modelFamily": "robomimic",
        "kind": "training_template",
        "checkpointFormat": "robomimic_hdf5",
        "provider": "robomimic",
        "warning": "RoboMimic is treated as a training template unless a checkpoint is selected.",
    },
}

MODEL_ALIASES: dict[str, str] = {
    "pi0_fast": "pi0",
    "pi0.5": "pi05",
    "pi05": "pi05",
    "groot": "gr00t",
    "gr00tn1": "gr00t",
    "gr00t-n1": "gr00t",
    "smol-vla": "smolvla",
    "rynn": "rynnvla",
    "rynnvla-001": "rynnvla",
    "diffusion_policy": "diffusion",
    "diffusion-policy": "diffusion",
}


def canonical_model_family(model_family: str) -> str:
    key = str(model_family or "").strip().lower().replace("_", "-")
    if not key:
        return ""
    return MODEL_ALIASES.get(key, key)


def resolve_builtin_model(model_family: str) -> dict[str, Any]:
    canonical = canonical_model_family(model_family)
    if not canonical:
        return {}
    entry = MODEL_CATALOG.get(canonical)
    if not entry:
        return {"modelFamily": canonical, "kind": "unknown"}
    return dict(entry)
