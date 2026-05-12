from __future__ import annotations

from typing import Any

from .base import (
    WorkflowPlan,
    estimate_hours,
    param_bool,
    param_float,
    param_int,
    param_string,
    quote_args,
    training_warnings,
)


WORKFLOW_NAME = "evf_metaworld"


def build_plan(request: dict[str, Any], params: dict[str, Any]) -> WorkflowPlan:
    env_name = param_string(params, "envName", param_string(params, "task", "pick-place-v2"))
    epochs = param_int(params, "epochs", 20)
    batch_size = param_int(params, "batchSize", 32)
    learning_rate = param_float(params, "learningRate", param_float(params, "lr", 1e-4))
    seed = param_int(params, "seed", 42)
    eval_episodes = param_int(params, "evalEpisodes", 10)
    save_video = param_bool(params, "saveVideo", True)
    workdir = param_string(params, "workdir", "/root/autodl-tmp/evf")
    dataset_path = param_string(params, "datasetPath", f"/root/autodl-tmp/datasets/metaworld/{env_name}")
    checkpoint_path = param_string(params, "checkpointPath", f"/root/autodl-tmp/evo_train/runs/{env_name}")
    provider = str(request.get("provider") or params.get("provider") or "autodl")
    gpu_spec = param_string(params, "gpuSpec", str(request.get("gpuSpec") or "default"))
    hourly_price_cents = int(request.get("hourlyPriceCents") or params.get("hourlyPriceCents") or 1000)
    missing_fields = [field for field in ("envName",) if not params.get(field)]
    warnings = training_warnings(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        eval_episodes=eval_episodes,
        gpu_spec=gpu_spec,
    )

    command_parts = [
        "python",
        "train.py",
        "--benchmark",
        "metaworld",
        "--env-name",
        env_name,
        "--dataset-path",
        dataset_path,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(learning_rate),
        "--seed",
        str(seed),
        "--checkpoint-path",
        checkpoint_path,
        "--eval-episodes",
        str(eval_episodes),
    ]
    if save_video:
        command_parts.append("--save-video")

    return WorkflowPlan(
        workflow=WORKFLOW_NAME,
        provider=provider,
        params={
            "envName": env_name,
            "epochs": epochs,
            "batchSize": batch_size,
            "learningRate": learning_rate,
            "seed": seed,
            "evalEpisodes": eval_episodes,
            "saveVideo": save_video,
        },
        command=quote_args(command_parts),
        workdir=workdir,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        gpu_spec=gpu_spec,
        hourly_price_cents=hourly_price_cents,
        missing_fields=missing_fields,
        warnings=warnings,
        estimated_hours=estimate_hours(epochs, eval_episodes),
        summary=f"MetaWorld {env_name}: train {epochs} epochs, then evaluate {eval_episodes} episodes.",
    )
