from __future__ import annotations

from typing import Any

from .base import WorkflowPlan, WorkflowStage, build_runner_command, param_string, quote_args


WORKFLOW_NAME = "custom_project"


def build_plan(request: dict[str, Any], params: dict[str, Any]) -> WorkflowPlan:
    repo_url = param_string(params, "repoUrl")
    branch = param_string(params, "branch", "main")
    workdir = param_string(params, "workdir", "/root/autodl-tmp/custom_project")
    setup_command = param_string(params, "setupCommand", "python -m pip install -r requirements.txt")
    data_command = param_string(params, "dataCommand", "true")
    train_command = param_string(params, "trainCommand")
    eval_command = param_string(params, "evalCommand", "true")
    artifact_path = param_string(params, "artifactPath", f"{workdir}/outputs")
    provider = str(request.get("provider") or params.get("provider") or "autodl")
    gpu_spec = param_string(params, "gpuSpec", str(request.get("gpuSpec") or "default"))
    hourly_price_cents = int(request.get("hourlyPriceCents") or params.get("hourlyPriceCents") or 1000)

    missing_fields = [field for field in ("repoUrl", "trainCommand") if not params.get(field)]
    warnings: list[str] = []
    if setup_command == "python -m pip install -r requirements.txt":
        warnings.append("setupCommand uses default requirements.txt; confirm the repo has this file")
    if eval_command == "true":
        warnings.append("evalCommand is empty; this run will not produce an evaluation metric unless trainCommand does it")
    if data_command == "true":
        warnings.append("dataCommand is empty; confirm data is already available in the instance")
    if gpu_spec == "default":
        warnings.append("gpuSpec is default; RoboClaw should confirm the target GPU tier")

    clone_command = quote_args(["git", "clone", "--branch", branch, "--depth", "1", repo_url, workdir])
    stages = [
        WorkflowStage("prepare_code", f"rm -rf {quote_args([workdir])} && {clone_command}"),
        WorkflowStage("setup_env", f"cd {quote_args([workdir])} && {setup_command}"),
        WorkflowStage("prepare_data", f"cd {quote_args([workdir])} && {data_command}", required=data_command != "true"),
        WorkflowStage("train", f"cd {quote_args([workdir])} && {train_command}"),
        WorkflowStage("evaluate", f"cd {quote_args([workdir])} && {eval_command}", required=eval_command != "true"),
        WorkflowStage("collect_artifacts", f"test -e {quote_args([artifact_path])}", required=False),
    ]

    return WorkflowPlan(
        workflow=WORKFLOW_NAME,
        provider=provider,
        params={
            "repoUrl": repo_url,
            "branch": branch,
            "setupCommand": setup_command,
            "dataCommand": data_command,
            "trainCommand": train_command,
            "evalCommand": eval_command,
            "artifactPath": artifact_path,
        },
        command=build_runner_command(stages),
        stages=stages,
        workdir=workdir,
        checkpoint_path=artifact_path,
        dataset_path="",
        gpu_spec=gpu_spec,
        hourly_price_cents=hourly_price_cents,
        missing_fields=missing_fields,
        warnings=warnings,
        estimated_hours=1,
        summary=f"Custom project from {repo_url or '<missing repo>'}: run setup, train, optional eval, and collect artifacts.",
    )
