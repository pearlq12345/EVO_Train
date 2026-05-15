from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import WorkflowPlan, WorkflowStage, build_runner_command, param_int, param_string, quote_args


WORKFLOW_NAME = "rlinf_vla"
GENERIC_WORKFLOW_NAME = "vla_rl_backend"


BUILTIN_TRAINING_PROFILES: dict[str, dict[str, Any]] = {
    "dexbotic_pi0_rlinf": {
        "launchMode": "project_backend",
        "launcherKind": "python_module",
        "repoUrl": "https://github.com/dexmal/dexbotic.git",
        "workdir": "/root/autodl-tmp/dexbotic",
        "launcherModule": "dexbotic.rl.model_rl_libero_pi0",
        "rlinfExtModule": "dexbotic.rl.rlinf_registry",
        "policyFamily": "pi0",
        "successMetric": "success_rate",
        "metricPaths": ["{artifactPath}/metrics.json", "{artifactPath}/eval_info.json"],
    },
    "dexbotic_dm0_rlinf": {
        "launchMode": "project_backend",
        "launcherKind": "python_module",
        "repoUrl": "https://github.com/dexmal/dexbotic.git",
        "workdir": "/root/autodl-tmp/dexbotic",
        "launcherModule": "dexbotic.rl.model_rl_libero_dm0",
        "rlinfExtModule": "dexbotic.rl.rlinf_registry",
        "modelFamily": "dm0",
        "policyFamily": "dm0",
        "trainingMode": "rl_post_train",
        "successMetric": "success_rate",
        "metricPaths": ["{artifactPath}/metrics.json", "{artifactPath}/eval_info.json"],
    },
    "dexbotic_simplevla_rl": {
        "backendKind": "dexbotic",
        "launchMode": "project_backend",
        "launcherKind": "deepspeed_script",
        "repoUrl": "https://github.com/dexmal/dexbotic.git",
        "workdir": "/root/autodl-tmp/dexbotic",
        "scriptPath": "playground/benchmarks/libero/libero_simplevla_rl.py",
        "task": "train",
        "successMetric": "success_rate",
    },
    "roboclaw_rlinf_backend": {
        "launchMode": "project_backend",
        "launcherKind": "python_module",
        "workdir": "/root/autodl-tmp/roboclaw-vla",
        "placementStrategy": "single_node",
        "launcherModule": "roboclaw_vla.rl.launcher",
        "rlinfExtModule": "roboclaw_vla.rl.registry",
        "evalModule": "roboclaw_vla.rl.evaluate",
        "successMetric": "success_rate",
    },
    "roboclaw_grpo_backend": {
        "backendKind": "rlinf",
        "launchMode": "project_backend",
        "launcherKind": "python_module",
        "repoUrl": "https://github.com/pearlq12345/RoboClaw.git",
        "workdir": "/root/autodl-tmp/roboclaw-vla",
        "configName": "libero_10_grpo_roboclaw",
        "configPath": "roboclaw_vla/config/rl/libero_10_grpo_roboclaw.yaml",
        "launcherModule": "roboclaw_vla.rl.launcher",
        "rlinfExtModule": "roboclaw_vla.rl.registry",
        "evalModule": "roboclaw_vla.rl.evaluate",
        "algorithm": "grpo",
        "groupSize": 8,
        "placementStrategy": "single_node",
        "trainingMode": "rl_post_train",
        "successMetric": "success_rate",
        "metricPaths": ["{artifactPath}/metrics.json", "{artifactPath}/eval_info.json"],
    },
}

MODEL_DEFAULT_PROFILES: dict[str, str] = {
    "pi0": "dexbotic_pi0_rlinf",
    "pi0_grpo": "roboclaw_grpo_backend",
    "gr00tn1": "roboclaw_rlinf_backend",
    "gr00t": "roboclaw_rlinf_backend",
    "dm0": "dexbotic_dm0_rlinf",
    "cogact": "roboclaw_rlinf_backend",
    "oft": "roboclaw_rlinf_backend",
    "navila": "roboclaw_rlinf_backend",
    "uni-navid": "roboclaw_rlinf_backend",
}


def _backend_interface(backend_kind: str) -> dict[str, Any]:
    configured = _configured_backend_interface(backend_kind)
    if configured:
        return configured
    if backend_kind == "rlinf":
        return {
            "interfaceVersion": "vla-rl-backend/v1",
            "backendKind": "rlinf",
            "launchModes": ["project_backend", "rlinf_frontend"],
            "launcherKinds": ["python_module", "python_script", "deepspeed_script"],
            "registryInjection": {
                "field": "rlinfExtModule",
                "env": "RLINF_EXT_MODULE",
                "description": "Import this module during preflight and expose it to the launcher for RLinf registry/model/env injection.",
            },
            "preflight": {
                "imports": ["rlinf", "launcherModule", "rlinfExtModule", "envModule", "rewardModule", "preflightModules"],
                "checks": ["launcher import", "registry import", "optional env/reward modules"],
            },
            "envExports": [
                "VLA_RL_BACKEND_KIND",
                "VLA_RL_BACKEND_EXT_MODULE",
                "RLINF_EXT_MODULE",
                "RLINF_ARTIFACT_DIR",
                "RLINF_DATASET_PATH",
                "RLINF_CHECKPOINT_PATH",
                "VLA_RL_CONTRACT_PATH",
                "VLA_RL_MODEL_FAMILY",
                "VLA_RL_TRAINING_MODE",
            ],
            "launcherContract": {
                "python_module": "python -m {launcherModule} --config-name {configName} [--dataset_path ...] [--checkpoint_path ...] [--artifact_path ...]",
                "python_script": "python {scriptPath} --config-name {configName} ...",
                "deepspeed_script": "deepspeed {scriptPath} --task train ...",
            },
            "algorithmToLauncherKind": {
                "ppo": "python_module",
                "sac": "python_module",
                "grpo": "python_module",
            },
            "artifactContract": {
                "contractFile": "run_contract.json",
                "requiredFields": ["backendKind", "modelFamily", "datasetPath", "checkpointPath", "artifactPath", "metricPaths"],
                "successMetricField": "successMetric",
            },
        }
    return {
        "interfaceVersion": "vla-rl-backend/v1",
        "backendKind": backend_kind or "custom",
        "launchModes": ["project_backend"],
        "launcherKinds": ["python_module", "python_script", "deepspeed_script"],
        "registryInjection": {
            "field": "backendExtModule",
            "env": "VLA_RL_BACKEND_EXT_MODULE",
            "description": "Import this module during preflight and expose it to the project launcher.",
        },
        "preflight": {
            "imports": ["launcherModule", "backendExtModule", "envModule", "rewardModule", "preflightModules"],
            "checks": ["launcher import", "optional backend/env/reward modules"],
        },
        "envExports": [
            "VLA_RL_BACKEND_KIND",
            "VLA_RL_BACKEND_EXT_MODULE",
            "RLINF_ARTIFACT_DIR",
            "RLINF_DATASET_PATH",
            "RLINF_CHECKPOINT_PATH",
            "VLA_RL_CONTRACT_PATH",
            "VLA_RL_MODEL_FAMILY",
            "VLA_RL_TRAINING_MODE",
        ],
        "launcherContract": {
            "python_module": "python -m {launcherModule} --config-name {configName} ...",
            "python_script": "python {scriptPath} --config-name {configName} ...",
            "deepspeed_script": "deepspeed {scriptPath} ...",
        },
        "algorithmToLauncherKind": {},
        "artifactContract": {
            "contractFile": "run_contract.json",
            "requiredFields": ["backendKind", "modelFamily", "datasetPath", "checkpointPath", "artifactPath", "metricPaths"],
            "successMetricField": "successMetric",
        },
    }


def _configured_backend_interface(backend_kind: str) -> dict[str, Any]:
    interfaces: dict[str, Any] = {}
    for source in (_load_backend_interfaces_file(), _load_backend_interfaces_env()):
        interfaces.update(source)
    value = interfaces.get(backend_kind)
    return dict(value) if isinstance(value, dict) else {}


def _load_backend_interfaces_env() -> dict[str, Any]:
    raw = os.environ.get("EVO_TRAIN_VLA_BACKEND_INTERFACES_JSON", "").strip()
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("EVO_TRAIN_VLA_BACKEND_INTERFACES_JSON must be a JSON object")
    return parsed


def _load_backend_interfaces_file() -> dict[str, Any]:
    path = os.environ.get("EVO_TRAIN_VLA_BACKEND_INTERFACES_FILE", "").strip()
    if not path:
        return {}
    parsed = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("EVO_TRAIN_VLA_BACKEND_INTERFACES_FILE must contain a JSON object")
    return parsed


def _param_list(params: dict[str, Any], key: str) -> list[str]:
    value = params.get(key)
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _interface_list(backend_interface: dict[str, Any], key: str) -> list[str]:
    value = backend_interface.get(key)
    if isinstance(value, dict):
        value = value.get("imports") or value.get("values")
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _interface_required_params(backend_interface: dict[str, Any]) -> list[str]:
    return _interface_list(backend_interface, "requiredParams")


def _interface_preflight_imports(backend_interface: dict[str, Any]) -> list[str]:
    imports = _interface_list(backend_interface, "preflightImports")
    if imports:
        return imports
    preflight = backend_interface.get("preflight")
    if isinstance(preflight, dict):
        return _interface_list(preflight, "imports")
    return []


def _interface_import_modules(backend_interface: dict[str, Any], values: dict[str, str], preflight_modules: list[str]) -> list[str]:
    modules = []
    for item in _interface_preflight_imports(backend_interface):
        if item == "preflightModules":
            modules.extend(preflight_modules)
        elif item in values:
            modules.append(values[item])
        else:
            modules.append(item)
    return [module for module in modules if module]


def _interface_env_exports(backend_interface: dict[str, Any], values: dict[str, str]) -> dict[str, str]:
    exports: dict[str, str] = {}
    raw = backend_interface.get("envExports")
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        items = ((str(name), str(name)) for name in raw if str(name) in values)
    else:
        items = ()
    for name, source in items:
        export_name = str(name).strip()
        if not export_name:
            continue
        source_text = str(source).strip()
        if source_text in values and values[source_text]:
            exports[export_name] = values[source_text]
        elif source_text.startswith("literal:"):
            exports[export_name] = source_text.removeprefix("literal:")
        elif source_text and source_text not in values:
            exports[export_name] = source_text
    return exports


def _interface_preflight_commands(backend_interface: dict[str, Any], values: dict[str, str]) -> list[str]:
    if not (
        backend_interface.get("usePreflightCommands") is True
        or str(backend_interface.get("preflightCommandMode") or "").strip() == "append"
    ):
        return []
    commands = _interface_list(backend_interface, "preflightCommands")
    return [_format_command_template(command, values) for command in commands if command]


def _interface_launcher_template(backend_interface: dict[str, Any], launcher_kind: str) -> str:
    if not (
        backend_interface.get("useLauncherContract") is True
        or str(backend_interface.get("launcherContractMode") or "").strip() == "template"
    ):
        return ""
    contract = backend_interface.get("launcherContract")
    if not isinstance(contract, dict):
        return ""
    value = contract.get(launcher_kind)
    return str(value).strip() if value not in (None, "") else ""


def _interface_launcher_kind_for_algorithm(backend_interface: dict[str, Any], algorithm: str) -> str:
    mapping = backend_interface.get("algorithmToLauncherKind")
    if not isinstance(mapping, dict) or not algorithm:
        return ""
    value = mapping.get(algorithm)
    return str(value).strip() if value not in (None, "") else ""


def _format_command_template(template: str, values: dict[str, str]) -> str:
    quoted_values = {key: quote_args([str(value)]) for key, value in values.items()}

    class _Missing(dict[str, str]):
        def __missing__(self, key: str) -> str:
            return ""

    return template.format_map(_Missing(quoted_values)).strip()


def _append_override(overrides: list[str], key: str, value: Any) -> None:
    if value is None or value == "":
        return
    text = str(value).strip()
    if text:
        overrides.append(f"{key}={text}")


def _append_flag(parts: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    text = str(value).strip()
    if text:
        parts.append(flag)
        parts.append(text)


def _import_check_command(modules: list[str], label: str) -> str:
    imports = "; ".join(f"importlib.import_module({module!r})" for module in modules if module)
    if not imports:
        imports = "pass"
    code = f"import importlib; {imports}; print({label!r})"
    return f"python -c {quote_args([code])}"


def _config_file_check_command(config_path: str) -> str:
    if not config_path:
        return ""
    return f"test -f {quote_args([config_path])}"


def _write_json_command(path: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    code = (
        "import pathlib; "
        f"path=pathlib.Path({path!r}); "
        "path.parent.mkdir(parents=True, exist_ok=True); "
        f"path.write_text({encoded!r}, encoding='utf-8'); "
        "print('contract written', path)"
    )
    return f"python -c {quote_args([code])}"


def _format_profile_value(value: Any, values: dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**values)
        except KeyError:
            return value
    if isinstance(value, list):
        return [_format_profile_value(item, values) for item in value]
    return value


def _apply_builtin_profile(params: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(params)
    profile_id = param_string(enriched, "builtinTrainingProfile")
    explicit_profile = bool(profile_id)
    if not profile_id:
        if any(enriched.get(key) for key in ("repoUrl", "launcherModule", "scriptPath", "entrypoint")):
            return enriched
        algorithm = param_string(enriched, "algorithm")
        model_family = param_string(enriched, "modelFamily")
        if algorithm == "grpo" and model_family == "pi0":
            model_family = "pi0_grpo"
        profile_id = MODEL_DEFAULT_PROFILES.get(model_family, "")
    profile = BUILTIN_TRAINING_PROFILES.get(profile_id)
    if not profile:
        return enriched
    values = dict(enriched)
    artifact_path = param_string(enriched, "artifactPath", f"{param_string(enriched, 'workdir', '/root/autodl-tmp/RLinf')}/outputs")
    values.setdefault("artifactPath", artifact_path)
    for key, value in profile.items():
        enriched.setdefault(key, _format_profile_value(value, values))
    enriched["builtinTrainingProfile"] = profile_id
    return enriched


def build_plan(request: dict[str, Any], params: dict[str, Any]) -> WorkflowPlan:
    params = _apply_builtin_profile(params)
    requested_workflow = str(request.get("workflow") or params.get("workflow") or WORKFLOW_NAME).strip()
    workflow_name = GENERIC_WORKFLOW_NAME if requested_workflow == GENERIC_WORKFLOW_NAME else WORKFLOW_NAME
    backend_kind = param_string(params, "backendKind", param_string(params, "backend", "rlinf"))
    configured_backend_interface = params.get("backendInterface")
    backend_interface = (
        dict(configured_backend_interface)
        if isinstance(configured_backend_interface, dict)
        else _backend_interface(backend_kind)
    )
    backend_interface.setdefault("backendKind", backend_kind)
    explicit_launch_mode = param_string(params, "launchMode")
    launch_mode = explicit_launch_mode or (
        "project_backend" if params.get("launcherModule") or params.get("scriptPath") else "rlinf_frontend"
    )
    interface_launcher_kinds = _interface_list(backend_interface, "launcherKinds")
    algorithm = param_string(params, "algorithm")
    launcher_kind = param_string(
        params,
        "launcherKind",
        _interface_launcher_kind_for_algorithm(backend_interface, algorithm)
        or (interface_launcher_kinds[0] if interface_launcher_kinds else "python_module"),
    )
    repo_url = param_string(params, "repoUrl")
    branch = param_string(params, "branch", "main")
    workdir = param_string(params, "workdir", "/root/autodl-tmp/RLinf")
    config_name = param_string(params, "configName")
    config_path = param_string(params, "configPath")
    entrypoint = param_string(params, "entrypoint", "examples/embodiment/train_embodied_agent.py")
    launcher_module = param_string(params, "launcherModule")
    script_path = param_string(params, "scriptPath")
    backend_ext_module = param_string(params, "backendExtModule", param_string(params, "rlinfExtModule"))
    rlinf_ext_module = backend_ext_module if backend_kind == "rlinf" else ""
    preflight_modules = _param_list(params, "preflightModules")
    model_registry_name = param_string(params, "modelRegistryName", param_string(params, "modelFamily"))
    policy_family = param_string(params, "policyFamily", param_string(params, "modelFamily"))
    model_family = param_string(params, "modelFamily", policy_family)
    builtin_training_profile = param_string(params, "builtinTrainingProfile")
    training_mode = param_string(params, "trainingMode", "rl_post_train")
    co_training_targets = _param_list(params, "coTrainingTargets")
    robot_adapter = param_string(params, "robotAdapter")
    image_profile = param_string(params, "imageProfile")
    env_module = param_string(params, "envModule")
    reward_module = param_string(params, "rewardModule")
    suite = param_string(params, "suite")
    sft_model_path = param_string(params, "sftModelPath")
    dataset_name = param_string(params, "datasetName")
    task = param_string(params, "task")
    eval_module = param_string(params, "evalModule")
    eval_script_path = param_string(params, "evalScriptPath")
    eval_kind = param_string(params, "evalKind", "python_module" if eval_module else "python_script")
    setup_command = param_string(params, "setupCommand", "python -m pip install -e .")
    artifact_path = param_string(params, "artifactPath", f"{workdir}/outputs")
    contract_path = param_string(params, "contractPath", f"{artifact_path}/run_contract.json")
    dataset_path = param_string(params, "datasetPath")
    checkpoint_path = param_string(params, "checkpointPath", artifact_path)
    metric_paths = _param_list(params, "metricPaths") or [f"{artifact_path}/metrics.json"]
    result_files = _param_list(params, "resultFiles")
    success_metric = param_string(params, "successMetric", "success_rate")
    placement_strategy = param_string(params, "placementStrategy", "single_node")
    group_size = param_int(params, "groupSize", 0)
    robot_embodiment = param_string(params, "robotEmbodiment")
    observation_schema = param_string(params, "observationSchema")
    action_schema = param_string(params, "actionSchema")
    provider = str(request.get("provider") or params.get("provider") or "autodl")
    gpu_spec = param_string(params, "gpuSpec", str(request.get("gpuSpec") or "default"))
    hourly_price_cents = int(request.get("hourlyPriceCents") or params.get("hourlyPriceCents") or 1000)
    estimated_hours = param_int(params, "estimatedHours", 1)

    missing_fields = []
    effective_values = {
        "repoUrl": repo_url,
        "workdir": workdir,
        "configName": config_name,
        "configPath": config_path,
        "entrypoint": entrypoint,
        "launcherModule": launcher_module,
        "scriptPath": script_path,
        "backendExtModule": backend_ext_module,
        "rlinfExtModule": rlinf_ext_module,
        "envModule": env_module,
        "rewardModule": reward_module,
        "datasetPath": dataset_path,
        "checkpointPath": checkpoint_path,
        "artifactPath": artifact_path,
        "modelFamily": model_family,
        "policyFamily": policy_family,
    }
    for field in _interface_required_params(backend_interface):
        if not effective_values.get(field) and params.get(field) in (None, ""):
            missing_fields.append(field)
    if launcher_kind != "deepspeed_script" and not config_name:
        missing_fields.append("configName")
    warnings: list[str] = []
    if launch_mode not in {"project_backend", "rlinf_frontend"}:
        missing_fields.append("launchMode")
        warnings.append("launchMode must be project_backend or rlinf_frontend")
    if launcher_kind not in {"python_module", "python_script", "deepspeed_script"}:
        missing_fields.append("launcherKind")
        warnings.append("launcherKind must be python_module, python_script, or deepspeed_script")
    if interface_launcher_kinds and launcher_kind not in interface_launcher_kinds:
        missing_fields.append("launcherKind")
        warnings.append(f"launcherKind must be one of: {', '.join(interface_launcher_kinds)}")
    if launch_mode == "project_backend" and launcher_kind == "python_module" and not launcher_module:
        missing_fields.append("launcherModule")
    if launch_mode == "project_backend" and launcher_kind in {"python_script", "deepspeed_script"} and not script_path:
        missing_fields.append("scriptPath")
    if not repo_url:
        warnings.append("repoUrl is empty; the selected image must already contain the RLinf code at workdir")
    if gpu_spec == "default":
        warnings.append("gpuSpec is default; RoboClaw should confirm the target GPU tier")
    if not dataset_path:
        warnings.append("datasetPath is empty; confirm the RLinf config does not require an external dataset path")
    if launch_mode == "project_backend" and not backend_ext_module:
        warnings.append("backendExtModule is empty; custom model registration must happen inside the launcher module")

    env_exports = {
        "MUJOCO_GL": param_string(params, "mujocoGl", "egl"),
        "PYTHONUNBUFFERED": "1",
        "RLINF_ARTIFACT_DIR": artifact_path,
        "VLA_RL_BACKEND_KIND": backend_kind,
    }
    if backend_ext_module:
        env_exports["VLA_RL_BACKEND_EXT_MODULE"] = backend_ext_module
    if rlinf_ext_module:
        env_exports["RLINF_EXT_MODULE"] = rlinf_ext_module
    if dataset_path:
        env_exports["RLINF_DATASET_PATH"] = dataset_path
    if checkpoint_path:
        env_exports["RLINF_CHECKPOINT_PATH"] = checkpoint_path
    if contract_path:
        env_exports["VLA_RL_CONTRACT_PATH"] = contract_path
    if env_module:
        env_exports["VLA_RL_ENV_MODULE"] = env_module
    if reward_module:
        env_exports["VLA_RL_REWARD_MODULE"] = reward_module
    if model_registry_name:
        env_exports["VLA_RL_MODEL_REGISTRY_NAME"] = model_registry_name
    if model_family:
        env_exports["VLA_RL_MODEL_FAMILY"] = model_family
    if training_mode:
        env_exports["VLA_RL_TRAINING_MODE"] = training_mode
    if robot_adapter:
        env_exports["VLA_RL_ROBOT_ADAPTER"] = robot_adapter
    env_exports.update(_interface_env_exports(backend_interface, effective_values))

    interface_import_modules = _interface_import_modules(backend_interface, effective_values, preflight_modules)
    if interface_import_modules:
        import_modules = interface_import_modules
    else:
        import_modules = [
            module for module in [launcher_module, backend_ext_module, env_module, reward_module, *preflight_modules] if module
        ]
    overrides = _param_list(params, "overrides")
    launcher_args = _param_list(params, "launcherArgs")
    _append_override(overrides, "runner.max_steps", params.get("maxSteps"))
    _append_override(overrides, "runner.eval_episodes", params.get("evalEpisodes"))
    _append_override(overrides, "env.name", params.get("benchmark") or params.get("envType"))
    _append_override(overrides, "actor.model.model_type", params.get("modelFamily"))
    _append_override(overrides, "algorithm.name", algorithm)
    if group_size > 0:
        _append_override(overrides, "algorithm.group_size", group_size)

    export_command = " ".join(
        f"export {name}={quote_args([value])};" for name, value in env_exports.items() if value
    )
    interface_preflight_commands = _interface_preflight_commands(backend_interface, effective_values)
    config_check = _config_file_check_command(config_path)
    launcher_template = _interface_launcher_template(backend_interface, launcher_kind)
    rendered_launcher_command = ""
    if launch_mode == "project_backend":
        backend_imports = [] if interface_import_modules else (["rlinf"] if backend_kind == "rlinf" else [])
        import_check = _import_check_command(
            [*import_modules, *backend_imports],
            f"launcher and {backend_kind} backend ok",
        )
        if launcher_kind == "python_module":
            train_parts = ["python", "-m", launcher_module, "--config-name", config_name]
        elif launcher_kind == "deepspeed_script":
            train_parts = ["deepspeed", script_path]
            _append_flag(train_parts, "--task", task or "train")
            _append_flag(train_parts, "--sft_model_path", sft_model_path)
            _append_flag(train_parts, "--dataset_name", dataset_name)
        else:
            train_parts = ["python", script_path, "--config-name", config_name]
        preflight_checks = [item for item in [config_check, import_check] if item]
        if launcher_kind in {"python_script", "deepspeed_script"}:
            preflight_checks.insert(0, f"test -f {quote_args([script_path])}")
        preflight_checks.extend(interface_preflight_commands)
        preflight_command = f"cd {quote_args([workdir])} && {' && '.join(preflight_checks)}"
        if suite:
            train_parts.append(f"--suite={suite}")
        _append_flag(train_parts, "--dataset_path", dataset_path)
        _append_flag(train_parts, "--checkpoint_path", checkpoint_path)
        _append_flag(train_parts, "--artifact_path", artifact_path)
        train_parts.extend(launcher_args)
        train_parts.extend(overrides)
        if launcher_template:
            rendered_launcher_command = _format_command_template(launcher_template, effective_values)
            if launcher_args or overrides:
                rendered_launcher_command = f"{rendered_launcher_command} {quote_args([*launcher_args, *overrides])}".strip()
    else:
        train_parts = ["python", entrypoint, "--config-name", config_name, *overrides]
        backend_imports = [] if interface_import_modules else (["rlinf"] if backend_kind == "rlinf" else [])
        import_check = _import_check_command([*import_modules, *backend_imports], "rlinf frontend ok")
        preflight_checks = [item for item in [f"test -f {quote_args([entrypoint])}", config_check, import_check] if item]
        preflight_command = f"cd {quote_args([workdir])} && {' && '.join(preflight_checks)}"
    train_body = rendered_launcher_command or quote_args(train_parts)
    train_command = f"cd {quote_args([workdir])} && {export_command} {train_body}"
    run_contract = {
        "workflow": workflow_name,
        "backendKind": backend_kind,
        "launchMode": launch_mode,
        "launcherKind": launcher_kind,
        "launcherModule": launcher_module,
        "scriptPath": script_path,
        "backendExtModule": backend_ext_module,
        "rlinfExtModule": rlinf_ext_module,
        "backendInterface": backend_interface,
        "builtinTrainingProfile": builtin_training_profile,
        "configPath": config_path,
        "modelRegistryName": model_registry_name,
        "modelFamily": model_family,
        "policyFamily": policy_family,
        "algorithm": algorithm,
        "trainingMode": training_mode,
        "coTrainingTargets": co_training_targets,
        "robotAdapter": robot_adapter,
        "imageProfile": image_profile,
        "envModule": env_module,
        "rewardModule": reward_module,
        "suite": suite,
        "datasetPath": dataset_path,
        "checkpointPath": checkpoint_path,
        "artifactPath": artifact_path,
        "metricPaths": metric_paths,
        "resultFiles": result_files,
        "successMetric": success_metric,
        "placementStrategy": placement_strategy,
        "groupSize": group_size,
        "robotEmbodiment": robot_embodiment,
        "observationSchema": observation_schema,
        "actionSchema": action_schema,
        "overrides": overrides,
    }
    write_contract_command = f"cd {quote_args([workdir])} && {_write_json_command(contract_path, run_contract)}"
    eval_command = ""
    if eval_module:
        eval_parts = ["python", "-m", eval_module]
        _append_flag(eval_parts, "--checkpoint_path", checkpoint_path)
        _append_flag(eval_parts, "--artifact_path", artifact_path)
        if suite:
            eval_parts.append(f"--suite={suite}")
        eval_command = f"cd {quote_args([workdir])} && {export_command} {quote_args(eval_parts)}"
    elif eval_script_path:
        eval_parts = ["deepspeed" if eval_kind == "deepspeed_script" else "python", eval_script_path]
        _append_flag(eval_parts, "--checkpoint_path", checkpoint_path)
        _append_flag(eval_parts, "--artifact_path", artifact_path)
        if suite:
            eval_parts.append(f"--suite={suite}")
        eval_command = f"cd {quote_args([workdir])} && {export_command} {quote_args(eval_parts)}"
    collect_command = f"test -e {quote_args([artifact_path])}"

    stages: list[WorkflowStage] = []
    if repo_url:
        clone_command = quote_args(["git", "clone", "--branch", branch, "--depth", "1", repo_url, workdir])
        stages.append(WorkflowStage("prepare_code", f"rm -rf {quote_args([workdir])} && {clone_command}"))
    stages.extend(
        [
            WorkflowStage("setup_env", f"cd {quote_args([workdir])} && {setup_command}"),
            WorkflowStage("preflight", preflight_command),
            WorkflowStage("write_contract", write_contract_command),
            WorkflowStage("train_rlinf_vla" if workflow_name == WORKFLOW_NAME else "train_vla_rl_backend", train_command),
        ]
    )
    if eval_command:
        stages.append(WorkflowStage("evaluate", eval_command, required=False))
    stages.append(WorkflowStage("collect_artifacts", collect_command, required=False))

    return WorkflowPlan(
        workflow=workflow_name,
        provider=provider,
        params={
            "backendKind": backend_kind,
            "launchMode": launch_mode,
            "launcherKind": launcher_kind,
            "repoUrl": repo_url,
            "branch": branch,
            "workdir": workdir,
            "configName": config_name,
            "configPath": config_path,
            "entrypoint": entrypoint,
            "launcherModule": launcher_module,
            "scriptPath": script_path,
            "backendExtModule": backend_ext_module,
            "rlinfExtModule": rlinf_ext_module,
            "backendInterface": backend_interface,
            "builtinTrainingProfile": builtin_training_profile,
            "preflightModules": preflight_modules,
            "modelRegistryName": model_registry_name,
            "modelFamily": model_family,
            "policyFamily": policy_family,
            "algorithm": algorithm,
            "trainingMode": training_mode,
            "coTrainingTargets": co_training_targets,
            "robotAdapter": robot_adapter,
            "imageProfile": image_profile,
            "envModule": env_module,
            "rewardModule": reward_module,
            "suite": suite,
            "sftModelPath": sft_model_path,
            "datasetName": dataset_name,
            "task": task,
            "launcherArgs": launcher_args,
            "evalModule": eval_module,
            "evalScriptPath": eval_script_path,
            "evalKind": eval_kind,
            "datasetPath": dataset_path,
            "artifactPath": artifact_path,
            "contractPath": contract_path,
            "checkpointPath": checkpoint_path,
            "metricPaths": metric_paths,
            "resultFiles": result_files,
            "successMetric": success_metric,
            "placementStrategy": placement_strategy,
            "groupSize": group_size,
            "robotEmbodiment": robot_embodiment,
            "observationSchema": observation_schema,
            "actionSchema": action_schema,
            "overrides": overrides,
        },
        command=build_runner_command(stages),
        stages=stages,
        workdir=workdir,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        gpu_spec=gpu_spec,
        hourly_price_cents=hourly_price_cents,
        missing_fields=missing_fields,
        warnings=warnings,
        estimated_hours=max(1, estimated_hours),
        summary=f"{backend_kind} VLA/RL backend training with config {config_name or '<missing configName>'}.",
    )
