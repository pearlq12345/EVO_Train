# RLinf VLA Workflow

`rlinf_vla` lets RoboClaw submit a VLA+RL post-training plan to EVO_Train. EVO_Train still owns billing, AutoDL lifecycle, status sync, logs, and artifact download.

## Required Fields

- `workflow`: `rlinf_vla`
- `provider`: usually `autodl`
- `params.configName`: RLinf Hydra config name

The selected AutoDL image must either contain RLinf at `params.workdir`, or the request must provide `params.repoUrl` so the workflow can clone it.

## Recommended Fields

Use `project_backend` for production-style VLA+RL. This mirrors Dexbotic's integration: the project owns the model adapter and user entrypoint, while RLinf remains the distributed RL backend.

```json
{
  "launchMode": "project_backend",
  "repoUrl": "https://github.com/dexmal/dexbotic.git",
  "branch": "main",
  "workdir": "/root/autodl-tmp/dexbotic",
  "configName": "libero_goal_ppo_dexbotic_pi0",
  "launcherModule": "dexbotic.rl.model_rl_libero_pi0",
  "rlinfExtModule": "dexbotic.rl.rlinf_registry",
  "suite": "libero_goal",
  "datasetPath": "/root/autodl-tmp/datasets/libero",
  "artifactPath": "/root/autodl-tmp/evo_train/jobs/rlinf-vla-smoke/artifacts",
  "algorithm": "ppo",
  "modelFamily": "pi0",
  "benchmark": "libero",
  "maxSteps": 1000,
  "evalEpisodes": 2,
  "overrides": [
    "cluster.num_nodes=1"
  ]
}
```

## Launch Modes

### `rlinf_frontend`

Use this only for low-level debugging when the selected image or cloned repo is RLinf itself.

```json
{
  "launchMode": "rlinf_frontend",
  "repoUrl": "https://github.com/RLinf/RLinf.git",
  "workdir": "/root/autodl-tmp/RLinf",
  "configName": "libero_pi0_grpo_smoke",
  "entrypoint": "examples/embodiment/train_embodied_agent.py"
}
```

This generates:

```bash
python examples/embodiment/train_embodied_agent.py --config-name libero_pi0_grpo_smoke
```

### `project_backend`

Use this by default for Dexbotic-style and RoboClaw-style integrations where the project owns the model, config, adapter, and entrypoint while RLinf is imported as the backend.

```json
{
  "launchMode": "project_backend",
  "repoUrl": "https://github.com/dexmal/dexbotic.git",
  "workdir": "/root/autodl-tmp/dexbotic",
  "configName": "libero_goal_ppo_dexbotic_pi0",
  "launcherModule": "dexbotic.rl.model_rl_libero_pi0",
  "rlinfExtModule": "dexbotic.rl.rlinf_registry",
  "suite": "libero_goal"
}
```

This generates:

```bash
export RLINF_EXT_MODULE=dexbotic.rl.rlinf_registry
python -m dexbotic.rl.model_rl_libero_pi0 --config-name libero_goal_ppo_dexbotic_pi0 --suite=libero_goal
```

`RLINF_EXT_MODULE` is the important backend bridge: RLinf worker processes import it so custom model builders are registered on every worker.

`preflightModules` can list extra Python modules that must import before paid runtime reaches the training stage. Use this for adapter packages, reward code, environment wrappers, or project registries.

Preflight is executable, not just descriptive: EVO_Train renders Python import checks for the selected launcher, registry injection module, and optional env/reward modules. Profiles that provide `configPath` also add a `test -f` check so a misspelled Hydra config fails before the training stage.

## Launcher Kinds

`project_backend` supports multiple structured launcher kinds. This keeps EVO_Train from accepting arbitrary shell while still covering the common VLA-RL patterns.

### `python_module`

Use this for Dexbotic/RoboClaw-style RLinf backend integration:

```json
{
  "launcherKind": "python_module",
  "launcherModule": "dexbotic.rl.model_rl_libero_pi0",
  "rlinfExtModule": "dexbotic.rl.rlinf_registry",
  "configName": "libero_goal_ppo_dexbotic_pi0",
  "suite": "libero_goal"
}
```

### `deepspeed_script`

Use this for SimpleVLA-RL-style project scripts:

```json
{
  "launcherKind": "deepspeed_script",
  "scriptPath": "playground/benchmarks/libero/libero_simplevla_rl.py",
  "task": "train",
  "sftModelPath": "/root/autodl-tmp/checkpoints/pi0-sft",
  "datasetName": "libero_10"
}
```

This generates:

```bash
deepspeed playground/benchmarks/libero/libero_simplevla_rl.py \
  --task train \
  --sft_model_path /root/autodl-tmp/checkpoints/pi0-sft \
  --dataset_name libero_10
```

### `python_script`

Use this for project-owned Python scripts that still accept Hydra `--config-name`.

## Structured Arguments

The workflow accepts structured arguments instead of raw shell:

- `launcherArgs`: extra CLI tokens appended to the launcher, such as `["--profile", "smoke"]`.
- `overrides`: Hydra-style overrides such as `["cluster.num_nodes=1"]`.
- `datasetPath`, `checkpointPath`, and `artifactPath`: exported as env vars and also passed to project launchers when supported.
- `evalModule`, `evalScriptPath`, and `evalKind`: optional evaluation entrypoint after training.
- `modelRegistryName`, `policyFamily`, `envModule`, `rewardModule`: project-level registry and adapter declarations.
- `metricPaths`, `resultFiles`, `successMetric`: expected result contract for RoboClaw artifact review.
- `robotEmbodiment`, `observationSchema`, `actionSchema`: deployment-facing compatibility hints.

## Run Contract

Before training starts, EVO_Train writes a `run_contract.json` file under `artifactPath` by default. This is the stable handoff between training, diagnosis, artifact review, and RoboClaw policy deployment.

Example fields:

```json
{
  "workflow": "rlinf_vla",
  "launchMode": "project_backend",
  "launcherKind": "python_module",
  "launcherModule": "roboclaw_vla.rl.launcher",
  "rlinfExtModule": "roboclaw_vla.rl.registry",
  "modelRegistryName": "roboclaw_pi0",
  "policyFamily": "pi0",
  "envModule": "roboclaw_vla.envs.libero",
  "rewardModule": "roboclaw_vla.rewards.success",
  "datasetPath": "/root/autodl-tmp/datasets/libero",
  "checkpointPath": "/root/autodl-tmp/evo_train/jobs/vla/checkpoints",
  "artifactPath": "/root/autodl-tmp/evo_train/jobs/vla/artifacts",
  "metricPaths": ["/root/autodl-tmp/evo_train/jobs/vla/artifacts/metrics.json"],
  "successMetric": "success_rate"
}
```

## Capability Normalization

`AI配置训练` normalizes common VLA release capabilities into contract fields:

| User language | Contract field |
| --- | --- |
| Uni-NaVid / UniNaVid | `modelFamily: "uni-navid"` |
| GR00TN1 | `modelFamily: "gr00tn1"` |
| GR00T | `modelFamily: "gr00t"` |
| Pi0.5 / Pi05 | `modelFamily: "pi0.5"` |
| DM0 | `modelFamily: "dm0"` |
| CogACT | `modelFamily: "cogact"` |
| OFT | `modelFamily: "oft"` |
| NaVILA | `modelFamily: "navila"` |
| GRPO / PPO | `algorithm` and Hydra override |
| co-training / joint optimization | `trainingMode: "co_training"` |
| action expert + LLM | `coTrainingTargets: ["action_expert", "llm"]` |
| SO-101 | `robotAdapter: "so-101"` |
| XLeRobot | `robotAdapter: "xlerobot"` |
| Blackwell image | `imageProfile: "blackwell"` |

## Built-In Training Profiles

The workflow includes built-in profile defaults so RoboClaw can request model training without asking users for low-level launcher modules.

| Profile | Purpose |
| --- | --- |
| `dexbotic_pi0_rlinf` | Dexbotic-style Pi0 RLinf backend launcher |
| `dexbotic_dm0_rlinf` | Dexbotic-style DM0 RLinf backend launcher |
| `dexbotic_simplevla_rl` | SimpleVLA-RL deepspeed script profile |
| `roboclaw_rlinf_backend` | RoboClaw-owned future VLA-RL launcher contract |
| `roboclaw_grpo_backend` | RoboClaw GRPO template with `groupSize=8`, `placementStrategy=single_node`, and a Hydra config path |

When a request declares only `modelFamily`, EVO_Train can infer a profile unless the request already provides an explicit repo, launcher, script, or entrypoint. Explicit launch fields always win.

`roboclaw_grpo_backend` is intentionally a template contract. The remote repository still needs a real RLinf launcher that constructs actor, rollout, and env worker groups before this can run full training.

## Generated Stages

1. `prepare_code`: clone the project or RLinf repo when `repoUrl` is provided.
2. `setup_env`: run `python -m pip install -e .` by default.
3. `preflight`: verify launcher/script imports, RLinf registry modules, and core dependencies.
4. `write_contract`: persist `run_contract.json` for review and deployment.
5. `train_rlinf_vla`: run the selected structured launcher.
6. `evaluate`: optional, runs only when an eval module or script is provided.
7. `collect_artifacts`: check `artifactPath` exists without failing the whole task.

## TCP Example

```bash
printf '%s\n' '{
  "username": "pearl",
  "taskName": "rlinf-vla-smoke",
  "action": "开始训练",
  "provider": "autodl",
  "workflow": "rlinf_vla",
  "skuId": "autodl-4090d",
  "imageId": "rlinf-vla-cu121",
  "params": {
    "launchMode": "project_backend",
    "repoUrl": "https://github.com/dexmal/dexbotic.git",
    "workdir": "/root/autodl-tmp/dexbotic",
    "configName": "libero_goal_ppo_dexbotic_pi0",
    "launcherModule": "dexbotic.rl.model_rl_libero_pi0",
    "rlinfExtModule": "dexbotic.rl.rlinf_registry",
    "suite": "libero_goal",
    "datasetPath": "/root/autodl-tmp/datasets/libero",
    "artifactPath": "/root/autodl-tmp/evo_train/jobs/rlinf-vla-smoke/artifacts",
    "maxSteps": 1000,
    "evalEpisodes": 2
  }
}' | nc 127.0.0.1 9000
```

## Real-Run Checklist

- AutoDL token is configured on the EVO_Train server.
- `AUTODL_GPU_SKUS_JSON` maps `skuId` to a real AutoDL GPU spec UUID and sale price.
- `AUTODL_IMAGES_JSON` maps `imageId` to a real AutoDL image UUID and CUDA version.
- The RLinf `configName` exists in the repo or image.
- Dataset paths referenced by the RLinf config exist on the remote instance.
