# RoboClaw AI Skills Architecture

## Purpose

RoboClaw should use AI skills as small, composable training workflows rather than asking users to write raw commands or provider-specific JSON. Each skill guides the user, validates inputs, estimates cost, and materializes a backend request that the TCP training service can execute.

The goal is:

- users describe intent in natural language;
- AI turns intent into a structured training plan;
- RoboClaw asks only the missing questions;
- backend providers execute a validated request;
- billing and result collection stay controlled by the platform.

## Domain Language

Use consistent terms across AI prompts, UI, backend actions, and logs.

| Term | Meaning |
| --- | --- |
| Skill | A reusable AI-guided training capability, such as custom project training, MetaWorld, LIBERO, or failure diagnosis. |
| Training Plan | The AI-generated plan shown to the user before starting a task. |
| Workflow | A backend recipe that materializes a plan into staged shell commands. |
| Provider | The execution backend, such as AutoDL or Aliyun DLC. |
| GPU SKU | A platform product for compute, with display name, provider GPU spec, cost, and sale price. |
| Image | A reusable training environment, mapped to provider image identifiers and CUDA requirements. |
| Artifact | Result files produced by training, including checkpoints, videos, metrics, and logs. |
| Smoke Test | A short run used to validate environment, data, and command wiring before burning full GPU budget. |
| Diagnosis | The process of reading logs, loss, metrics, and task metadata to classify and fix failures. |

Avoid exposing provider-only terms such as `gpu_spec_uuid`, `image_uuid`, or SSH details to normal users. Those belong to admin configuration and backend mappings.

## Skill Contract

Each RoboClaw skill should be described with this shape:

```yaml
id: evf-metaworld
name: MetaWorld EVF Training
intent:
  - train a MetaWorld policy
  - run pick-place or other MetaWorld tasks
required_inputs:
  - task or envName
  - training goal
  - budget or max runtime
recommended_defaults:
  provider: autodl
  gpuSkuHint: 4090-class
  imageHint: pytorch-cu121
  smokeTest: true
backend:
  workflow: evf_metaworld
  providerActions:
    - AI配置训练
    - GPU规格查询
    - AutoDL镜像查询
    - 开始训练
    - 查询状态
    - 结果下载
diagnostics:
  logs:
    - run.log
    - loss.txt
    - eval_info.json
```

The AI should never directly submit a raw command unless the platform explicitly enables raw command mode.

## Core Skills

### 1. Training Intake

Purpose: turn vague user intent into structured training requirements.

Questions to resolve:

- What benchmark or project is this? Custom project, MetaWorld, LIBERO, LeRobot, or other?
- Is the goal a smoke test, baseline, reproduction, or formal training run?
- Where is the code and data?
- What metric matters? Success rate, loss, reward, video behavior, or artifact existence?
- What budget or max runtime is acceptable?
- Does the user need checkpoint, video, loss, or evaluation JSON?

Output:

- selected skill;
- missing fields;
- recommended smoke test;
- recommended GPU SKU class;
- recommended image class;
- estimated minimum cost.

### 2. Environment Selection

Purpose: choose compute and image separately.

Flow:

1. Call `GPU规格查询` to list available compute products.
2. Call `AutoDL镜像查询` to list available environments.
3. Recommend a pair based on task type:
   - smoke test: cheaper GPU, shorter runtime;
   - formal training: stronger GPU and longer budget;
   - simulation-heavy workloads: prefer known-good robotics image.
4. Confirm `skuId`, `imageId`, and estimated first-hour charge with the user.

Backend mapping:

```json
{
  "skuId": "sku-4090",
  "imageId": "pytorch-cu121"
}
```

The backend maps these to provider fields such as `gpu_spec_uuid`, `image_uuid`, and `cuda_v_from`.

### 3. Plan Materialization

Purpose: compile a confirmed training plan into a backend request.

The AI should produce:

- workflow name;
- provider;
- `skuId`;
- `imageId`;
- workflow params;
- expected artifact paths;
- budget warning;
- explicit confirmation state.

Example request:

```json
{
  "username": "pearl",
  "taskName": "metaworld-smoke-001",
  "action": "开始训练",
  "provider": "autodl",
  "skuId": "sku-4090",
  "imageId": "pytorch-cu121",
  "workflow": "evf_metaworld",
  "params": {
    "envName": "pick-place-v2",
    "epochs": 5,
    "evalEpisodes": 2,
    "saveVideo": true
  }
}
```

### 4. Runtime Monitor

Purpose: keep the user informed while preserving cost control.

Loop:

- call `查询状态`;
- call `账单查询`;
- warn when balance is near the next hour freeze;
- call `请求用户日志` when status is failed or unknown;
- stop task if wallet policy requires it.

The AI should summarize status in user language, not backend language.

Example:

```text
任务还在 Running。当前已冻结首小时费用，下一次计费检查前建议保持余额大于 1 小时费用。日志暂未出现失败信号。
```

### 5. Diagnosis

Purpose: classify failures and suggest next actions.

Inputs:

- provider status;
- last error;
- run log;
- loss file;
- eval info;
- wallet and billing records;
- workflow params.

Failure classes:

- insufficient balance;
- provider instance failed to start;
- SSH or image bootstrap failure;
- dependency installation failure;
- dataset path missing;
- CUDA or driver mismatch;
- GPU OOM;
- training command failed;
- evaluation produced zero success;
- artifact path missing.

Output:

- likely cause;
- evidence from logs;
- recommended fix;
- whether retry should reuse same GPU/image;
- whether a smoke test should be run first.

### 6. Artifact Review

Purpose: help users understand results instead of only downloading files.

Flow:

1. Call `结果下载` or `下载损失`.
2. Parse available metrics if present.
3. Summarize:
   - success rate;
   - loss behavior;
   - checkpoints;
   - videos;
   - notable warnings.
4. Suggest next run configuration.

## Skill Lifecycle

Every skill should follow this loop:

```text
Understand intent
  -> ask missing questions
  -> build plan
  -> validate GPU/image/budget
  -> confirm
  -> start training
  -> monitor
  -> diagnose or summarize results
```

Do not skip confirmation before starting paid compute.

## Backend Action Mapping

| AI skill need | TCP action |
| --- | --- |
| Query wallet | `余额查询` |
| Query billing | `账单查询` |
| List GPU products | `GPU规格查询` |
| List AutoDL images | `AutoDL镜像查询` |
| Generate plan | `AI配置训练` |
| Start task | `开始训练` |
| Sync tasks | `任务同步` |
| Query status | `查询状态` |
| Query logs | `请求用户日志` |
| Query artifacts | `查询下载目录` |
| Download artifacts | `结果下载` |
| Download loss | `下载损失` |
| Stop task | `结束训练` |
| Delete task | `删除任务` |

## Guardrails

- Raw commands are disabled by default.
- Paid compute must require explicit user confirmation.
- GPU SKU and image must be selected from backend-provided lists.
- The user should see sale price, not provider internals.
- Admin-only fields such as AutoDL token, image UUID, and GPU spec UUID must not be exposed as editable user inputs.
- The first run should usually be a smoke test unless the user explicitly asks for a full run.
- Long training must include balance and runtime warnings.
- Failure diagnosis must cite evidence from logs or task metadata.

## Recommended Initial Skill Set

Start with these first:

1. `training-intake`
2. `environment-selection`
3. `custom-project-training`
4. `evf-metaworld-training`
5. `evf-libero-training`
6. `runtime-monitor`
7. `failure-diagnosis`
8. `artifact-review`

These map cleanly to the provider runtime and billing architecture already present in the current branch.

