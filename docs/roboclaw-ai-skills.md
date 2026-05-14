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

## Feedback Loops

Every skill must create a fast pass/fail signal before it claims confidence. For RoboClaw, feedback loops are training-specific:

| Loop | When to use | Good signal |
| --- | --- | --- |
| Plan validation | Before paid compute | missing fields, selected `skuId`, selected `imageId`, estimated first-hour cost |
| Backend dry path | Before provider launch | request materializes without raw command or missing provider fields |
| Smoke Test | Before full training | task starts, logs stream, loss path exists, artifact path exists |
| Status loop | During training | `查询状态` changes predictably and billing remains valid |
| Artifact loop | After training | `结果下载` and `下载损失` return expected chunks |
| Diagnosis loop | After failure | one failure class is supported by logs or task metadata |

Treat the feedback loop itself as part of the product. A vague "task failed" response is not enough; the skill should sharpen the signal until it can say which layer failed.

## Diagnosis Discipline

Failure diagnosis should follow a fixed order:

1. Build or identify a feedback loop.
2. Reproduce the failure or collect the exact failed artifact.
3. Produce 3-5 ranked hypotheses.
4. Test one hypothesis at a time with targeted evidence.
5. Recommend a fix and a regression guard.
6. Clean up temporary instrumentation.

RoboClaw diagnosis should prefer evidence in this order:

1. provider status and instance lifecycle metadata;
2. task `error` field;
3. `run.log`;
4. `loss.txt`;
5. `eval_info.json`;
6. wallet and billing records;
7. workflow params and selected GPU/Image.

Do not say "probably CUDA" or "probably data issue" unless the evidence points there. If the evidence is missing, the recommendation should be "collect better evidence" and name the missing artifact.

## Vertical Slices

Build RoboClaw AI capability in vertical slices, not horizontal layers.

Bad:

```text
first build all skill prompts
then all UI
then all backend APIs
then tests
```

Good:

```text
one skill intent
  -> one query path
  -> one materialized request
  -> one smoke test
  -> one diagnosis path
```

Example vertical slices:

1. `environment-selection`: `GPU规格查询` + `AutoDL镜像查询` + recommendation + confirmation.
2. `custom-project-smoke`: repo URL + image + GPU SKU + one short training command + artifact check.
3. `loss-diagnosis`: failed task + `下载损失` + last log + classified failure.
4. `metaworld-smoke`: env name + 5 epochs + 2 eval episodes + result summary.

Each slice should be demoable without waiting for the entire training platform to be complete.

## Architecture Language

Use these architecture terms consistently when improving RoboClaw:

| Term | Meaning for RoboClaw |
| --- | --- |
| Module | Anything with an interface and implementation, from `AutoDLPlatform` to a training skill. |
| Interface | Everything callers must know: fields, order, errors, config, billing behavior, and performance expectations. |
| Seam | The place where behavior can vary without callers changing. Provider runtime and workflow registry are seams. |
| Adapter | A concrete implementation at a seam, such as AutoDL or Aliyun DLC. |
| Depth | How much useful behavior sits behind a small interface. |
| Locality | Whether changes and bugs are concentrated in one place. |

Use the deletion test before adding new abstractions:

```text
If this module is deleted, does complexity disappear, or does it reappear across many callers?
```

If complexity reappears across many callers, the module is earning its keep. If it disappears, the module was likely a pass-through.

## Deep Module Targets

RoboClaw should deliberately deepen these modules:

| Module | Desired interface | Complexity hidden behind it |
| --- | --- | --- |
| Training Skill | intent in, Training Plan out | questions, defaults, warnings, validation |
| Workflow | params in, staged command out | benchmark-specific command assembly |
| Provider | submit/query/stop/download | AutoDL/PAI API details, SSH, artifact packaging |
| Billing | freeze/charge/refund/settle | wallet invariants and hourly accounting |
| Diagnosis | task evidence in, fix recommendation out | log parsing, failure classification, retry advice |

This keeps AI prompts, UI, provider APIs, and billing from leaking into one another.

## ADR and Context Rules

Create or update `CONTEXT.md` when a term becomes canonical, especially for:

- task vs Training Task;
- GPU vs GPU SKU;
- image vs provider image ID;
- cost price vs sale price;
- smoke test vs full run.

Create an ADR only when all are true:

- the decision is hard to reverse;
- future contributors would wonder why it was chosen;
- real alternatives existed.

Do not use ADRs for temporary implementation notes.

## Handoff Rule

Every long-running RoboClaw AI session should leave a concise handoff when context is about to shift:

- current branch and PR;
- latest validation commands;
- what was decided in `CONTEXT.md` or ADRs;
- which skill should continue next;
- what evidence is still missing.

Do not duplicate full docs in handoff. Link to the relevant artifact paths instead.

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
