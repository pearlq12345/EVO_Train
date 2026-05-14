# EVO_Train / RoboClaw Top-Level Architecture

## Intent

This document defines the top-level architecture boundary between RoboClaw and EVO_Train.

RoboClaw is the user-facing training center and AI interaction layer.
EVO_Train is the backend training execution engine.

The key rule is:

```text
RoboClaw decides what should happen.
EVO_Train executes it safely.
Providers supply compute.
Billing constrains cost.
Skills keep the user interaction repeatable.
```

## Architecture Layers

```text
┌──────────────────────────────────────────────────────────────┐
│ RoboClaw Product Layer                                        │
│ - chat / UI / training center                                 │
│ - user confirmation                                           │
│ - task presentation                                           │
│ - calls EVO_Train TCP actions                                 │
└──────────────────────────────┬───────────────────────────────┘
                               │ JSON request + "\n"
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ RoboClaw AI Skills Layer                                      │
│ - training-intake                                             │
│ - environment-selection                                       │
│ - runtime-monitor                                             │
│ - failure-diagnosis                                           │
│ - artifact-review                                             │
└──────────────────────────────┬───────────────────────────────┘
                               │ workflow + params + skuId + imageId
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ EVO_Train TCP Runtime                                         │
│ - server_connection.py reactor                                │
│ - thread_pool.py lite/download workers                        │
│ - server_function.py action dispatch                          │
└──────────────────────────────┬───────────────────────────────┘
                               │ materialized training request
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ EVO_Train Control Plane                                       │
│ - workflow registry                                           │
│ - billing / wallet / pricing                                  │
│ - GPU SKU catalog                                             │
│ - AutoDL image catalog                                        │
│ - provider factory                                            │
└──────────────────────────────┬───────────────────────────────┘
                               │ provider job config
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Provider Runtime                                              │
│ - AutoDL managed instance lifecycle                           │
│ - Aliyun DLC adapter                                          │
│ - submit / metadata / stop / artifact download                │
└──────────────────────────────┬───────────────────────────────┘
                               │ remote compute
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Training Execution                                            │
│ - code setup                                                  │
│ - data preparation                                            │
│ - train                                                       │
│ - evaluate                                                    │
│ - collect artifacts / loss                                    │
└──────────────────────────────────────────────────────────────┘
```

## Ownership Boundaries

| Layer | Owns | Must not own |
| --- | --- | --- |
| RoboClaw Product | UI, chat, confirmation, user-facing summaries | provider tokens, raw provider IDs, billing invariants |
| AI Skills | questions, plans, recommendations, diagnosis | direct provider calls, hidden billing changes |
| TCP Runtime | sockets, queueing, action routing, response shape | training semantics, provider-specific rules |
| Control Plane | workflow materialization, wallet, catalogs, auth | UI state, natural language policy |
| Provider Runtime | cloud API details, SSH, artifacts, lifecycle | product decisions, skill dialogue |
| Storage | task/wallet/billing persistence | provider API behavior |

## Request Flow

### Planning Flow

```text
User intent
  -> RoboClaw AI skill
  -> AI配置训练
  -> Training Plan
  -> missing fields / warnings / cost estimate
  -> user confirmation
```

Planning does not start paid compute.

### Environment Flow

```text
RoboClaw
  -> GPU规格查询
  -> AutoDL镜像查询
  -> user chooses skuId + imageId
  -> backend maps to provider identifiers
```

Normal users choose `skuId` and `imageId`, not `gpu_spec_uuid` or `image_uuid`.

### Execution Flow

```text
开始训练
  -> auth check
  -> workflow materialization
  -> GPU SKU / Image mapping
  -> first-hour wallet freeze
  -> provider submit
  -> task metadata persisted
```

If any step before provider submit fails, the user wallet must not remain frozen.

### Runtime Flow

```text
任务同步 / 查询状态
  -> provider metadata
  -> SQLite task update
  -> billing reconciliation
  -> optional low-balance stop
```

### Artifact Flow

```text
结果下载 / 下载损失
  -> download worker queue
  -> provider chunk download
  -> JSON chunk response
```

EVO_Train keeps the same TCP JSON request protocol. It does not require RoboClaw to run a separate file server.

## Top-Level Invariants

1. Paid compute requires an explicit confirmed plan.
2. Raw commands are disabled unless explicitly enabled for trusted debugging.
3. User-facing GPU choice is always a GPU SKU.
4. User-facing environment choice is always an Image.
5. Provider tokens and provider UUIDs stay server-side.
6. Wallet freeze happens before provider submit.
7. Failed submit paths refund or avoid frozen balance.
8. Download-heavy actions run in the download worker queue.
9. Diagnosis uses evidence from logs, loss, metrics, task metadata, or billing records.
10. Skills should recommend a smoke test before full training.

## How To Extend Without Making The Architecture Messy

Add a new benchmark or training style by adding:

1. one Skill spec under `docs/skills/`;
2. one Workflow recipe under `train/workflows/`;
3. tests for plan materialization and task creation;
4. optional diagnosis rules if the workflow has unique failure modes.

Do not add provider-specific logic inside skills.
Do not add AI prompt logic inside providers.
Do not add billing changes inside workflows.
Do not ask RoboClaw users to enter raw provider identifiers.

## Current Extension Points

| Extension point | Path |
| --- | --- |
| TCP action dispatch | `train/server_function.py` |
| Workflow registry | `train/workflows/registry.py` |
| Provider factory | `train/platform/factory.py` |
| AutoDL provider | `train/platform/autodl.py` |
| Billing scheduler | `train/billing_scheduler.py` |
| SQLite persistence | `sql_lite/sql_pack.py` |
| AI skill architecture | `docs/roboclaw-ai-skills.md` |
| Shared language | `CONTEXT.md` |
| AutoDL runbook | `docs/runbooks/autodl-provider.md` |

## PR Review Checklist

When reviewing architecture changes, check:

- Does this change stay inside one layer?
- If it crosses layers, is there a clear interface?
- Does it expose provider internals to RoboClaw users?
- Does it preserve wallet and billing invariants?
- Does it keep download-heavy work out of lite workers?
- Does it add or update a feedback loop?
- Does it need a new Skill, Workflow, Provider adapter, or ADR?

