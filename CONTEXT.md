# RoboClaw Training Center Context

This file defines the shared language for RoboClaw training-center work. It is a glossary, not a product spec and not an implementation plan.

## Language

User: A person using RoboClaw to configure, run, monitor, or diagnose training tasks.

Admin: A platform operator who configures provider credentials, GPU products, images, pricing, and billing policy.

Training Task: A user-visible unit of work that starts from a confirmed plan and ends with success, failure, stop, or deletion.

Training Plan: A structured AI-generated preview of what will run, which provider will execute it, which GPU and image are recommended, estimated cost, missing inputs, and warnings.

Skill: A reusable AI-guided capability that turns user intent into a Training Plan, monitor loop, diagnosis, or artifact review.

Workflow: A backend recipe that materializes a Training Plan into staged commands and paths.

Provider: An execution backend that can submit, query, stop, and download artifacts for a Training Task.

GPU SKU: A RoboClaw compute product shown to users. It hides provider-specific GPU identifiers and carries cost/sale pricing.

Image: A reusable training environment shown to users. It hides provider-specific image identifiers and CUDA requirements.

Wallet: A user's RoboClaw balance used for first-hour freeze, hourly continuation, settlement, and refunds.

Billing Record: An immutable accounting event for wallet changes, freezes, charges, refunds, or admin adjustments.

Artifact: Any user-consumable output from training, including checkpoints, logs, videos, metrics, evaluation JSON, and loss files.

Smoke Test: A short validation run that proves code, data, image, GPU, logging, and artifact paths work before a full run.

Diagnosis: A structured failure investigation that starts from a reproducible signal and ends with evidence, cause, fix, and regression guard.

## Relationships

- A User owns many Training Tasks.
- A Training Task uses one Provider.
- A Training Task may use one GPU SKU and one Image.
- A Skill produces or interprets a Training Plan.
- A Workflow materializes a Training Plan.
- A Provider executes a materialized request.
- A Wallet constrains whether a Training Task may start or continue.
- Artifacts belong to a Training Task.

## Flagged Ambiguities

- "GPU" can mean display name, provider GPU spec, or product SKU. Use GPU SKU for the user-facing product and provider GPU spec for backend identifiers.
- "Image" can mean a Docker image string or an AutoDL image UUID. Use Image for the user-facing environment and provider image ID for backend identifiers.
- "Task" can mean a benchmark task or a training job. Use Training Task for backend jobs and benchmark task for MetaWorld/LIBERO task names.
- "Cost" can mean provider cost or user sale price. Use cost price for provider spend and sale price for what users pay.

