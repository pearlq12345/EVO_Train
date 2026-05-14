# ADR 0001: Use AI Skills as the Training Control Plane

## Status

Accepted

## Context

RoboClaw users often do not know the exact provider parameters, image identifiers, CUDA requirements, workflow commands, or budget trade-offs needed to run robotics training. Letting users submit raw commands or provider JSON directly creates repeated failure modes:

- wrong GPU or image selection;
- missing dataset, checkpoint, or artifact paths;
- unclear budget and billing behavior;
- logs that are hard to interpret after failure;
- provider-specific details leaking into the user experience.

The backend already has a provider runtime, billing, GPU SKU, image, workflow, and download layer. The missing product layer is a repeatable AI interaction model that turns user intent into safe backend requests.

## Decision

RoboClaw will treat AI skills as the training control plane.

Skills will be small, composable workflows that:

- ask targeted missing questions;
- use shared RoboClaw domain language;
- select GPU SKU and Image from backend-provided lists;
- generate a Training Plan before starting paid compute;
- require explicit confirmation before `开始训练`;
- prefer Smoke Tests before full training;
- monitor status and billing after launch;
- diagnose failures from evidence, not guesswork.

Raw command submission remains disabled by default and is reserved for trusted/admin scenarios.

## Consequences

Positive:

- users interact with training intent instead of provider internals;
- AI behavior becomes repeatable and reviewable;
- provider details stay behind backend seams;
- billing guardrails are easier to enforce;
- failure diagnosis can reuse consistent evidence and vocabulary.

Trade-offs:

- each new benchmark or training style needs a Skill and Workflow definition;
- admins must maintain GPU SKU and Image catalogs;
- AI must be judged by plan quality and diagnosis quality, not just command generation.

## Follow-ups

- Add concrete skill definitions for `training-intake`, `environment-selection`, `runtime-monitor`, and `failure-diagnosis`.
- Add smoke-test-first acceptance criteria to workflow docs.
- Add a small issue breakdown for implementing skill execution in the RoboClaw UI or agent layer.

