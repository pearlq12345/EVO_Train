# Skill: Training Intake

## Purpose

Turn a user's vague training request into a structured Training Plan draft with missing fields, risks, and next questions.

## When To Use

Use this skill when the user says things like:

- "I want to train a robot policy."
- "Run LIBERO / MetaWorld for me."
- "I have a GitHub repo and want to train it."
- "Help me reproduce this training."
- "I do not know which GPU or image to choose."

## Inputs To Collect

Ask only for missing information. Prefer concise follow-up questions.

Required:

- training target: custom project, MetaWorld, LIBERO, or other;
- goal: smoke test, baseline, reproduction, full run, or debugging;
- code source or benchmark name;
- dataset source or expected dataset path;
- output expectations: checkpoint, video, metric, loss, or logs;
- budget or max runtime.

Optional:

- preferred provider;
- preferred GPU class;
- preferred image;
- expected success metric;
- whether previous logs or failed runs exist.

## Decision Rules

- If the user is uncertain, default to a smoke test.
- If code or data is unclear, do not start training.
- If the user asks for a full run without prior validation, recommend a smoke test first.
- If the task is custom code, prefer `custom_project`.
- If the prompt mentions MetaWorld task names, prefer `evf_metaworld`.
- If the prompt mentions LIBERO suites, prefer `evf_libero`.

## Output Shape

Return a Training Plan draft:

```json
{
  "skill": "training-intake",
  "workflow": "custom_project",
  "goal": "smoke-test",
  "missingFields": ["repoUrl", "datasetPath"],
  "recommendedNextSkill": "environment-selection",
  "warnings": ["Start with a short smoke test before a full run."]
}
```

## Backend Actions

This skill usually prepares for, but does not directly call:

- `AI配置训练`
- `GPU规格查询`
- `AutoDL镜像查询`

It must not call `开始训练`.

## Done Criteria

- The target workflow is selected or the ambiguity is explicitly named.
- Missing fields are listed.
- The next question is clear.
- Paid compute has not started.

