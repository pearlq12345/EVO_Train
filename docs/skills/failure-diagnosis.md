# Skill: Failure Diagnosis

## Purpose

Classify training failures from evidence and recommend the smallest useful fix.

## When To Use

Use this skill when:

- task status is Failed, Unknown, stopped unexpectedly, or stuck;
- evaluation success rate is zero;
- artifacts are missing;
- logs show dependency, dataset, CUDA, memory, or command errors.

## Evidence Order

Collect evidence in this order:

1. `查询状态`
2. `请求用户日志`
3. `下载损失`
4. `结果下载` for `eval_info.json` or other metrics
5. `账单查询`
6. original workflow params, `skuId`, and `imageId`

## Failure Classes

Use one primary class:

- insufficient balance;
- provider instance startup failure;
- SSH or connection failure;
- image or dependency mismatch;
- dataset path missing;
- CUDA or driver mismatch;
- GPU OOM;
- training command failed;
- evaluation failed or zero success;
- artifact path missing;
- billing stop.

## Diagnosis Loop

1. State the observed symptom.
2. List 3-5 hypotheses, ranked.
3. Test the top hypothesis with available evidence.
4. Name the most likely class.
5. Recommend a fix.
6. Recommend a regression guard, usually a smoke test.

## Output Shape

```json
{
  "skill": "failure-diagnosis",
  "primaryClass": "dataset path missing",
  "evidence": ["run.log contains FileNotFoundError for /data/demo"],
  "recommendedFix": "Update datasetPath or run the data preparation stage.",
  "nextRun": {
    "goal": "smoke-test",
    "epochs": 1
  }
}
```

## Guardrails

- Do not guess without evidence.
- If evidence is missing, say which artifact or log is missing.
- Do not recommend a larger GPU unless logs indicate OOM or resource pressure.
- Do not retry full training before a smoke test confirms the fix.

