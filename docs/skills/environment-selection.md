# Skill: Environment Selection

## Purpose

Help the user select compute and environment independently through platform-managed GPU SKUs and AutoDL images.

## When To Use

Use this skill after the training target is roughly known and before starting a paid task.

## Backend Actions

Call:

- `GPU规格查询`
- `AutoDL镜像查询`
- `余额查询`

Do not ask users to provide provider internals such as AutoDL token, `gpu_spec_uuid`, `image_uuid`, or SSH information.

## Flow

1. Query available GPU SKUs.
2. Query available images.
3. Filter out entries where `readyToStart` is false.
4. Recommend one cheap smoke-test pair and one stronger full-run pair when possible.
5. Show first-hour sale price and service-fee context.
6. Ask for confirmation of `skuId` and `imageId`.

## Recommendation Rules

- Smoke test: pick the cheapest ready GPU SKU that can run the image.
- Full run: pick a stronger GPU only if the user has budget and the benchmark benefits from it.
- Custom project: prefer a general PyTorch CUDA image unless the repo states otherwise.
- LIBERO / RoboSuite: prefer a known robotics image with MuJoCo / robosuite already installed.
- If no ready GPU or no ready image exists, stop and ask the admin to configure catalogs.

## Output Shape

```json
{
  "skill": "environment-selection",
  "recommended": {
    "skuId": "sku-4090",
    "imageId": "pytorch-cu121",
    "hourlyPriceCents": "992",
    "reason": "Good smoke-test balance for this workload."
  },
  "alternatives": [],
  "readyToStart": true
}
```

## Done Criteria

- The user has a selected `skuId`.
- The user has a selected `imageId`.
- The selected entries came from backend query results.
- The first-hour cost is visible.

