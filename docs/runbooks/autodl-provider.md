# AutoDL Provider Runbook

This runbook explains the minimum configuration needed to verify the AutoDL provider path without exposing AutoDL internals to normal users.

## Roles

- Admin configures `AUTODL_TOKEN`, GPU SKUs, images, platform balance thresholds, and user balance.
- User selects `skuId` and `imageId`, then confirms a Training Plan.
- Backend maps `skuId + imageId` to AutoDL `gpu_spec_uuid`, `image_uuid`, and `cuda_v_from`.

## Required Admin Configuration

```bash
export AUTODL_TOKEN='<autodl-developer-token>'
export EVO_TRAIN_SERVICE_FEE_RATE='0.10'
export EVO_TRAIN_ADMIN_TOKEN='<admin-token>'
export AUTODL_MIN_ASSETS=1000
```

`AUTODL_MIN_ASSETS` is checked against the platform AutoDL wallet balance before creating managed instances.

## GPU SKU Catalog

GPU SKU is the user-facing compute product. It should not contain provider secrets or image choices.

```bash
export AUTODL_GPU_SKUS_JSON='[
  {
    "skuId": "sku-4090d",
    "provider": "autodl",
    "displayName": "RTX 4090D 1卡",
    "gpuSpec": "4090d",
    "autodlGpuSpecUuid": "4090D",
    "gpuCount": 1,
    "costHourlyCents": 900,
    "autodlDataCenters": "westDC3,beijingDC2"
  }
]'
```

Pricing rule:

```text
saleHourlyCents = ceil(costHourlyCents * (1 + EVO_TRAIN_SERVICE_FEE_RATE))
```

With the default 10% service fee, `900` cost cents becomes `990` sale cents.

## Image Catalog

Image is the user-facing training environment. Keep it separate from GPU SKU so one GPU can support many environments.

```bash
export AUTODL_IMAGES_JSON='[
  {
    "imageId": "pytorch-cu121",
    "displayName": "PyTorch CUDA 12.1",
    "autodlImageUuid": "image-xxxx",
    "cudaVFrom": 121
  }
]'
```

`imageId` is a RoboClaw identifier. `autodlImageUuid` is the provider identifier from AutoDL.

## Start The TCP Server

```bash
python3 server_tcp/server_connection.py \
  --host 127.0.0.1 \
  --port 9000 \
  --workers 4
```

Use `--host 0.0.0.0` only for a controlled network environment with authentication configured.

## Smoke Test Sequence

### 1. Query GPU SKUs

```bash
printf '%s\n' '{"action":"GPU规格查询","provider":"autodl"}' | nc 127.0.0.1 9000
```

Expected:

```json
{
  "message": "gpu sku query success",
  "provider": "autodl",
  "skus": [
    {
      "skuId": "sku-4090d",
      "readyToStart": true
    }
  ]
}
```

If `skus` is empty, the catalog is incomplete or disabled.

### 2. Query Images

```bash
printf '%s\n' '{"action":"AutoDL镜像查询"}' | nc 127.0.0.1 9000
```

Expected:

```json
{
  "message": "autodl image query success",
  "images": [
    {
      "imageId": "pytorch-cu121",
      "readyToStart": true
    }
  ]
}
```

### 3. Check Platform Balance

```bash
printf '%s\n' '{"action":"平台余额查询","provider":"autodl","minimumAssets":1000,"adminToken":"<admin-token>"}' | nc 127.0.0.1 9000
```

Expected:

```json
{
  "message": "platform balance query success",
  "lowBalance": false
}
```

### 4. Give A User Test Balance

```bash
printf '%s\n' '{"username":"pearl","action":"管理员充值","balanceCents":10000,"adminToken":"<admin-token>"}' | nc 127.0.0.1 9000
```

### 5. Start A Small Custom Project Smoke Task

```bash
printf '%s\n' '{
  "username": "pearl",
  "taskName": "autodl-smoke-001",
  "action": "开始训练",
  "provider": "autodl",
  "skuId": "sku-4090d",
  "imageId": "pytorch-cu121",
  "workflow": "custom_project",
  "params": {
    "repoUrl": "https://github.com/example/project.git",
    "branch": "main",
    "setupCommand": "python -m pip install -r requirements.txt",
    "trainCommand": "python train.py --epochs 1",
    "evalCommand": "true",
    "artifactPath": "/root/autodl-tmp/custom_project/outputs"
  }
}' | nc 127.0.0.1 9000
```

Expected:

```json
{
  "message": "create task success"
}
```

## Existing Instance Mode

For early debugging, an admin can reuse an existing AutoDL instance instead of creating a new managed instance.

```json
{
  "provider": "autodl",
  "autodlInstanceUuid": "defa41a7c0-015b874b",
  "hourlyPriceCents": 990
}
```

This is useful for smoke tests, but product users should normally use `skuId + imageId`.

## Common Failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `GPU规格查询` returns empty list | GPU SKU lacks `autodlGpuSpecUuid` or price | Complete `AUTODL_GPU_SKUS_JSON` |
| `AutoDL镜像查询` returns empty list | image lacks `autodlImageUuid` or `cudaVFrom` | Complete `AUTODL_IMAGES_JSON` |
| `Missing required value: AUTODL_TOKEN` | managed AutoDL path is enabled but token is absent | Set `AUTODL_TOKEN` |
| `platform balance is low` | AutoDL wallet is below `AUTODL_MIN_ASSETS` | Recharge AutoDL or lower threshold |
| `insufficient balance` | RoboClaw user wallet cannot cover first hour | Admin recharge or lower SKU price |
| task starts but artifacts missing | wrong `artifactPath` or command output path | Run a smoke test and inspect logs |

## Review Checklist

- GPU SKU and Image are separate.
- Normal users never edit provider UUIDs.
- Sale price includes the configured service fee.
- First paid run is a smoke test.
- User wallet and platform AutoDL wallet are both checked.
- Failure diagnosis has logs, loss, or task metadata to inspect.

