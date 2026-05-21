from __future__ import annotations

from typing import Any

from .model_catalog import resolve_builtin_model


DATASET_SOURCE_TYPES = {
    "platform_dataset",
    "public_reference",
    "user_object_storage",
    "builtin_benchmark",
}

MODEL_SOURCE_TYPES = {
    "builtin_policy",
    "evo_studio_checkpoint",
    "public_model_repo",
    "user_object_storage",
    "from_scratch",
}


def _source_type(source: dict[str, Any]) -> str:
    return str(source.get("sourceType") or source.get("kind") or source.get("type") or "").strip()


def _source_value(source: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _copy_source(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def normalize_training_sources(params: dict[str, Any]) -> dict[str, Any]:
    """Map Evo Studio source contracts to legacy workflow fields.

    RoboClaw's newer frontend sends structured `datasetSource` and `modelSource`
    contracts. Existing EVO_Train recipes still consume flat fields such as
    `datasetPath`, `checkpointPath`, `modelFamily`, and `suite`. This function is
    intentionally conservative: it preserves the source contract for auditability
    while filling only the flat fields needed by current launchers.
    """

    normalized = dict(params)
    warnings: list[str] = []
    missing_fields: list[str] = []

    dataset_source = _copy_source(params.get("datasetSource"))
    if dataset_source:
        dataset_kind = _source_type(dataset_source)
        normalized["datasetSource"] = dataset_source
        normalized.setdefault("datasetSourceKind", dataset_kind)
        dataset_format = _source_value(dataset_source, "format", "datasetFormat") or str(
            params.get("datasetFormat") or "auto"
        )
        normalized.setdefault("datasetFormat", dataset_format)

        if dataset_kind not in DATASET_SOURCE_TYPES:
            missing_fields.append("datasetSource.sourceType")
            warnings.append(f"unsupported datasetSource.sourceType: {dataset_kind or '<empty>'}")
        elif dataset_kind == "platform_dataset":
            dataset_id = _source_value(dataset_source, "datasetId", "id", "datasetName")
            uri = _source_value(dataset_source, "uri", "path", "datasetPath")
            if dataset_id:
                normalized.setdefault("datasetName", dataset_id)
            if uri:
                normalized.setdefault("datasetPath", uri)
            else:
                missing_fields.append("datasetSource.uri")
                warnings.append("platform_dataset requires a resolved cloud uri or runtime path")
        elif dataset_kind == "builtin_benchmark":
            benchmark = _source_value(dataset_source, "benchmark", "suite", "name")
            suite = _source_value(dataset_source, "suite", "taskSuite") or benchmark
            if benchmark:
                normalized.setdefault("benchmark", benchmark)
            if suite:
                normalized.setdefault("suite", suite)
                normalized.setdefault("taskSuite", suite)
            if not benchmark and not suite:
                missing_fields.append("datasetSource.benchmark")
        elif dataset_kind == "public_reference":
            uri = _source_value(dataset_source, "uri", "repo", "repoId", "url", "datasetPath")
            if uri:
                normalized.setdefault("datasetPath", uri)
            else:
                missing_fields.append("datasetSource.uri")
            auth_ref = _source_value(dataset_source, "authRef")
            if auth_ref:
                normalized.setdefault("datasetAuthRef", auth_ref)
            warnings.append(
                "public_reference is passed as a URI; the cloud worker or launcher must resolve/cache it"
            )
        elif dataset_kind == "user_object_storage":
            uri = _source_value(dataset_source, "uri", "path", "datasetPath")
            auth_ref = _source_value(dataset_source, "authRef")
            if uri:
                normalized.setdefault("datasetPath", uri)
            else:
                missing_fields.append("datasetSource.uri")
            if auth_ref:
                normalized.setdefault("datasetAuthRef", auth_ref)
            else:
                missing_fields.append("datasetSource.authRef")
                warnings.append(
                    "user_object_storage dataset sources require an authRef; do not pass raw credentials"
                )

    model_source = _copy_source(params.get("modelSource"))
    if model_source:
        model_kind = _source_type(model_source)
        normalized["modelSource"] = model_source
        normalized.setdefault("modelSourceKind", model_kind)
        checkpoint_format = _source_value(model_source, "format", "checkpointFormat") or str(
            params.get("checkpointFormat") or "auto"
        )
        normalized.setdefault("checkpointFormat", checkpoint_format)
        model_family = _source_value(model_source, "modelFamily", "policyType")
        if model_family:
            normalized.setdefault("modelFamily", model_family)
            normalized.setdefault("policyFamily", model_family)

        if model_kind not in MODEL_SOURCE_TYPES:
            missing_fields.append("modelSource.sourceType")
            warnings.append(f"unsupported modelSource.sourceType: {model_kind or '<empty>'}")
        elif model_kind == "builtin_policy":
            resolved = resolve_builtin_model(model_family)
            if resolved:
                normalized["resolvedModelSource"] = resolved
                resolved_family = _source_value(resolved, "modelFamily")
                if resolved_family:
                    normalized["modelFamily"] = resolved_family
                    normalized.setdefault("policyFamily", resolved_family)
                resolved_format = _source_value(resolved, "checkpointFormat")
                if resolved_format:
                    if normalized.get("checkpointFormat") in (None, "", "auto"):
                        normalized["checkpointFormat"] = resolved_format
                if resolved.get("kind") == "pretrained_checkpoint":
                    normalized.setdefault("checkpointPath", _source_value(resolved, "uri"))
                elif resolved.get("warning"):
                    warnings.append(str(resolved["warning"]))
                elif resolved.get("kind") == "unknown":
                    warnings.append(
                        f"builtin_policy modelFamily {model_family or '<empty>'} has no catalog entry"
                    )
        elif model_kind == "evo_studio_checkpoint":
            checkpoint = _source_value(model_source, "checkpoint", "checkpointPath", "uri", "path")
            if checkpoint:
                normalized.setdefault("checkpointPath", checkpoint)
            else:
                missing_fields.append("modelSource.checkpoint")
        elif model_kind in {"public_model_repo", "user_object_storage"}:
            uri = _source_value(model_source, "uri", "repo", "repoId", "url", "checkpointPath")
            auth_ref = _source_value(model_source, "authRef")
            if uri:
                normalized.setdefault("checkpointPath", uri)
            else:
                missing_fields.append("modelSource.uri")
            if auth_ref:
                normalized.setdefault("modelAuthRef", auth_ref)
            elif model_kind == "user_object_storage":
                missing_fields.append("modelSource.authRef")
                warnings.append(
                    "user_object_storage model sources require an authRef; do not pass raw credentials"
                )
        elif model_kind == "from_scratch":
            normalized.setdefault("trainFromScratch", True)

    source_contract = {
        "datasetSource": dataset_source,
        "modelSource": model_source,
        "datasetSourceKind": normalized.get("datasetSourceKind", ""),
        "modelSourceKind": normalized.get("modelSourceKind", ""),
        "datasetFormat": normalized.get("datasetFormat", ""),
        "checkpointFormat": normalized.get("checkpointFormat", ""),
        "datasetAuthRef": normalized.get("datasetAuthRef", ""),
        "modelAuthRef": normalized.get("modelAuthRef", ""),
        "resolvedModelSource": normalized.get("resolvedModelSource", {}),
    }
    if any(value for value in source_contract.values()):
        normalized["sourceContract"] = source_contract

    if warnings:
        normalized["sourceWarnings"] = (
            [*params.get("sourceWarnings", []), *warnings]
            if isinstance(params.get("sourceWarnings"), list)
            else warnings
        )
    if missing_fields:
        normalized["sourceMissingFields"] = (
            [*params.get("sourceMissingFields", []), *missing_fields]
            if isinstance(params.get("sourceMissingFields"), list)
            else missing_fields
        )
    return normalized


def source_warnings(params: dict[str, Any]) -> list[str]:
    value = params.get("sourceWarnings")
    return [str(item) for item in value] if isinstance(value, list) else []


def source_missing_fields(params: dict[str, Any]) -> list[str]:
    value = params.get("sourceMissingFields")
    return [str(item) for item in value] if isinstance(value, list) else []
