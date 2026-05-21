from __future__ import annotations

import hashlib
import posixpath
import re
import shlex
from dataclasses import dataclass
from urllib.parse import urlparse


ARCHIVE_EXTENSIONS = (".tar", ".tar.gz", ".tgz", ".zip")
FILE_EXTENSIONS = ARCHIVE_EXTENSIONS + (
    ".hdf5",
    ".h5",
    ".jsonl",
    ".parquet",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
)


@dataclass(frozen=True)
class SourceResolution:
    role: str
    original_uri: str
    kind: str
    strategy: str
    resolved_path: str
    command: str = ""
    auth_ref: str = ""
    missing_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_contract(self) -> dict[str, object]:
        return {
            "role": self.role,
            "originalUri": self.original_uri,
            "kind": self.kind,
            "strategy": self.strategy,
            "resolvedPath": self.resolved_path,
            "authRef": self.auth_ref,
            "missingFields": list(self.missing_fields),
            "warnings": list(self.warnings),
        }


def resolve_source_uri(
    uri: str,
    *,
    role: str,
    cache_root: str,
    auth_ref: str = "",
    source_type: str = "",
    source_format: str = "",
) -> SourceResolution:
    text = str(uri or "").strip()
    if not text:
        return SourceResolution(role=role, original_uri="", kind="empty", strategy="empty", resolved_path="")

    if _is_local_path(text):
        resolved = text.removeprefix("file://")
        return SourceResolution(
            role=role,
            original_uri=text,
            kind="local_path",
            strategy="pass_through",
            resolved_path=resolved,
        )

    destination = _destination_path(cache_root, role, text)
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    warnings: list[str] = []
    missing_fields: list[str] = []
    command = ""
    kind = scheme or "unknown"
    strategy = "stage_to_local"

    if scheme in {"hf", "huggingface"} or _is_huggingface_url(text):
        repo_id, repo_type = _huggingface_repo(text, role)
        command = _huggingface_command(repo_id, repo_type, destination)
        kind = "huggingface"
    elif scheme == "modelscope" or "modelscope.cn/" in text:
        repo_id = _repo_tail(text, scheme="modelscope")
        command = _modelscope_command(repo_id, destination)
        kind = "modelscope"
    elif scheme == "kaggle" or "kaggle.com/datasets/" in text:
        dataset_id = _repo_tail(text, scheme="kaggle")
        command = _kaggle_command(dataset_id, destination)
        kind = "kaggle"
    elif scheme in {"s3", "r2", "minio"}:
        command = _aws_s3_command(text, destination)
        kind = scheme
    elif scheme == "oss":
        command = _cli_copy_command("ossutil", ["cp", "-r", text, destination], destination)
        kind = "aliyun_oss"
    elif scheme == "cos":
        command = _cli_copy_command("coscli", ["cp", "-r", text, destination], destination)
        kind = "tencent_cos"
    elif scheme == "obs":
        command = _cli_copy_command("obsutil", ["cp", "-r", text, destination], destination)
        kind = "huawei_obs"
    elif scheme == "gs":
        command = _cli_copy_command("gsutil", ["-m", "cp", "-r", text, destination], destination)
        kind = "google_cloud_storage"
    elif scheme in {"http", "https"}:
        if _is_direct_file(text):
            command = _http_file_command(text, destination)
            kind = "http_file"
        elif _is_cloud_drive_url(text):
            strategy = "manual_connector"
            resolved_path = text
            missing_fields.append(f"{role}Source.authRef")
            warnings.append("cloud drive URLs require a connector, share token, or manual import before training")
            return SourceResolution(
                role=role,
                original_uri=text,
                kind="cloud_drive",
                strategy=strategy,
                resolved_path=resolved_path,
                auth_ref=auth_ref,
                missing_fields=tuple(missing_fields),
                warnings=tuple(warnings),
            )
        else:
            command = _http_reference_command(text, destination)
            kind = "http_reference"
            warnings.append("generic HTTP references are staged with curl; private pages need a connector/authRef")
    else:
        strategy = "pass_through"
        destination = text
        warnings.append(f"unrecognized source URI scheme {scheme or '<none>'}; passing through to launcher")

    if source_type == "user_object_storage" and not auth_ref and strategy != "pass_through":
        missing_fields.append(f"{role}Source.authRef")
    if auth_ref and command:
        command = _with_auth_env(command, auth_ref=auth_ref, kind=kind)
    if source_format == "custom":
        warnings.append("custom source format selected; launcher must know how to read the staged files")

    return SourceResolution(
        role=role,
        original_uri=text,
        kind=kind,
        strategy=strategy,
        resolved_path=destination,
        command=command,
        auth_ref=auth_ref,
        missing_fields=tuple(missing_fields),
        warnings=tuple(warnings),
    )


def _is_local_path(uri: str) -> bool:
    return uri.startswith("/") or uri.startswith("./") or uri.startswith("../") or uri.startswith("file://")


def _is_direct_file(uri: str) -> bool:
    return urlparse(uri).path.lower().endswith(FILE_EXTENSIONS)


def _is_huggingface_url(uri: str) -> bool:
    return "huggingface.co/" in uri


def _is_cloud_drive_url(uri: str) -> bool:
    lowered = uri.lower()
    return any(
        marker in lowered
        for marker in (
            "drive.google.com/",
            "docs.google.com/",
            "onedrive.live.com/",
            "1drv.ms/",
            "pan.baidu.com/",
            "alipan.com/",
            "aliyundrive.com/",
            "dropbox.com/",
        )
    )


def _destination_path(cache_root: str, role: str, uri: str) -> str:
    digest = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:12]
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", uri.strip().rstrip("/").split("/")[-1] or role).strip("-")
    return posixpath.join(cache_root.rstrip("/"), f"{role}s", f"{label[:36]}-{digest}")


def _repo_tail(uri: str, *, scheme: str) -> str:
    text = uri.strip()
    if "://" in text:
        parsed = urlparse(text)
        if parsed.netloc and parsed.path:
            return f"{parsed.netloc}{parsed.path}".strip("/")
        return parsed.path.strip("/")
    marker = f"{scheme}.com/"
    if marker in text:
        return text.split(marker, 1)[1].strip("/")
    return text.strip("/")


def _huggingface_repo(uri: str, role: str) -> tuple[str, str]:
    text = uri.strip()
    repo_type = "dataset" if role == "dataset" else "model"
    if text.startswith("hf://") or text.startswith("huggingface://"):
        parsed = urlparse(text)
        path = f"{parsed.netloc}{parsed.path}".strip("/")
        return path.removeprefix("datasets/").removeprefix("models/"), repo_type
    marker = "huggingface.co/"
    tail = text.split(marker, 1)[1].strip("/") if marker in text else text.strip("/")
    if tail.startswith("datasets/"):
        repo_type = "dataset"
        tail = tail.removeprefix("datasets/")
    elif tail.startswith("models/"):
        repo_type = "model"
        tail = tail.removeprefix("models/")
    return tail.split("/tree/", 1)[0].strip("/"), repo_type


def _quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _mkdir(destination: str) -> str:
    return _quote(["mkdir", "-p", destination])


def _cli_copy_command(binary: str, args: list[str], destination: str) -> str:
    return f"command -v {shlex.quote(binary)} >/dev/null && {_mkdir(destination)} && {_quote([binary, *args])}"


def _aws_s3_command(uri: str, destination: str) -> str:
    if _is_direct_file(uri):
        filename = posixpath.basename(urlparse(uri).path)
        return _cli_copy_command("aws", ["s3", "cp", uri, posixpath.join(destination, filename)], destination)
    return _cli_copy_command("aws", ["s3", "sync", uri, destination], destination)


def _huggingface_command(repo_id: str, repo_type: str, destination: str) -> str:
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo_id!r}, repo_type={repo_type!r}, local_dir={destination!r})"
    )
    return f"{_mkdir(destination)} && python -c {shlex.quote(code)}"


def _modelscope_command(repo_id: str, destination: str) -> str:
    code = (
        "from modelscope.hub.snapshot_download import snapshot_download; "
        f"snapshot_download({repo_id!r}, local_dir={destination!r})"
    )
    return f"{_mkdir(destination)} && python -c {shlex.quote(code)}"


def _kaggle_command(dataset_id: str, destination: str) -> str:
    return _cli_copy_command("kaggle", ["datasets", "download", "-d", dataset_id, "-p", destination, "--unzip"], destination)


def _http_file_command(uri: str, destination: str) -> str:
    filename = posixpath.basename(urlparse(uri).path) or "download"
    output = posixpath.join(destination, filename)
    unpack = ""
    lowered = filename.lower()
    if lowered.endswith((".tar", ".tar.gz", ".tgz")):
        unpack = f" && tar -xf {shlex.quote(output)} -C {shlex.quote(destination)}"
    elif lowered.endswith(".zip"):
        unpack = f" && unzip -oq {shlex.quote(output)} -d {shlex.quote(destination)}"
    return f"command -v curl >/dev/null && {_mkdir(destination)} && curl -L --fail {shlex.quote(uri)} -o {shlex.quote(output)}{unpack}"


def _http_reference_command(uri: str, destination: str) -> str:
    marker = posixpath.join(destination, "SOURCE_URL.txt")
    return f"{_mkdir(destination)} && printf %s {shlex.quote(uri)} > {shlex.quote(marker)}"


def _with_auth_env(command: str, *, auth_ref: str, kind: str) -> str:
    exports = _auth_env_exports(auth_ref, kind)
    if not exports:
        return command
    return f"{exports} && {command}"


def _auth_env_exports(auth_ref: str, kind: str) -> str:
    prefix = _auth_env_prefix(auth_ref)
    if not prefix:
        return ""
    env_names: list[tuple[str, str, bool]] = []
    if kind == "huggingface":
        env_names = [("HF_TOKEN", f"{prefix}_HF_TOKEN", True)]
    elif kind == "modelscope":
        env_names = [("MODELSCOPE_API_TOKEN", f"{prefix}_MODELSCOPE_API_TOKEN", True)]
    elif kind in {"s3", "r2", "minio"}:
        env_names = [
            ("AWS_ACCESS_KEY_ID", f"{prefix}_AWS_ACCESS_KEY_ID", True),
            ("AWS_SECRET_ACCESS_KEY", f"{prefix}_AWS_SECRET_ACCESS_KEY", True),
            ("AWS_DEFAULT_REGION", f"{prefix}_AWS_REGION", False),
        ]
    elif kind == "kaggle":
        env_names = [
            ("KAGGLE_USERNAME", f"{prefix}_KAGGLE_USERNAME", True),
            ("KAGGLE_KEY", f"{prefix}_KAGGLE_KEY", True),
        ]
    else:
        return ""
    parts = []
    for target, source, required in env_names:
        if required:
            value = f'"${{{source}:?missing {source} for authRef}}"'
        else:
            value = f'"${{{source}:-}}"'
        parts.append(f"export {target}={value}")
    return " && ".join(parts)


def _auth_env_prefix(auth_ref: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", auth_ref.strip()).strip("_").upper()
    return f"EVO_AUTH_{clean}" if clean else ""
