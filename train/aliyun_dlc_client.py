from pathlib import Path
import os

from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_tea_openapi.models import Config

ENV_FILES = (
    Path("/home/evomind/EVO_Train/.env"),
    Path("/home/evomind/evo-data_backend/.env"),
    Path(__file__).resolve().parents[1] / ".env",
)


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split(" #", 1)[0].strip().strip("\"").strip("'")
        os.environ.setdefault(key, value)


def load_env() -> None:
    for path in ENV_FILES:
        load_dotenv_file(path)


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def require_env(value: str | None, label: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {label}")
    return value


def create_dlc_client(region: str) -> Client:
    load_env()
    access_key_id = require_env(
        env_first("ALIBABA_CLOUD_ACCESS_KEY_ID", "ALIBABACLOUD_ACCESS_KEY_ID", "OSS_ACCESS_KEY_ID"),
        "ALIBABA_CLOUD_ACCESS_KEY_ID or OSS_ACCESS_KEY_ID",
    )
    access_key_secret = require_env(
        env_first("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "ALIBABACLOUD_ACCESS_KEY_SECRET", "OSS_ACCESS_KEY_SECRET"),
        "ALIBABA_CLOUD_ACCESS_KEY_SECRET or OSS_ACCESS_KEY_SECRET",
    )
    return Client(
        config=Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id=region,
            endpoint=os.environ.get("PAI_DLC_ENDPOINT", f"pai-dlc.{region}.aliyuncs.com"),
        )
    )
