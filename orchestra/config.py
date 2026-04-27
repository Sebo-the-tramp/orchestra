import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "orchestra.yaml"
DEFAULT_HOME = Path(os.environ.get("ORCHESTRA_HOME", Path.home() / ".orchestra"))
DEFAULT_BROKER_BIND_ADDRESS = "tcp://0.0.0.0:5556"
DEFAULT_BROKER_CONNECT_ADDRESS = "tcp://127.0.0.1:5556"


@dataclass(frozen=True)
class OrchestraConfig:
    root: Path
    model_cache: Path
    engine_cache: Path
    env_cache: Path
    logs: Path
    broker_address: str
    broker_bind_address: str


def _data() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text()) or {}


def load_config() -> OrchestraConfig:
    data = _data()
    paths = data.get("paths", {})
    broker = data.get("broker", {})
    root = Path(paths.get("root", DEFAULT_HOME)).expanduser()
    model_cache_value = os.environ.get(
        "ORCHESTRA_MODEL_CACHE",
        paths.get("model_cache", root / "models"),
    )
    model_cache = Path(model_cache_value).expanduser()
    return OrchestraConfig(
        root=root,
        model_cache=model_cache,
        engine_cache=Path(paths.get("engine_cache", root / "engines")).expanduser(),
        env_cache=Path(paths.get("env_cache", root / "envs")).expanduser(),
        logs=Path(paths.get("logs", root / "logs")).expanduser(),
        broker_address=os.environ.get(
            "ORCHESTRA_BROKER_ADDRESS",
            broker.get("connect_address", broker.get("address", DEFAULT_BROKER_CONNECT_ADDRESS)),
        ),
        broker_bind_address=os.environ.get(
            "ORCHESTRA_BROKER_BIND_ADDRESS",
            broker.get("bind_address", DEFAULT_BROKER_BIND_ADDRESS),
        ),
    )


def init_dirs(config: OrchestraConfig) -> None:
    paths = [config.root, config.model_cache, config.engine_cache, config.env_cache, config.logs]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
