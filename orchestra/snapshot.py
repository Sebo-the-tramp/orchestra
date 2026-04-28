import json
from pathlib import Path

from orchestra.config import OrchestraConfig
from orchestra.hardware import HardwareReport
from orchestra.model_store import model_status
from orchestra.registry import Registry, runtime_for_model
from orchestra.runtime import compatible, env_status


def snapshot_data(
    config: OrchestraConfig,
    registry: Registry,
    hardware: HardwareReport,
) -> dict:
    models = []
    for model in registry.models.values():
        runtime = runtime_for_model(registry, model)
        models.append(
            {
                "name": model.name,
                "aliases": model.aliases,
                "task": model.task,
                "format": model.format,
                "engine": model.default_engine,
                "runtime": model.default_runtime,
                "artifact": model.artifact,
                "artifact_file": model.artifact_file,
                "local_path": model.local_path,
                "default_args": model.default_args,
                "model_status": model_status(config, model),
                "env_status": env_status(runtime),
                "compatible": compatible(runtime, hardware),
                "min_vram_mb": model.min_vram_mb,
            }
        )
    return {
        "hardware": hardware,
        "paths": {
            "root": str(config.root),
            "model_cache": str(config.model_cache),
            "engine_cache": str(config.engine_cache),
            "env_cache": str(config.env_cache),
            "logs": str(config.logs),
        },
        "broker_address": config.broker_address,
        "broker_bind_address": config.broker_bind_address,
        "process_manager": config.process_manager,
        "models": models,
        "runtimes": [runtime.__dict__ for runtime in registry.runtimes.values()],
        "engines": [engine.__dict__ for engine in registry.engines.values()],
    }


def write_snapshot(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        answer = input(f"{path} exists. Override it? [y/N]: ").strip().lower()
        assert answer in {"y", "yes"}, f"Refusing to override {path}"
    path.write_text(json.dumps(data, default=lambda item: item.__dict__, indent=2))
