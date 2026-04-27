from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from orchestra.config import ROOT

MODELS_ROOT = ROOT / "models"


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    worker_dir: Path
    python: str
    accelerator: str
    packages: dict[str, str]
    index_url: str | None
    env: dict[str, str]
    setup: list[str]


@dataclass(frozen=True)
class EngineSpec:
    name: str
    worker_dir: Path
    install: str
    entrypoint: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    aliases: list[str]
    family: str
    worker_dir: Path
    source: str
    artifact: str
    artifact_file: str | None
    local_path: str | None
    task: str
    format: str
    min_vram_mb: int
    recommended_vram_mb: int
    default_engine: str
    supported_engines: list[str]
    default_runtime: str
    worker: str
    default_args: list[str]


@dataclass(frozen=True)
class Registry:
    models: dict[str, ModelSpec]
    runtimes: dict[str, RuntimeSpec]
    engines: dict[str, EngineSpec]


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _model_specs(path: Path) -> list[ModelSpec]:
    data = _read_yaml(path)
    worker_dir = path.parent
    family = data["family"]
    worker = data.get("worker", "worker.py")
    specs = []
    for item in data["models"]:
        specs.append(
            ModelSpec(
                name=item["name"],
                aliases=[str(value) for value in item.get("aliases", [])],
                family=family,
                worker_dir=worker_dir,
                source=item.get("source", "manual"),
                artifact=item.get("artifact", item["name"]),
                artifact_file=item.get("artifact_file"),
                local_path=item.get("local_path"),
                task=item.get("task", "unknown"),
                format=item.get("format", "unknown"),
                min_vram_mb=int(item.get("min_vram_mb", item.get("gpu_memory", 0))),
                recommended_vram_mb=int(
                    item.get("recommended_vram_mb", item.get("min_vram_mb", 0))
                ),
                default_engine=item.get("default_engine", "native"),
                supported_engines=list(item.get("supported_engines", ["native"])),
                default_runtime=item.get("default_runtime", "cpu"),
                worker=item.get("worker", worker),
                default_args=[str(value) for value in item.get("default_args", [])],
            )
        )
    return specs


def _runtime_specs(path: Path) -> tuple[list[RuntimeSpec], list[EngineSpec]]:
    data = _read_yaml(path)
    worker_dir = path.parent
    runtimes = [
        RuntimeSpec(
            name=name,
            worker_dir=worker_dir,
            python=str(item.get("python", "3.12")),
            accelerator=item.get("accelerator", "cpu"),
            packages=dict(item.get("packages", {})),
            index_url=item.get("index_url"),
            env=dict(item.get("env", {})),
            setup=list(item.get("setup", [])),
        )
        for name, item in data.get("runtimes", {}).items()
    ]
    engines = [
        EngineSpec(
            name=name,
            worker_dir=worker_dir,
            install=item.get("install", "uv"),
            entrypoint=item.get("entrypoint", "worker.py"),
        )
        for name, item in data.get("engines", {}).items()
    ]
    return runtimes, engines


def load_registry() -> Registry:
    models: dict[str, ModelSpec] = {}
    runtimes: dict[str, RuntimeSpec] = {}
    engines: dict[str, EngineSpec] = {}
    for path in sorted(MODELS_ROOT.rglob("model.yaml")):
        for spec in _model_specs(path):
            models[spec.name] = spec
    for path in sorted(MODELS_ROOT.rglob("runtime.yaml")):
        runtime_specs, engine_specs = _runtime_specs(path)
        for spec in runtime_specs:
            runtimes[f"{spec.worker_dir}:{spec.name}"] = spec
        for spec in engine_specs:
            engines[f"{path.parent}:{spec.name}"] = spec
    return Registry(models=models, runtimes=runtimes, engines=engines)


def runtime_for_model(
    registry: Registry,
    model: ModelSpec,
    runtime_name: str | None = None,
) -> RuntimeSpec:
    name = runtime_name or model.default_runtime
    key = f"{model.worker_dir}:{name}"
    return registry.runtimes[key]


def model_by_name(registry: Registry, name: str) -> ModelSpec:
    if name in registry.models:
        return registry.models[name]
    matches = [model for model in registry.models.values() if name in model.aliases]
    assert len(matches) == 1, f"Unknown or ambiguous model: {name}"
    return matches[0]


def engine_for_model(
    registry: Registry,
    model: ModelSpec,
    engine_name: str | None = None,
) -> EngineSpec:
    name = engine_name or model.default_engine
    assert name in model.supported_engines, name
    key = f"{model.worker_dir}:{name}"
    return registry.engines[key]
