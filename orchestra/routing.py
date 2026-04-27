from dataclasses import dataclass

from orchestra.config import OrchestraConfig
from orchestra.hardware import HardwareReport
from orchestra.model_store import model_status
from orchestra.registry import ModelSpec, Registry, runtime_for_model
from orchestra.runtime import compatible, env_status


@dataclass(frozen=True)
class RouteCandidate:
    model: ModelSpec
    ready: bool
    reason: str


def candidate_status(
    config: OrchestraConfig,
    registry: Registry,
    hardware: HardwareReport,
    model: ModelSpec,
) -> RouteCandidate:
    runtime = runtime_for_model(registry, model)
    if model_status(config, model) != "downloaded":
        return RouteCandidate(model, False, "model missing")
    if env_status(runtime) != "ready":
        return RouteCandidate(model, False, "env missing")
    if not compatible(runtime, hardware):
        return RouteCandidate(model, False, "incompatible runtime")
    if hardware.gpus and max(gpu.free_mb for gpu in hardware.gpus) < model.min_vram_mb:
        return RouteCandidate(model, False, "insufficient VRAM")
    return RouteCandidate(model, True, "ready")


def candidates_for_task(
    config: OrchestraConfig,
    registry: Registry,
    hardware: HardwareReport,
    task: str,
    engine: str | None = None,
    runtime: str | None = None,
) -> list[RouteCandidate]:
    models = [model for model in registry.models.values() if model.task == task]
    if engine is not None:
        models = [model for model in models if model.default_engine == engine]
    if runtime is not None:
        models = [model for model in models if model.default_runtime == runtime]
    candidates = [candidate_status(config, registry, hardware, model) for model in models]
    return sorted(
        candidates,
        key=lambda item: (not item.ready, item.model.min_vram_mb, item.model.name),
    )


def choose_model(
    config: OrchestraConfig,
    registry: Registry,
    hardware: HardwareReport,
    task: str,
    engine: str | None = None,
    runtime: str | None = None,
) -> RouteCandidate:
    candidates = candidates_for_task(config, registry, hardware, task, engine, runtime)
    assert candidates, f"No models found for task={task} engine={engine} runtime={runtime}"
    ready = [candidate for candidate in candidates if candidate.ready]
    return ready[0] if ready else candidates[0]
