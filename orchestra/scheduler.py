from dataclasses import dataclass

from orchestra.config import OrchestraConfig
from orchestra.hardware import HardwareReport
from orchestra.model_store import model_status
from orchestra.registry import ModelSpec, Registry, runtime_for_model
from orchestra.runtime import compatible, env_status


@dataclass(frozen=True)
class ScheduleDecision:
    model_name: str
    action: str
    reason: str
    min_vram_mb: int


def model_ready(
    config: OrchestraConfig,
    registry: Registry,
    hardware: HardwareReport,
    model: ModelSpec,
) -> tuple[bool, str]:
    runtime = runtime_for_model(registry, model)
    if model_status(config, model) != "downloaded":
        return False, "download model"
    if env_status(runtime) != "ready":
        return False, "setup env"
    if not compatible(runtime, hardware):
        return False, "incompatible runtime"
    free_vram = max((gpu.free_mb for gpu in hardware.gpus), default=0)
    if hardware.gpus and free_vram < model.min_vram_mb:
        return True, "ready: VRAM below advisory estimate"
    return True, "ready"


def schedule_plan(
    config: OrchestraConfig,
    registry: Registry,
    hardware: HardwareReport,
    warm_tasks: set[str] | None = None,
) -> list[ScheduleDecision]:
    warm_tasks = warm_tasks or set()
    decisions = []
    for model in sorted(registry.models.values(), key=lambda item: (item.task, item.min_vram_mb)):
        ready, reason = model_ready(config, registry, hardware, model)
        if ready and model.task in warm_tasks:
            action = "keep_warm"
        elif ready:
            action = "available"
        else:
            action = "blocked"
        decisions.append(ScheduleDecision(model.name, action, reason, model.min_vram_mb))
    return decisions
