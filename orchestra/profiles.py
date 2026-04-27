from dataclasses import dataclass

from orchestra.config import OrchestraConfig
from orchestra.hardware import HardwareReport
from orchestra.model_store import model_status
from orchestra.registry import Registry, RuntimeSpec
from orchestra.runtime import compatible, env_status, setup_env_commands


@dataclass(frozen=True)
class ArchitectureProfile:
    name: str
    accelerator: str
    preferred_text_engine: str
    preferred_gguf_runtime: str
    preferred_hf_runtime: str


@dataclass(frozen=True)
class SetupAction:
    target: str
    status: str
    action: str
    command: str


def architecture_profile(hardware: HardwareReport) -> ArchitectureProfile:
    if hardware.cuda:
        return ArchitectureProfile("nvidia-cuda", "cuda", "vllm", "llama_cpp_cuda", "vllm_cuda")
    if hardware.rocm:
        return ArchitectureProfile(
            "amd-hip", "hip", "llama_cpp", "llama_cpp_hip", "transformers_cpu"
        )
    if hardware.mlx:
        return ArchitectureProfile(
            "apple-mlx", "mlx", "llama_cpp", "llama_cpp_metal", "transformers_cpu"
        )
    return ArchitectureProfile("cpu", "cpu", "transformers", "llama_cpp_cpu", "transformers_cpu")


def runtime_install_action(runtime: RuntimeSpec, hardware: HardwareReport) -> SetupAction:
    status = env_status(runtime)
    if not compatible(runtime, hardware):
        return SetupAction(runtime.name, status, "blocked", "incompatible hardware")
    if status == "ready":
        return SetupAction(runtime.name, status, "none", "already ready")
    commands = [" ".join(command) for command in setup_env_commands(runtime)]
    return SetupAction(runtime.name, status, "setup", " && ".join(commands))


def setup_plan(
    config: OrchestraConfig,
    registry: Registry,
    hardware: HardwareReport,
) -> list[SetupAction]:
    actions = [runtime_install_action(runtime, hardware) for runtime in registry.runtimes.values()]
    for model in registry.models.values():
        status = model_status(config, model)
        action = "none" if status == "downloaded" else "download"
        actions.append(SetupAction(model.name, status, action, model.artifact))
    return actions


def default_llm_engine_runtime(hardware: HardwareReport, model_format: str) -> tuple[str, str]:
    profile = architecture_profile(hardware)
    if model_format == "gguf":
        return "llama_cpp", profile.preferred_gguf_runtime
    return profile.preferred_text_engine, profile.preferred_hf_runtime
