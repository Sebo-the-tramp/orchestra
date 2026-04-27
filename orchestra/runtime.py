import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from orchestra.hardware import HardwareReport
from orchestra.registry import EngineSpec, ModelSpec, RuntimeSpec

TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}


def compatible(runtime: RuntimeSpec, hardware: HardwareReport) -> bool:
    if runtime.accelerator == "cpu":
        return True
    if runtime.accelerator == "cuda":
        return hardware.cuda
    if runtime.accelerator in {"hip", "rocm"}:
        return hardware.rocm
    if runtime.accelerator in {"mlx", "mps"}:
        return hardware.mlx
    return False


def env_path(runtime: RuntimeSpec) -> Path:
    if runtime.python == "system":
        return Path(sys.executable)
    return runtime.worker_dir / ".venv"


def env_status(runtime: RuntimeSpec) -> str:
    if runtime.python == "system":
        return "ready"
    path = env_path(runtime)
    if not path.exists():
        return "missing"
    if not (path / "bin" / "python").is_file():
        return "broken"
    return "ready"


def package_groups(runtime: RuntimeSpec) -> tuple[list[str], list[str]]:
    all_packages = [
        name if version == "*" else f"{name}=={version}"
        for name, version in runtime.packages.items()
    ]
    torch_packages = [
        package
        for name, package in zip(runtime.packages, all_packages, strict=True)
        if name in TORCH_PACKAGES
    ]
    other_packages = [
        package
        for name, package in zip(runtime.packages, all_packages, strict=True)
        if name not in TORCH_PACKAGES
    ]
    return torch_packages, other_packages


def setup_env_commands(runtime: RuntimeSpec) -> list[list[str]]:
    if runtime.python == "system":
        return []
    torch_packages, other_packages = package_groups(runtime)
    commands = [["uv", "venv", "--python", runtime.python]]
    if torch_packages:
        command = ["uv", "pip", "install", *torch_packages]
        if runtime.index_url:
            command += ["--index-url", runtime.index_url]
        commands.append([f"{key}={value}" for key, value in runtime.env.items()] + command)
    if other_packages:
        command = ["uv", "pip", "install", *other_packages]
        commands.append([f"{key}={value}" for key, value in runtime.env.items()] + command)
    for command in runtime.setup:
        commands.append([f"{key}={value}" for key, value in runtime.env.items()] + command.split())
    return commands


def setup_env(runtime: RuntimeSpec, force: bool = False) -> None:
    assert runtime.python != "system", f"Runtime uses current interpreter: {runtime.name}"
    path = env_path(runtime)
    if path.exists() and not force:
        assert env_status(runtime) == "missing", f"Env already exists: {path}"
    if path.exists() and force:
        answer = input(f"{path} exists. Override it? [y/N]: ").strip().lower()
        assert answer in {"y", "yes"}, f"Refusing to override {path}"
        shutil.rmtree(path)

    subprocess.run(
        ["uv", "venv", "--python", runtime.python],
        cwd=runtime.worker_dir,
        check=True,
    )
    torch_packages, other_packages = package_groups(runtime)
    if torch_packages:
        command = ["uv", "pip", "install", *torch_packages]
        if runtime.index_url:
            command += ["--index-url", runtime.index_url]
        subprocess.run(
            command,
            cwd=runtime.worker_dir,
            env=os.environ | runtime.env,
            check=True,
        )
    if other_packages:
        subprocess.run(
            ["uv", "pip", "install", *other_packages],
            cwd=runtime.worker_dir,
            env=os.environ | runtime.env,
            check=True,
        )
    for command in runtime.setup:
        subprocess.run(
            command.split(),
            cwd=runtime.worker_dir,
            env=os.environ | runtime.env,
            check=True,
        )


def build_worker_command(
    model: ModelSpec,
    runtime: RuntimeSpec,
    engine: EngineSpec,
    router_address: str,
    worker_id: str,
    args_per_model: dict[str, Any] | None = None,
) -> list[str]:
    python = env_path(runtime) / "bin" / "python"
    if runtime.python == "system":
        python = Path(sys.executable)
    worker = model.worker_dir / engine.entrypoint
    assert python.is_file(), python
    assert worker.is_file(), worker
    args = args_per_model or {}
    extra = []
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True or value is None:
            extra.append(flag)
        elif value is False:
            continue
        else:
            extra += [flag, str(value)]
    return [
        str(python),
        str(worker),
        "--model-id",
        model.name,
        "--router-connect",
        router_address,
        "--worker-id",
        worker_id,
        *model.default_args,
        *extra,
    ]
