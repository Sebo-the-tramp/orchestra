import json
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Gpu:
    vendor: str
    name: str
    total_mb: int
    free_mb: int


@dataclass(frozen=True)
class HardwareReport:
    os: str
    machine: str
    python: str
    uv: bool
    cuda: bool
    rocm: bool
    mlx: bool
    gpus: list[Gpu]


def _nvidia_gpus() -> list[Gpu]:
    if shutil.which("nvidia-smi") is None:
        return []
    query = "name,memory.total,memory.free"
    command = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True).strip()
    if not output:
        return []
    gpus = []
    for line in output.splitlines():
        name, total, free = [part.strip() for part in line.split(",")]
        gpus.append(Gpu("nvidia", name, int(total), int(free)))
    return gpus


def detect_hardware() -> HardwareReport:
    machine = platform.machine()
    system = platform.system().lower()
    gpus = _nvidia_gpus()
    return HardwareReport(
        os=system,
        machine=machine,
        python=platform.python_version(),
        uv=shutil.which("uv") is not None,
        cuda=bool(gpus) or shutil.which("nvcc") is not None,
        rocm=shutil.which("rocm-smi") is not None or shutil.which("hipcc") is not None,
        mlx=system == "darwin" and machine == "arm64",
        gpus=gpus,
    )


def as_json(report: HardwareReport) -> str:
    return json.dumps(report, default=lambda item: item.__dict__, indent=2)
