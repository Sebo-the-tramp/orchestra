import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

RUNNING_STATES = {"PD", "PENDING", "QUEUED", "H", "HELD", "R", "RUNNING"}
DONE_STATES = {"CD", "COMPLETED", "FINISHED", "F", "FAILED", "CA", "CANCELLED", "TO", "TIMEOUT"}
GFLOW_POLL_SECONDS = 2.0
GFLOW_COMMANDS = ["gflowd", "gbatch", "gqueue", "gjob", "gcancel"]
GFLOW_PROJECT = "orchestra"
GFLOW_INSTALL_COMMANDS = [
    "uv tool install runqd",
    "pipx install runqd",
    "cargo install gflow",
]


class ManagedProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


@dataclass
class GflowJob:
    job_id: str
    script_path: Path
    last_poll_at: float = 0.0
    last_status: int | None = None

    def poll(self) -> int | None:
        now = time.monotonic()
        if now - self.last_poll_at < GFLOW_POLL_SECONDS:
            return self.last_status
        self.last_poll_at = now
        result = subprocess.run(
            ["gjob", "show", self.job_id],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            self.last_status = 1
            return self.last_status
        output = result.stdout
        states = [
            line.split(":", 1)[1].strip().upper()
            for line in output.splitlines()
            if line.lower().startswith("state:")
        ]
        assert states, output
        state = states[0]
        if state in RUNNING_STATES:
            self.last_status = None
        elif state in DONE_STATES:
            self.last_status = 0 if state in {"CD", "COMPLETED", "FINISHED"} else 1
        else:
            self.last_status = None
        return self.last_status

    def terminate(self) -> None:
        subprocess.run(["gcancel", self.job_id], check=False)


def gflow_available() -> bool:
    return not missing_gflow_commands()


def missing_gflow_commands() -> list[str]:
    return [command for command in GFLOW_COMMANDS if shutil.which(command) is None]


def assert_gflow_available() -> None:
    missing = missing_gflow_commands()
    assert not missing, (
        "gflow process manager requested but commands are missing: "
        f"{', '.join(missing)}. Install with 'uv tool install runqd' "
        "or use ORCHESTRA_PROCESS_MANAGER=auto/local."
    )


def process_manager_kind(kind: str) -> str:
    if kind == "auto":
        return "gflow" if gflow_available() else "local"
    assert kind in {"local", "gflow"}, kind
    if kind == "gflow":
        assert_gflow_available()
    return kind


def worker_gpus(accelerator: str) -> int:
    return 0 if accelerator == "cpu" else 1


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def write_job_script(command: list[str], logs: Path, name: str) -> Path:
    directory = logs / "gflow"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{safe_name(name)}-{time.time_ns()}.sh"
    content = "#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n"
    path.write_text(content)
    path.chmod(0o755)
    return path


def submit_gflow(command: list[str], name: str, gpus: int, logs: Path) -> GflowJob:
    assert_gflow_available()
    subprocess.run(["gflowd", "up"], check=True)
    script = write_job_script(command, logs, name)
    output = subprocess.check_output(
        ["gbatch", "--gpus", str(gpus), "--project", GFLOW_PROJECT, "--name", name, str(script)],
        text=True,
    )
    matches = re.findall(r"\d+", output)
    assert matches, output
    return GflowJob(job_id=matches[-1], script_path=script)


def start_process(
    command: list[str],
    name: str,
    accelerator: str,
    logs: Path,
    manager: str,
) -> ManagedProcess:
    kind = process_manager_kind(manager)
    if kind == "gflow":
        return submit_gflow(command, name, worker_gpus(accelerator), logs)
    return subprocess.Popen(command)
