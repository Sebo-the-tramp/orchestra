import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from zmq import DEALER, IDENTITY, Context

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from orchestra.config import load_config  # noqa: E402
from orchestra.model_store import model_file_path  # noqa: E402
from orchestra.registry import load_registry  # noqa: E402

POLL_TIMEOUT_MS = 1000
IDLE_SHUTDOWN_SECONDS = 3
STARTUP_SECONDS = 20
STARTUP_TIMEOUT_SECONDS = 300
WORKER_VALUE_FLAGS = {
    "--model-id",
    "--router-connect",
    "--worker-id",
    "--platform",
    "--gpu",
    "--startup-seconds",
    "--startup-timeout-seconds",
    "--health-path",
}
WORKER_BOOL_FLAGS = {"--isolate-gpu-devices"}
ISOLATION_ENV_KEYS = {"VLLM_PLATFORM", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}


def arg_value(args: list[str], name: str, default: str | None = None) -> str | None:
    if name not in args:
        return default
    return args[args.index(name) + 1]


def required(args: list[str], name: str) -> str:
    value = arg_value(args, name)
    assert value is not None, name
    return value


def serve_args(args: list[str]) -> list[str]:
    out = []
    skip = False
    for item in args:
        if skip:
            skip = False
            continue
        if item in WORKER_VALUE_FLAGS:
            skip = True
            continue
        if item in WORKER_BOOL_FLAGS:
            continue
        out.append(item)
    return out


def connect_to_router(model_name: str, router_connect: str, worker_id: str):
    socket = Context.instance().socket(DEALER)
    socket.setsockopt(IDENTITY, f"{model_name}-{worker_id}".encode("utf-8"))
    socket.connect(router_connect)
    return socket


def prompt_from_payload(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt", payload.get("input", ""))
    if isinstance(prompt, list):
        return "\n".join(str(item) for item in prompt)
    return str(prompt)


def messages_from_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in payload:
        return payload["messages"]
    return [{"role": "user", "content": prompt_from_payload(payload)}]


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    response = urllib.request.urlopen(request, timeout=600)
    return json.loads(response.read().decode("utf-8"))


def wait_for_http(port: str, path: str, timeout_seconds: int, process: subprocess.Popen) -> None:
    url = f"http://127.0.0.1:{port}{path}"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        assert process.poll() is None, f"server exited before becoming healthy: {url}"
        result = subprocess.run(
            ["curl", "-fsS", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise AssertionError(f"server health timeout after {timeout_seconds}s: {url}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(2)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def isolated_command(command: list[str], env: dict[str, str]) -> list[str]:
    exports = "\n".join(
        f"export {key}={shlex.quote(env[key])}" for key in sorted(ISOLATION_ENV_KEYS) if key in env
    )
    joined = " ".join(shlex.quote(item) for item in command)
    script = "\n".join(
        [
            "set -euo pipefail",
            "mkdir -p /tmp/no-nvidia",
            "shopt -s nullglob",
            "for dev in /dev/nvidia*; do",
            "  base=$(basename \"$dev\")",
            "  : > \"/tmp/no-nvidia/$base\"",
            "  mount --bind \"/tmp/no-nvidia/$base\" \"$dev\" 2>/dev/null || true",
            "done",
            "if [ -d /proc/driver/nvidia ]; then",
            "  mkdir -p /tmp/no-nvidia/proc_nvidia",
            "  mount --bind /tmp/no-nvidia/proc_nvidia /proc/driver/nvidia 2>/dev/null || true",
            "fi",
            exports,
            f"exec {joined}",
        ]
    )
    return ["sudo", "unshare", "-m", "--fork", "bash", "-lc", script]


def chat(port: str, model_name: str, payload: dict[str, Any]) -> str:
    config = payload.get("config", {})
    data = {
        "model": model_name,
        "messages": messages_from_payload(payload),
        "temperature": float(config.get("temperature", 0.0)),
        "top_p": float(config.get("top_p", 1.0)),
        "max_tokens": int(config.get("max_tokens", 512)),
    }
    response = post_json(f"http://127.0.0.1:{port}/v1/chat/completions", data)
    return response["choices"][0]["message"]["content"]


def embedding(port: str, model_name: str, payload: dict[str, Any]) -> list[float]:
    data = {"model": model_name, "input": payload.get("input", prompt_from_payload(payload))}
    response = post_json(f"http://127.0.0.1:{port}/v1/embeddings", data)
    return response["data"][0]["embedding"]


def main() -> None:
    args = sys.argv[1:]
    model_id = required(args, "--model-id")
    router_connect = required(args, "--router-connect")
    worker_id = required(args, "--worker-id")
    startup_seconds = int(arg_value(args, "--startup-seconds", str(STARTUP_SECONDS)))
    startup_timeout = int(
        arg_value(args, "--startup-timeout-seconds", str(STARTUP_TIMEOUT_SECONDS))
    )
    health_path = arg_value(args, "--health-path", "/health")
    isolate_gpu_devices = "--isolate-gpu-devices" in args
    platform = arg_value(args, "--platform")
    gpu = arg_value(args, "--gpu")
    model = load_registry().models[model_id]
    model_path = model_file_path(load_config(), model)
    assert model_path.exists(), model_path

    extra = serve_args(args)
    port = arg_value(extra, "--port", "8001")
    env = os.environ.copy()
    if platform == "rocm":
        env["VLLM_PLATFORM"] = "rocm"
        env["HIP_VISIBLE_DEVICES"] = gpu or "0"
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    command = ["vllm", "serve", str(model_path), *extra]
    if isolate_gpu_devices:
        command = isolated_command(command, env)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    time.sleep(startup_seconds)
    wait_for_http(port, str(health_path), startup_timeout, process)

    socket = connect_to_router(model_id, router_connect, worker_id)
    last_work_time = time.monotonic()
    while True:
        socket.send_json({"type": "HEARTBEAT", "model_name": model_id})
        if not socket.poll(timeout=POLL_TIMEOUT_MS):
            if time.monotonic() - last_work_time >= IDLE_SHUTDOWN_SECONDS:
                stop_process(process)
                return
            continue
        last_work_time = time.monotonic()
        frames = socket.recv_multipart()
        payload = json.loads(frames[1].decode("utf-8"))
        if payload.get("type") == "SHUTDOWN":
            stop_process(process)
            return
        result_key = "embedding" if model.task == "embedding" else "answer"
        result = (
            embedding(port, model_id, payload)
            if model.task == "embedding"
            else chat(port, model_id, payload)
        )
        socket.send_json(
            {
                "type": "SUCCESS",
                "request_id": payload["request_id"],
                result_key: result,
                "model_name": model_id,
            }
        )


if __name__ == "__main__":
    main()
