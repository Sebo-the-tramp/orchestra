import json
import os
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
IDLE_SHUTDOWN_SECONDS = 60
STARTUP_SECONDS = 5
STARTUP_TIMEOUT_SECONDS = 180
DEFAULT_LLAMA_DIR = "/home/kilolab/Documents/shared/llama.cpp"
WORKER_VALUE_FLAGS = {
    "--model-id",
    "--router-connect",
    "--worker-id",
    "--llama-dir",
    "--startup-seconds",
    "--startup-timeout-seconds",
    "--health-path",
}


def arg_value(args: list[str], name: str, default: str | None = None) -> str | None:
    if name not in args:
        return default
    return args[args.index(name) + 1]


def required(args: list[str], name: str) -> str:
    value = arg_value(args, name)
    assert value is not None, name
    return value


def server_args(args: list[str]) -> list[str]:
    out = []
    skip = False
    for item in args:
        if skip:
            skip = False
            continue
        if item in WORKER_VALUE_FLAGS:
            skip = True
            continue
        out.append(item)
    return out


def flag_value(args: list[str], names: set[str], default: str) -> str:
    for name in names:
        value = arg_value(args, name)
        if value is not None:
            return value
    return default


def connect_to_router(model_name: str, router_connect: str, worker_id: str):
    socket = Context.instance().socket(DEALER)
    socket.setsockopt(IDENTITY, f"{model_name}-{worker_id}".encode("utf-8"))
    socket.connect(router_connect)
    return socket


def llama_server_bin(llama_dir: str) -> Path:
    root = Path(llama_dir).expanduser()
    direct = root / "llama-server"
    built = root / "build" / "bin" / "llama-server"
    if direct.is_file():
        return direct
    assert built.is_file(), f"llama-server not found in {root}"
    return built


def prompt_from_payload(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt", "")
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
        process.wait(timeout=10)


def completion(port: str, alias: str, payload: dict[str, Any]) -> str:
    config = payload.get("config", {})
    data = {
        "model": alias,
        "messages": messages_from_payload(payload),
        "temperature": float(config.get("temperature", 0.0)),
        "top_p": float(config.get("top_p", 1.0)),
        "max_tokens": int(config.get("max_tokens", 512)),
    }
    response = post_json(f"http://127.0.0.1:{port}/v1/chat/completions", data)
    return response["choices"][0]["message"]["content"]


def main() -> None:
    args = sys.argv[1:]
    model_id = required(args, "--model-id")
    router_connect = required(args, "--router-connect")
    worker_id = required(args, "--worker-id")
    llama_dir = arg_value(args, "--llama-dir", DEFAULT_LLAMA_DIR)
    startup_seconds = int(arg_value(args, "--startup-seconds", str(STARTUP_SECONDS)))
    startup_timeout = int(
        arg_value(args, "--startup-timeout-seconds", str(STARTUP_TIMEOUT_SECONDS))
    )
    health_path = arg_value(args, "--health-path", "/health")
    model = load_registry().models[model_id]
    model_path = model_file_path(load_config(), model)
    assert model_path.is_file(), model_path

    extra = server_args(args)
    port = flag_value(extra, {"--port"}, "8001")
    alias = flag_value(extra, {"-a", "--alias"}, model_id)
    if "--host" not in extra:
        extra += ["--host", "0.0.0.0"]
    if "--port" not in extra:
        extra += ["--port", port]

    command = [str(llama_server_bin(str(llama_dir))), "-m", str(model_path), *extra]
    process = subprocess.Popen(
        command,
        cwd=str(Path(str(llama_dir)).expanduser()),
        start_new_session=True,
    )
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
        socket.send_json(
            {
                "type": "SUCCESS",
                "request_id": payload["request_id"],
                "answer": completion(port, alias, payload),
                "model_name": model_id,
            }
        )


if __name__ == "__main__":
    main()
