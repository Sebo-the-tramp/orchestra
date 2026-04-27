import json
import sys
import time
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


def arg_value(args: list[str], name: str, default: str | None = None) -> str | None:
    if name not in args:
        return default
    return args[args.index(name) + 1]


def required(args: list[str], name: str) -> str:
    value = arg_value(args, name)
    assert value is not None, name
    return value


def connect_to_router(model_name: str, router_connect: str, worker_id: str):
    socket = Context.instance().socket(DEALER)
    socket.setsockopt(IDENTITY, f"{model_name}-{worker_id}".encode("utf-8"))
    socket.connect(router_connect)
    return socket


def audio_path(payload: dict[str, Any]) -> str:
    path = str(payload.get("audio_path", payload.get("path", "")))
    assert path, "audio_path"
    return path


def main() -> None:
    import torch
    from transformers import pipeline

    args = sys.argv[1:]
    model_id = required(args, "--model-id")
    router_connect = required(args, "--router-connect")
    worker_id = required(args, "--worker-id")
    chunk_length_s = int(arg_value(args, "--chunk-length-s", "30"))
    batch_size = int(arg_value(args, "--batch-size", "8"))
    model = load_registry().models[model_id]
    path = model_file_path(load_config(), model)
    assert path.is_dir(), path
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=str(path),
        torch_dtype=dtype,
        device=0 if cuda else -1,
    )
    socket = connect_to_router(model_id, router_connect, worker_id)
    last_work_time = time.monotonic()

    while True:
        socket.send_json({"type": "HEARTBEAT", "model_name": model_id})
        if not socket.poll(timeout=POLL_TIMEOUT_MS):
            if time.monotonic() - last_work_time >= IDLE_SHUTDOWN_SECONDS:
                return
            continue
        last_work_time = time.monotonic()
        frames = socket.recv_multipart()
        payload = json.loads(frames[1].decode("utf-8"))
        if payload.get("type") == "SHUTDOWN":
            return
        result = transcriber(
            audio_path(payload),
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=payload.get("return_timestamps", False),
        )
        socket.send_json(
            {
                "type": "SUCCESS",
                "request_id": payload["request_id"],
                "text": result["text"],
                "segments": result.get("chunks", []),
                "model_name": model_id,
            }
        )


if __name__ == "__main__":
    main()
