import json
import sys
import time
from pathlib import Path
from typing import Any

from zmq import DEALER, IDENTITY, Context

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))



def connect_to_router(model_name: str, router_connect: str, worker_id: str):
    socket = Context.instance().socket(DEALER)
    socket.setsockopt(IDENTITY, f"{model_name}-{worker_id}".encode("utf-8"))
    socket.connect(router_connect)
    return socket


POLL_TIMEOUT_MS = 1000
IDLE_SHUTDOWN_SECONDS = 60
SHUTDOWN_MESSAGE_TYPE = "SHUTDOWN"


def arg_value(args: list[str], name: str) -> str:
    flag = f"--{name}"
    assert flag in args, flag
    return args[args.index(flag) + 1]


def parse_args() -> dict[str, str]:
    args = sys.argv[1:]
    return {
        "model_id": arg_value(args, "model-id"),
        "router_connect": arg_value(args, "router-connect"),
        "worker_id": arg_value(args, "worker-id"),
    }


def answer(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(str(item) for item in prompt)
    return f"echo: {prompt}"


def main() -> None:
    args = parse_args()
    socket = connect_to_router(args["model_id"], args["router_connect"], args["worker_id"])
    last_work_time = time.monotonic()

    while True:
        socket.send_json({"type": "HEARTBEAT", "model_name": args["model_id"]})
        if not socket.poll(timeout=POLL_TIMEOUT_MS):
            if time.monotonic() - last_work_time >= IDLE_SHUTDOWN_SECONDS:
                return
            continue
        last_work_time = time.monotonic()
        frames = socket.recv_multipart()
        payload = json.loads(frames[1].decode("utf-8"))
        if payload.get("type") == SHUTDOWN_MESSAGE_TYPE:
            return
        socket.send_json(
            {
                "type": "SUCCESS",
                "request_id": payload["request_id"],
                "answer": answer(payload),
                "model_name": args["model_id"],
            }
        )


if __name__ == "__main__":
    main()
