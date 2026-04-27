import json
import sys
import time
from typing import Any

from zmq import DEALER, IDENTITY, Context


def connect_to_router(model_name: str, router_connect: str, worker_id: str):
    socket = Context.instance().socket(DEALER)
    socket.setsockopt(IDENTITY, f"{model_name}-{worker_id}".encode("utf-8"))
    socket.connect(router_connect)
    return socket


POLL_TIMEOUT_MS = 1000
IDLE_SHUTDOWN_SECONDS = 60
SHUTDOWN_MESSAGE_TYPE = "SHUTDOWN"


def arg_value(args: list[str], name: str, default: str | None = None) -> str | None:
    flag = f"--{name}"
    if flag not in args:
        return default
    return args[args.index(flag) + 1]


def parse_args() -> dict[str, str]:
    args = sys.argv[1:]
    parsed = {
        "model_id": arg_value(args, "model-id"),
        "router_connect": arg_value(args, "router-connect"),
        "worker_id": arg_value(args, "worker-id"),
    }
    assert parsed["model_id"]
    assert parsed["router_connect"]
    assert parsed["worker_id"]
    return {key: str(value) for key, value in parsed.items()}


def prompt_from_payload(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt", "")
    if isinstance(prompt, list):
        return "\n".join(str(item) for item in prompt)
    return str(prompt)


def main() -> None:
    from vllm import LLM, SamplingParams

    args = parse_args()
    model = LLM(model=args["model_id"])
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
        config = payload.get("config", {})
        sampling = SamplingParams(
            max_tokens=int(config.get("max_tokens", 512)),
            temperature=float(config.get("temperature", 0.7)),
        )
        outputs = model.generate([prompt_from_payload(payload)], sampling)
        answer = outputs[0].outputs[0].text
        socket.send_json(
            {
                "type": "SUCCESS",
                "request_id": payload["request_id"],
                "answer": answer,
                "model_name": args["model_id"],
            }
        )


if __name__ == "__main__":
    main()
