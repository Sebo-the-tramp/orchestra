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
        "device_map": arg_value(args, "device-map", "auto"),
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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        args["model_id"],
        device_map=args["device_map"],
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).eval()
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
        inputs = tokenizer(prompt_from_payload(payload), return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=int(config.get("max_tokens", 512)))
        answer = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
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
