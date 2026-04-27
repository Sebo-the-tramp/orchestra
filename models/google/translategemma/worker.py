import json
import re
import sys
import threading
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
MODEL_LOCK = threading.Lock()


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


def normalize_lang_code(code: str) -> str:
    code = code.strip().replace("_", "-")
    assert re.fullmatch(r"[A-Za-z]{2}(?:-[A-Za-z]{2})?", code), code
    return code.lower()


def cleanup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\s*(translation|translated text)\s*[:\-]\s*", "", text, flags=re.I)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prompt_text(payload: dict[str, Any]) -> str:
    text = payload.get("text", payload.get("prompt", ""))
    if isinstance(text, list):
        return "\n".join(str(item) for item in text)
    return str(text)


def build_messages(text: str, source: str, target: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source,
                    "target_lang_code": target,
                    "text": text,
                }
            ],
        }
    ]


def load_model(model_id: str, use_4bit: bool, strict_gpu_only: bool):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    model = load_registry().models[model_id]
    path = model_file_path(load_config(), model)
    assert path.is_dir(), path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    processor = AutoProcessor.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
    )
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda" and use_4bit:
        kwargs["device_map"] = {"": 0} if strict_gpu_only else "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        load_mode = "4bit-nf4"
    elif device.type == "cuda":
        kwargs["device_map"] = {"": 0} if strict_gpu_only else "auto"
        kwargs["torch_dtype"] = dtype
        load_mode = "bf16" if dtype == torch.bfloat16 else "fp16"
    else:
        kwargs["device_map"] = "cpu"
        kwargs["torch_dtype"] = torch.float32
        load_mode = "fp32-cpu"
    loaded = AutoModelForImageTextToText.from_pretrained(str(path), **kwargs).eval()
    setattr(loaded, "_orchestra_load_mode", load_mode)
    return processor, loaded, device, load_mode


def model_device(model, fallback):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    for parameter in model.parameters():
        return parameter.device
    return fallback


def translate(processor, model, device, payload: dict[str, Any], max_new_tokens: int) -> str:
    import torch

    source = normalize_lang_code(str(payload.get("source", payload.get("source_lang", "it"))))
    target = normalize_lang_code(str(payload.get("target", payload.get("target_lang", "en"))))
    text = prompt_text(payload).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if source == target:
        return text
    messages = build_messages(text, source, target)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_device(model, device))
    input_len = int(inputs["input_ids"].shape[-1])
    with MODEL_LOCK:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
    output_tokens = (
        generated[0]
        if getattr(model.config, "is_encoder_decoder", False)
        else generated[0][input_len:]
    )
    return cleanup(processor.decode(output_tokens, skip_special_tokens=True))


def main() -> None:
    args = sys.argv[1:]
    model_id = required(args, "--model-id")
    router_connect = required(args, "--router-connect")
    worker_id = required(args, "--worker-id")
    max_new_tokens = int(arg_value(args, "--max-new-tokens", "1800"))
    use_4bit = arg_value(args, "--use-4bit", "1") == "1"
    strict_gpu_only = arg_value(args, "--strict-gpu-only", "1") == "1"
    processor, model, device, load_mode = load_model(model_id, use_4bit, strict_gpu_only)
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
        socket.send_json(
            {
                "type": "SUCCESS",
                "request_id": payload["request_id"],
                "translation": translate(processor, model, device, payload, max_new_tokens),
                "model_name": model_id,
                "load_mode": load_mode,
            }
        )


if __name__ == "__main__":
    main()
