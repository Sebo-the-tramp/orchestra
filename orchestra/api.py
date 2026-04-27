import json
import uuid
from typing import Any

import zmq
from fastapi import FastAPI
from pydantic import BaseModel, Field

from orchestra.config import load_config
from orchestra.hardware import detect_hardware
from orchestra.job_store import tail_jobs
from orchestra.metrics import tail_events
from orchestra.nodes import load_node_specs
from orchestra.profiles import architecture_profile, setup_plan
from orchestra.registry import load_registry
from orchestra.scheduler import schedule_plan

DEFAULT_TIMEOUT_MS = 60000

app = FastAPI(title="ORCHESTRA API")


class GenerateRequest(BaseModel):
    model_name: str
    prompt: str | list[str]
    engine: str | None = None
    runtime: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    timeout_ms: int = DEFAULT_TIMEOUT_MS


def broker_request(payload: dict[str, Any], timeout_ms: int) -> dict[str, Any]:
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect(load_config().broker_address)
    socket.send_multipart([json.dumps(payload).encode("utf-8")])
    assert socket.poll(timeout_ms), f"Broker response timed out after {timeout_ms} ms"
    return json.loads(socket.recv_multipart()[-1].decode("utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/events")
def events(limit: int = 20) -> list[dict[str, Any]]:
    return tail_events(limit)


@app.get("/jobs")
def jobs(limit: int = 20) -> list[dict[str, Any]]:
    return tail_jobs(limit)


@app.get("/schedule")
def schedule(warm_task: list[str] | None = None) -> list[dict[str, Any]]:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    return [
        {
            "model_name": decision.model_name,
            "action": decision.action,
            "reason": decision.reason,
            "min_vram_mb": decision.min_vram_mb,
        }
        for decision in schedule_plan(config, registry, hardware, set(warm_task or []))
    ]


@app.get("/profile")
def profile() -> dict[str, str]:
    current = architecture_profile(detect_hardware())
    return {
        "name": current.name,
        "accelerator": current.accelerator,
        "preferred_text_engine": current.preferred_text_engine,
        "preferred_gguf_runtime": current.preferred_gguf_runtime,
        "preferred_hf_runtime": current.preferred_hf_runtime,
    }


@app.get("/setup-plan")
def api_setup_plan() -> list[dict[str, str]]:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    return [
        {
            "target": action.target,
            "status": action.status,
            "action": action.action,
            "command": action.command,
        }
        for action in setup_plan(config, registry, hardware)
    ]


@app.get("/nodes")
def nodes() -> list[dict[str, Any]]:
    return [
        {
            "name": node.name,
            "role": node.role,
            "address": node.address,
            "labels": node.labels,
        }
        for node in load_node_specs()
    ]


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    payload = {
        "request_id": str(uuid.uuid4()),
        "model_name": request.model_name,
        "prompt": request.prompt,
        "config": request.config,
    }
    if request.engine is not None:
        payload["engine"] = request.engine
    if request.runtime is not None:
        payload["runtime"] = request.runtime
    return broker_request(payload, request.timeout_ms)
