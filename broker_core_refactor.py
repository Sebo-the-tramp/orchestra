from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import zmq

ROOT = Path(__file__).resolve().parent
MODELS_PATH = ROOT / "models"
ROUTER_ADDRESS = "tcp://10.10.151.14:5556"
POLL_TIMEOUT_MS = 1000
IDLE_WORKER_TIMEOUT_SECONDS = 10.0
BUSY_WORKER_TIMEOUT_SECONDS = 360.0
SPAWN_TIMEOUT_SECONDS = 30.0
USE_GBATCH = False
GBATCH_GPUS = "2"
GBATCH_TIME = "2:00:00"

LOGGER = logging.getLogger(__name__)


def load_model_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for config_path in sorted(MODELS_PATH.glob("*/config.yaml")):
        with config_path.open() as handle:
            configs.update((yaml.safe_load(handle) or {}).get("models", {}))
    return configs


MODEL_CONFIGS = load_model_configs()


@dataclass(slots=True)
class Job:
    request_id: str
    client_id: bytes
    model_name: str
    payload: dict[str, Any]
    images: list[bytes] = field(default_factory=list)


@dataclass(slots=True)
class Worker:
    model_name: str
    process: subprocess.Popen[Any] | None
    started_at: float


@dataclass(slots=True)
class BrokerState:
    queue: deque[Job] = field(default_factory=deque)
    workers: dict[bytes, Worker] = field(default_factory=dict)
    inflight_by_worker: dict[bytes, Job] = field(default_factory=dict)
    last_seen_by_worker: dict[bytes, float] = field(default_factory=dict)
    workers_by_model: dict[str, deque[bytes]] = field(default_factory=dict)
    idle_workers: set[bytes] = field(default_factory=set)
    spawning_by_model: dict[str, bytes] = field(default_factory=dict)


def get_model_config(model_name: str) -> dict[str, Any] | None:
    return MODEL_CONFIGS.get(model_name)


def build_worker_command(model_name: str, model_config: dict[str, Any]) -> tuple[bytes, list[str]]:
    worker_token = str(uuid.uuid4())
    worker_id = f"{model_name}-{worker_token}".encode("utf-8")
    worker_path = MODELS_PATH / model_config["basefolder"]
    python_path = worker_path / ".venv/bin/python"
    worker_file = worker_path / "worker.py"
    assert python_path.is_file(), python_path
    assert worker_file.is_file(), worker_file
    command = [
        str(python_path),
        str(worker_file),
        "--model-id",
        model_name,
        "--router-connect",
        ROUTER_ADDRESS,
        "--worker-id",
        worker_token,
    ]
    if model_config.get("tp") is not None:
        command.extend(["--tp", str(model_config["tp"])])
    if not USE_GBATCH:
        return worker_id, command
    gpu_memory = f"{model_config['gpu_memory']}M"
    return worker_id, [
        "gbatch",
        "--gpus",
        GBATCH_GPUS,
        "--shared",
        "--gpu-memory",
        gpu_memory,
        "--time",
        GBATCH_TIME,
        *command,
    ]


def send_client_payload(socket: zmq.Socket, client_id: bytes, payload: dict[str, Any]) -> None:
    socket.send_multipart([client_id, b"", json.dumps(payload).encode("utf-8")])


def enqueue_job(state: BrokerState, client_id: bytes, payload: dict[str, Any], images: list[bytes]) -> None:
    model_name = payload.get("model_name") or payload.get("model")
    assert model_name is not None, payload
    state.queue.append(
        Job(
            request_id=payload["request_id"],
            client_id=client_id,
            model_name=model_name,
            payload=payload,
            images=images,
        )
    )
    LOGGER.info("queued request_id=%s model=%s images=%s", payload["request_id"], model_name, len(images))


def spawn_worker(state: BrokerState, model_name: str) -> None:
    model_config = get_model_config(model_name)
    assert model_config is not None, model_name
    worker_id, command = build_worker_command(model_name, model_config)
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    state.workers[worker_id] = Worker(model_name=model_name, process=process, started_at=time.monotonic())
    state.spawning_by_model[model_name] = worker_id
    LOGGER.info("spawned worker=%s pid=%s model=%s", worker_id.decode(), process.pid, model_name)
    

def remove_worker(state: BrokerState, worker_id: bytes) -> None:
    worker = state.workers.pop(worker_id, None)
    if worker is None:
        return
    state.last_seen_by_worker.pop(worker_id, None)
    state.idle_workers.discard(worker_id)
    if state.spawning_by_model.get(worker.model_name) == worker_id:
        state.spawning_by_model.pop(worker.model_name, None)
    workers = state.workers_by_model.get(worker.model_name)
    if workers is not None:
        if worker_id in workers:
            workers.remove(worker_id)
        if not workers:
            del state.workers_by_model[worker.model_name]
    job = state.inflight_by_worker.pop(worker_id, None)
    if job is not None:
        state.queue.appendleft(job)
        LOGGER.warning("requeued request_id=%s after worker=%s died", job.request_id, worker_id.decode())


def mark_worker_ready(state: BrokerState, worker_id: bytes, model_name: str, now: float) -> None:
    worker = state.workers.get(worker_id)
    if worker is None:
        state.workers[worker_id] = Worker(model_name=model_name, process=None, started_at=now)
    else:
        worker.model_name = model_name
    state.last_seen_by_worker[worker_id] = now
    if state.spawning_by_model.get(model_name) == worker_id:
        state.spawning_by_model.pop(model_name, None)
    workers = state.workers_by_model.setdefault(model_name, deque())
    if worker_id not in workers:
        workers.append(worker_id)
    if worker_id in state.inflight_by_worker:
        state.idle_workers.discard(worker_id)
    else:
        state.idle_workers.add(worker_id)


def handle_client_message(socket: zmq.Socket, state: BrokerState, frames: list[bytes]) -> None:
    client_id = frames[0]
    payload = json.loads(frames[2].decode("utf-8"))
    model_name = payload.get("model_name") or payload.get("model")
    assert model_name is not None, payload
    if model_name not in MODEL_CONFIGS and not state.workers_by_model.get(model_name):
        send_client_payload(
            socket,
            client_id,
            {
                "type": "ERROR",
                "req_id": payload["request_id"],
                "message": f"Unknown model '{model_name}'",
            },
        )
        LOGGER.warning("rejected request_id=%s unknown model=%s", payload["request_id"], model_name)
        return
    enqueue_job(state, client_id, payload, frames[3:])


def handle_worker_message(
    socket: zmq.Socket,
    state: BrokerState,
    frames: list[bytes],
    now: float,
) -> None:
    worker_id, raw_payload = frames
    payload = json.loads(raw_payload.decode("utf-8"))
    model_name = payload.get("model_name") or payload.get("model")
    if model_name is None:
        worker = state.workers.get(worker_id)
        assert worker is not None, worker_id
        model_name = worker.model_name
    mark_worker_ready(state, worker_id, model_name, now)
    message_type = payload["type"]
    if message_type == "HEARTBEAT":
        return
    if message_type not in {"SUCCESS", "ERROR"}:
        LOGGER.warning("ignored worker=%s message_type=%s", worker_id.decode(), message_type)
        return
    job = state.inflight_by_worker.pop(worker_id, None)
    state.idle_workers.add(worker_id)
    if job is None:
        LOGGER.warning("worker=%s returned without inflight job", worker_id.decode())
        return
    assert job.request_id == payload["req_id"], (job.request_id, payload["req_id"])
    send_client_payload(socket, job.client_id, payload)
    LOGGER.info("forwarded req_id=%s type=%s", payload["req_id"], message_type)


def receive_message(socket: zmq.Socket, state: BrokerState) -> None:
    if not socket.poll(timeout=POLL_TIMEOUT_MS):
        return
    frames = socket.recv_multipart()
    now = time.monotonic()
    if len(frames) == 2:
        handle_worker_message(socket, state, frames, now)
        return
    if len(frames) >= 3 and frames[1] == b"":
        handle_client_message(socket, state, frames)
        return
    LOGGER.warning("ignored malformed message with %s frames", len(frames))


def spawn_missing_workers(state: BrokerState) -> None:
    for model_name in {job.model_name for job in state.queue}:
        if state.workers_by_model.get(model_name):
            continue
        if state.spawning_by_model.get(model_name) is not None:
            continue
        if get_model_config(model_name) is None:
            continue
        spawn_worker(state, model_name)


def purge_dead_workers(state: BrokerState) -> None:
    now = time.monotonic()
    for worker_id, worker in list(state.workers.items()):
        if worker.process is not None and worker.process.poll() is not None:
            LOGGER.warning("worker exited worker=%s", worker_id.decode())
            remove_worker(state, worker_id)
            continue
        last_seen = state.last_seen_by_worker.get(worker_id)
        if last_seen is None:
            if now - worker.started_at <= SPAWN_TIMEOUT_SECONDS:
                continue
            LOGGER.warning("worker spawn timed out worker=%s", worker_id.decode())
            remove_worker(state, worker_id)
            continue
        timeout_seconds = (
            BUSY_WORKER_TIMEOUT_SECONDS
            if worker_id in state.inflight_by_worker
            else IDLE_WORKER_TIMEOUT_SECONDS
        )
        if now - last_seen <= timeout_seconds:
            continue
        LOGGER.warning("worker timed out worker=%s timeout=%ss", worker_id.decode(), timeout_seconds)
        remove_worker(state, worker_id)


def next_idle_worker(state: BrokerState, model_name: str) -> bytes | None:
    workers = state.workers_by_model.get(model_name)
    if workers is None:
        return None
    for _ in range(len(workers)):
        worker_id = workers[0]
        workers.rotate(-1)
        if worker_id in state.idle_workers:
            return worker_id
    return None


def dispatch_jobs(socket: zmq.Socket, state: BrokerState) -> None:
    for _ in range(len(state.queue)):
        job = state.queue.popleft()
        worker_id = next_idle_worker(state, job.model_name)
        if worker_id is None:
            state.queue.append(job)
            continue
        state.idle_workers.remove(worker_id)
        state.inflight_by_worker[worker_id] = job
        socket.send_multipart([worker_id, b"", json.dumps(job.payload).encode("utf-8"), *job.images])
        LOGGER.info("dispatched request_id=%s worker=%s", job.request_id, worker_id.decode())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    context = zmq.Context.instance()
    socket = context.socket(zmq.ROUTER)
    socket.bind(ROUTER_ADDRESS)
    LOGGER.info("broker listening on %s", ROUTER_ADDRESS)
    state = BrokerState()
    while True:
        receive_message(socket, state)
        spawn_missing_workers(state)
        purge_dead_workers(state)
        dispatch_jobs(socket, state)


if __name__ == "__main__":
    main()
