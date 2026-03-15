from __future__ import annotations

import json
import logging
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
import zmq

ROOT = Path(__file__).resolve().parent
WORKERS_PATH = ROOT / "workers"
ROUTER_ADDRESS = "tcp://10.10.151.14:5555"
POLL_TIMEOUT_MS = 1000
IDLE_WORKER_TIMEOUT_SECONDS = 100.0
BUSY_WORKER_TIMEOUT_SECONDS = 3600.0
SPAWN_TIMEOUT_SECONDS = 120.0
GBATCH_GPUS = "1"
GBATCH_TIME = "2:00:00"

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Job:
    request_id: str
    client_id: bytes
    model: str
    payload: dict[str, Any]
    images: list[bytes]


@dataclass(slots=True)
class Spawn:
    model: str
    process: subprocess.Popen[Any]
    started_at: float


@dataclass(slots=True)
class BrokerState:
    queue: deque[Job] = field(default_factory=deque)
    pending_clients: dict[str, bytes] = field(default_factory=dict)
    inflight_by_worker: dict[bytes, Job] = field(default_factory=dict)
    last_seen_by_worker: dict[bytes, float] = field(default_factory=dict)
    model_by_worker: dict[bytes, str] = field(default_factory=dict)
    workers_by_model: dict[str, deque[bytes]] = field(default_factory=dict)
    idle_workers: set[bytes] = field(default_factory=set)
    spawning_by_model: dict[str, Spawn] = field(default_factory=dict)


def get_model_config(model_name: str) -> dict[str, Any] | None:
    config_path = WORKERS_PATH / model_name.split("/", 1)[0] / "config.yaml"
    if not config_path.is_file():
        return None
    with config_path.open() as handle:
        return yaml.safe_load(handle)["models"].get(model_name)


def build_command_for_model(model_name: str, model_config: dict[str, Any]) -> list[str]:
    worker_path = WORKERS_PATH / model_config["basefolder"]
    python_path = worker_path / ".venv/bin/python"
    worker_file = worker_path / "worker.py"
    assert python_path.is_file(), python_path
    assert worker_file.is_file(), worker_file
    return [
        "gbatch",
        "--gpus",
        GBATCH_GPUS,
        "--time",
        GBATCH_TIME,
        str(python_path),
        str(worker_file),
        "--model-id",
        model_name,
        "--router-connect",
        ROUTER_ADDRESS,
        "--tp",
        str(model_config["tp"]),
    ]


def send_client_payload(socket: zmq.Socket, client_id: bytes, payload: dict[str, Any]) -> None:
    socket.send_multipart([client_id, b"", json.dumps(payload).encode("utf-8")])


def enqueue_job(
    state: BrokerState, client_id: bytes, payload: dict[str, Any], images: list[bytes]
) -> None:
    request_id = payload["request_id"]
    model_name = payload["model"]
    assert request_id not in state.pending_clients, request_id
    state.pending_clients[request_id] = client_id
    state.queue.append(Job(request_id, client_id, model_name, payload, images))
    LOGGER.info("queued request_id=%s model=%s images=%s", request_id, model_name, len(images))


def remove_worker(state: BrokerState, worker_id: bytes) -> None:
    state.last_seen_by_worker.pop(worker_id, None)
    state.idle_workers.discard(worker_id)
    model_name = state.model_by_worker.pop(worker_id, None)
    if model_name is not None:
        workers = state.workers_by_model[model_name]
        if worker_id in workers:
            workers.remove(worker_id)
        if not workers:
            del state.workers_by_model[model_name]
    job = state.inflight_by_worker.pop(worker_id, None)
    if job is not None:
        state.queue.appendleft(job)
        LOGGER.warning(
            "re-queued request_id=%s after worker=%s died",
            job.request_id,
            worker_id.decode(),
        )


def mark_worker_ready(state: BrokerState, worker_id: bytes, model_name: str, now: float) -> None:
    state.last_seen_by_worker[worker_id] = now
    state.model_by_worker[worker_id] = model_name
    state.spawning_by_model.pop(model_name, None)
    workers = state.workers_by_model.setdefault(model_name, deque())
    if worker_id not in workers:
        workers.append(worker_id)
    if worker_id not in state.inflight_by_worker:
        state.idle_workers.add(worker_id)


def handle_client_message(socket: zmq.Socket, state: BrokerState, frames: list[bytes]) -> None:
    client_id = frames[0]
    payload = json.loads(frames[2].decode("utf-8"))
    model_name = payload["model"]
    if not state.workers_by_model.get(model_name) and get_model_config(model_name) is None:
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
    socket: zmq.Socket, state: BrokerState, frames: list[bytes], now: float
) -> None:
    worker_id, raw_payload = frames
    payload = json.loads(raw_payload.decode("utf-8"))
    message_type = payload["type"]
    model_name = payload.get("model") or state.model_by_worker.get(worker_id)
    assert model_name is not None, worker_id
    mark_worker_ready(state, worker_id, model_name, now)
    if message_type == "HEARTBEAT":
        return
    if message_type not in {"SUCCESS", "ERROR"}:
        LOGGER.warning("ignored worker=%s message_type=%s", worker_id.decode(), message_type)
        return
    job = state.inflight_by_worker.pop(worker_id, None)
    if job is not None:
        assert job.request_id == payload["req_id"], (job.request_id, payload["req_id"])
    client_id = state.pending_clients.pop(payload["req_id"], None)
    if client_id is None:
        LOGGER.warning("missing client for req_id=%s", payload["req_id"])
        return
    send_client_payload(socket, client_id, payload)
    state.idle_workers.add(worker_id)
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
    now = time.monotonic()
    for model_name in {job.model for job in state.queue}:
        if state.workers_by_model.get(model_name):
            continue
        spawn = state.spawning_by_model.get(model_name)
        if (
            spawn is not None
            and spawn.process.poll() is None
            and now - spawn.started_at < SPAWN_TIMEOUT_SECONDS
        ):
            continue
        if spawn is not None:
            LOGGER.warning("spawn expired for model=%s", model_name)
            state.spawning_by_model.pop(model_name, None)
        model_config = get_model_config(model_name)
        if model_config is None:
            continue
        process = subprocess.Popen(build_command_for_model(model_name, model_config))
        state.spawning_by_model[model_name] = Spawn(model_name, process, now)
        LOGGER.info("spawned pid=%s model=%s", process.pid, model_name)


def purge_dead_workers(state: BrokerState) -> None:
    now = time.monotonic()
    for worker_id, last_seen in list(state.last_seen_by_worker.items()):
        timeout_seconds = (
            BUSY_WORKER_TIMEOUT_SECONDS
            if worker_id in state.inflight_by_worker
            else IDLE_WORKER_TIMEOUT_SECONDS
        )
        if now - last_seen <= timeout_seconds:
            continue
        LOGGER.warning(
            "worker timed out worker=%s timeout=%ss",
            worker_id.decode(),
            timeout_seconds,
        )
        remove_worker(state, worker_id)
    for model_name, spawn in list(state.spawning_by_model.items()):
        if spawn.process.poll() is None and now - spawn.started_at <= SPAWN_TIMEOUT_SECONDS:
            continue
        LOGGER.warning("spawn failed model=%s pid=%s", model_name, spawn.process.pid)
        state.spawning_by_model.pop(model_name, None)


def next_idle_worker(state: BrokerState, model_name: str) -> bytes | None:
    workers = state.workers_by_model.get(model_name)
    if not workers:
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
        worker_id = next_idle_worker(state, job.model)
        if worker_id is None:
            state.queue.append(job)
            continue
        state.idle_workers.remove(worker_id)
        assert worker_id not in state.inflight_by_worker, worker_id
        state.inflight_by_worker[worker_id] = job
        metadata = json.dumps(job.payload).encode("utf-8")
        socket.send_multipart([worker_id, b"", metadata, *job.images])
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
