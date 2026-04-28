import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import zmq

from orchestra.config import load_config
from orchestra.hardware import detect_hardware
from orchestra.job_store import append_job_event
from orchestra.metrics import emit
from orchestra.model_store import model_status
from orchestra.process_manager import start_process
from orchestra.registry import (
    ModelSpec,
    engine_for_model,
    load_registry,
    model_by_name,
    runtime_for_model,
)
from orchestra.runtime import build_worker_command, compatible, env_status

POLL_TIMEOUT_MS = 10
IDLE_PRESSURE_EVICT_SECONDS = 3.0
IDLE_NO_QUEUE_EVICT_SECONDS = 3.0
STOPPING_WORKER_GRACE_SECONDS = 10.0
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Job:
    request_id: str
    client_id: bytes
    payload: dict[str, Any]
    images: list[bytes] = field(default_factory=list)


class WorkerStatus(Enum):
    IDLE = "IDLE"
    BUSY = "BUSY"
    STOPPING = "STOPPING"
    WAITING = "WAITING"


@dataclass(slots=True)
class Worker:
    key: str
    model_name: str
    process: Any
    started_at: float
    status: WorkerStatus = WorkerStatus.WAITING
    idle_since: float | None = None
    stopping_since: float | None = None


@dataclass(slots=True)
class WorkerPool:
    busy_workers: set[str] = field(default_factory=set)
    idle_workers: set[str] = field(default_factory=set)
    wait_workers: set[str] = field(default_factory=set)

    def add_waiting_worker(self, worker_id: str) -> None:
        self.wait_workers.add(worker_id)

    def set_idle(self, worker_id: str) -> None:
        self.wait_workers.discard(worker_id)
        self.busy_workers.discard(worker_id)
        self.idle_workers.add(worker_id)

    def set_busy(self, worker_id: str) -> None:
        self.wait_workers.discard(worker_id)
        self.idle_workers.discard(worker_id)
        self.busy_workers.add(worker_id)

    def discard_worker_id(self, worker_id: str) -> None:
        self.busy_workers.discard(worker_id)
        self.idle_workers.discard(worker_id)
        self.wait_workers.discard(worker_id)

    def is_empty(self) -> bool:
        return len(self.busy_workers) + len(self.idle_workers) + len(self.wait_workers) == 0


@dataclass(slots=True)
class BrokerState:
    worker_registry: dict[str, WorkerPool] = field(default_factory=dict)
    worker_map: dict[str, Worker] = field(default_factory=dict)
    jobs_registry: dict[str, deque[Job]] = field(default_factory=dict)
    pending_jobs: dict[str, Job] = field(default_factory=dict)
    inflight_by_worker: dict[str, Job] = field(default_factory=dict)

    def enqueue_job(self, job: Job) -> None:
        key = worker_key(job.payload)
        self.jobs_registry.setdefault(key, deque()).append(job)

    def spawn_worker_for_job(self, job: Job) -> None:
        model = model_for_payload(job.payload)
        registry = load_registry()
        config = load_config()
        runtime = runtime_for_model(registry, model, job.payload.get("runtime"))
        engine = engine_for_model(registry, model, job.payload.get("engine"))
        worker_id = str(uuid.uuid4())
        command = build_worker_command(
            model=model,
            runtime=runtime,
            engine=engine,
            router_address=config.broker_address,
            worker_id=worker_id,
            args_per_model=job.payload.get("args_per_model", {}),
        )
        LOGGER.info("spawning worker key=%s command=%s", worker_key(job.payload), " ".join(command))
        emit("worker_spawn", key=worker_key(job.payload), model=model.name, command=command)
        process = start_process(
            command=command,
            name=f"orchestra-{worker_id}",
            accelerator=runtime.accelerator,
            logs=config.logs,
            manager=config.process_manager,
        )
        key = worker_key(job.payload)
        self.worker_map[worker_id] = Worker(
            key=key,
            model_name=model.name,
            process=process,
            started_at=time.monotonic(),
        )
        self.worker_registry.setdefault(key, WorkerPool()).add_waiting_worker(worker_id)


def worker_key(payload: dict[str, Any]) -> str:
    model = payload["model_name"]
    spec = model_by_name(load_registry(), model)
    engine = payload.get("engine", spec.default_engine)
    runtime = payload.get("runtime", spec.default_runtime)
    args = json.dumps(payload.get("args_per_model", {}), sort_keys=True)
    return f"{model}|{engine}|{runtime}|{args}"


def set_worker_idle(worker: Worker) -> None:
    if worker.status != WorkerStatus.IDLE:
        worker.idle_since = time.monotonic()
    worker.status = WorkerStatus.IDLE
    worker.stopping_since = None


def set_worker_busy(worker: Worker) -> None:
    worker.status = WorkerStatus.BUSY
    worker.idle_since = None
    worker.stopping_since = None


def model_for_payload(payload: dict[str, Any]) -> ModelSpec:
    return model_by_name(load_registry(), payload["model_name"])


def error_payload(
    request_id: str | None,
    code: str,
    message: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "ERROR",
        "request_id": request_id,
        "code": code,
        "message": message,
    }
    if model_name is not None:
        payload["model_name"] = model_name
    return payload


def send_client_payload(socket: zmq.Socket, client_id: bytes, payload: dict[str, Any]) -> None:
    socket.send_multipart([client_id, b"", json.dumps(payload).encode("utf-8")])


def validate_job(payload: dict[str, Any]) -> dict[str, Any] | None:
    request_id = payload.get("request_id")
    model_name = payload.get("model_name")
    registry = load_registry()
    config = load_config()
    matches = [
        model
        for model in registry.models.values()
        if model.name == model_name or model_name in model.aliases
    ]
    if not matches:
        emit("request_rejected", code="UNKNOWN_MODEL", model=model_name)
        return error_payload(
            request_id,
            "UNKNOWN_MODEL",
            f"Unknown model '{model_name}'",
            model_name,
        )

    model = matches[0]
    payload["model_name"] = model.name
    runtime = runtime_for_model(registry, model, payload.get("runtime"))
    engine_for_model(registry, model, payload.get("engine"))
    hardware = detect_hardware()

    if model_status(config, model) != "downloaded":
        emit("request_rejected", code="MODEL_NOT_DOWNLOADED", model=model_name)
        return error_payload(
            request_id,
            "MODEL_NOT_DOWNLOADED",
            f"Model '{model_name}' is not downloaded",
            model_name,
        )
    if env_status(runtime) != "ready":
        emit("request_rejected", code="ENV_NOT_READY", model=model_name, runtime=runtime.name)
        return error_payload(
            request_id,
            "ENV_NOT_READY",
            f"Runtime '{runtime.name}' is not ready",
            model_name,
        )
    if not compatible(runtime, hardware):
        emit(
            "request_rejected",
            code="INCOMPATIBLE_RUNTIME",
            model=model_name,
            runtime=runtime.name,
        )
        return error_payload(
            request_id,
            "INCOMPATIBLE_RUNTIME",
            f"Runtime '{runtime.name}' is not compatible",
            model_name,
        )
    if hardware.gpus:
        free_vram = max(gpu.free_mb for gpu in hardware.gpus)
        if free_vram < model.min_vram_mb:
            emit(
                "request_vram_warning",
                model=model.name,
                free_vram_mb=free_vram,
                advisory_min_vram_mb=model.min_vram_mb,
            )
    return None


def receive_worker_payload(frames: list[bytes]) -> tuple[str, dict[str, Any]]:
    identity = frames[0].decode("utf-8")
    payload = json.loads(frames[1].decode("utf-8"))
    return identity, payload


def worker_uuid(identity: str, model_name: str) -> str:
    return identity.replace(f"{model_name}-", "", 1)


def handle_worker_message(socket: zmq.Socket, state: BrokerState, frames: list[bytes]) -> None:
    identity, payload = receive_worker_payload(frames)
    message_type = payload["type"]
    model_name = payload.get("model_name")
    worker_id = worker_uuid(identity, model_name)
    worker = state.worker_map.get(worker_id)
    key = worker.key if worker is not None else None

    if message_type == "HEARTBEAT" and key is not None:
        if worker.status == WorkerStatus.STOPPING:
            return
        emit("worker_heartbeat", worker_id=worker_id, model=model_name, key=key)
        set_worker_idle(worker)
        state.worker_registry.setdefault(key, WorkerPool()).set_idle(worker_id)
        return

    if message_type in {"SUCCESS", "ERROR"} and key is not None:
        emit("worker_response", type=message_type, worker_id=worker_id, model=model_name, key=key)
        append_job_event(
            "completed",
            payload.get("request_id"),
            worker_id=worker_id,
            model=model_name,
            type=message_type,
        )
        set_worker_idle(worker)
        state.worker_registry[key].set_idle(worker_id)
        job = state.inflight_by_worker.pop(worker_id, None)
        request_id = payload.get("request_id")
        job = job or state.pending_jobs.pop(request_id)
        state.pending_jobs.pop(request_id, None)
        payload.setdefault("profile", {})["broker_forward_started_time"] = time.time_ns()
        socket.send_multipart(
            [job.client_id, b"", json.dumps(payload).encode("utf-8"), *frames[2:]]
        )


def handle_client_message(socket: zmq.Socket, state: BrokerState, frames: list[bytes]) -> None:
    client_id = frames[0]
    payload = json.loads(frames[2].decode("utf-8"))
    payload.setdefault("profile", {})["broker_received_request_time"] = time.time_ns()
    error = validate_job(payload)
    if error is not None:
        append_job_event(
            "rejected",
            payload.get("request_id"),
            model=payload.get("model_name"),
            code=error["code"],
        )
        send_client_payload(socket, client_id, error)
        return
    emit("request_queued", request_id=payload["request_id"], model=payload["model_name"])
    append_job_event("queued", payload["request_id"], model=payload["model_name"])
    state.enqueue_job(Job(payload["request_id"], client_id, payload, frames[3:]))


def receive_message(socket: zmq.Socket, state: BrokerState) -> None:
    if not socket.poll(timeout=POLL_TIMEOUT_MS):
        return
    frames = socket.recv_multipart()
    if len(frames) >= 3 and frames[1] == b"":
        handle_client_message(socket, state, frames)
        return
    handle_worker_message(socket, state, frames)


def shutdown_idle_worker(
    socket: zmq.Socket,
    state: BrokerState,
    worker_id: str,
    reason: str,
) -> None:
    worker = state.worker_map[worker_id]
    if worker.status == WorkerStatus.STOPPING:
        return
    pool = state.worker_registry.get(worker.key)
    if pool is not None:
        pool.discard_worker_id(worker_id)
    worker.status = WorkerStatus.STOPPING
    worker.idle_since = None
    worker.stopping_since = time.monotonic()
    destination = f"{worker.model_name}-{worker_id}".encode("utf-8")
    payload = {"type": "SHUTDOWN", "model_name": worker.model_name}
    socket.send_multipart([destination, b"", json.dumps(payload).encode("utf-8")])
    emit(
        "worker_shutdown_request",
        worker_id=worker_id,
        model=worker.model_name,
        key=worker.key,
        reason=reason,
    )
    append_job_event(
        "worker_shutdown",
        None,
        worker_id=worker_id,
        model=worker.model_name,
        key=worker.key,
        reason=reason,
    )


def idle_pressure_blocks_spawn(socket: zmq.Socket, state: BrokerState, target_key: str) -> bool:
    now = time.monotonic()
    blocked = False
    for worker_id, worker in list(state.worker_map.items()):
        if worker.key == target_key:
            continue
        if worker.status == WorkerStatus.STOPPING:
            blocked = True
            continue
        if worker.status != WorkerStatus.IDLE or worker.idle_since is None:
            continue
        blocked = True
        if now - worker.idle_since >= IDLE_PRESSURE_EVICT_SECONDS:
            shutdown_idle_worker(socket, state, worker_id, "queued_model_pressure")
    return blocked


def has_stopping_worker_for_key(state: BrokerState, target_key: str) -> bool:
    return any(
        worker.key == target_key and worker.status == WorkerStatus.STOPPING
        for worker in state.worker_map.values()
    )


def evict_idle_workers_without_work(socket: zmq.Socket, state: BrokerState) -> None:
    now = time.monotonic()
    active_queue_keys = {
        key for key, job_queue in state.jobs_registry.items() if len(job_queue) > 0
    }
    for worker_id, worker in list(state.worker_map.items()):
        if worker.status != WorkerStatus.IDLE or worker.idle_since is None:
            continue
        if worker.key in active_queue_keys:
            continue
        if now - worker.idle_since >= IDLE_NO_QUEUE_EVICT_SECONDS:
            shutdown_idle_worker(socket, state, worker_id, "idle_no_queue")


def dispatch_jobs(socket: zmq.Socket, state: BrokerState) -> None:
    evict_idle_workers_without_work(socket, state)
    queues_to_remove = []
    for key, job_queue in state.jobs_registry.items():
        if not job_queue:
            queues_to_remove.append(key)
            continue

        pool = state.worker_registry.get(key)
        if pool is not None and pool.is_empty():
            del state.worker_registry[key]
            pool = None
        if pool is None:
            if has_stopping_worker_for_key(state, key):
                continue
            if idle_pressure_blocks_spawn(socket, state, key):
                continue
            state.spawn_worker_for_job(job_queue[0])
            continue
        if pool.wait_workers or not pool.idle_workers:
            continue

        worker_id = pool.idle_workers.pop()
        pool.set_busy(worker_id)
        set_worker_busy(state.worker_map[worker_id])
        job = job_queue.popleft()
        state.pending_jobs[job.request_id] = job
        state.inflight_by_worker[worker_id] = job
        emit("request_dispatch", request_id=job.request_id, worker_id=worker_id, key=key)
        append_job_event("dispatched", job.request_id, worker_id=worker_id, key=key)
        job.payload.setdefault("profile", {})["broker_dispatch_started_time"] = time.time_ns()
        model_name = job.payload["model_name"]
        destination = f"{model_name}-{worker_id}".encode("utf-8")
        socket.send_multipart(
            [destination, b"", json.dumps(job.payload).encode("utf-8"), *job.images]
        )

    for key in queues_to_remove:
        del state.jobs_registry[key]


def purge_dead_workers(state: BrokerState) -> None:
    for worker_id, worker in list(state.worker_map.items()):
        if (
            worker.status == WorkerStatus.STOPPING
            and worker.stopping_since is not None
            and time.monotonic() - worker.stopping_since >= STOPPING_WORKER_GRACE_SECONDS
            and worker.process.poll() is None
        ):
            worker.process.terminate()
        if worker.process.poll() is None:
            continue
        pool = state.worker_registry.get(worker.key)
        if pool is not None:
            pool.discard_worker_id(worker_id)
        job = state.inflight_by_worker.pop(worker_id, None)
        if job is not None:
            emit(
                "worker_dead_requeue",
                worker_id=worker_id,
                key=worker.key,
                request_id=job.request_id,
            )
            append_job_event("requeued", job.request_id, worker_id=worker_id, key=worker.key)
            state.pending_jobs.pop(job.request_id, None)
            state.jobs_registry.setdefault(worker.key, deque()).appendleft(job)
        del state.worker_map[worker_id]
        if worker.key in state.worker_registry and state.worker_registry[worker.key].is_empty():
            del state.worker_registry[worker.key]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    socket = zmq.Context.instance().socket(zmq.ROUTER)
    socket.bind(config.broker_bind_address)
    LOGGER.info("broker listening on %s", config.broker_bind_address)
    state = BrokerState()
    while True:
        receive_message(socket, state)
        purge_dead_workers(state)
        dispatch_jobs(socket, state)


if __name__ == "__main__":
    main()
