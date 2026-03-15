import os
import zmq
import time
import json
import yaml
import uuid
import pkgutil
import logging
import inspect
import subprocess

import models # import the models folder 
import importlib

from tqdm import tqdm
from enum import Enum
from typing import Any
from pathlib import Path
from collections import deque
from pydantic import BaseModel
from dataclasses import dataclass, field

# Constants

ROOT = Path(__file__).resolve().parent
WORKERS_PATH = ROOT / "models"
ROUTER_ADDRESS = "tcp://10.10.151.14:5556"
POLL_TIMEOUT_MS = 1000
IDLE_WORKER_TIMEOUT_SECONDS = 100.0
BUSY_WORKER_TIMEOUT_SECONDS = 3600.0
SPAWN_TIMEOUT_SECONDS = 120.0
GBATCH_GPUS = "2"
GBATCH_TIME = "2:00:00"

LOGGER = logging.getLogger(__name__)

MODELS_REGISTRY: Dict[str, Type[BaseModel]] = {}
MODELS_CONFIG: Dict[str, Any] = {}

## Startup functions

def discover_models():
    mods=[m for _,m,p in pkgutil.walk_packages(models.__path__,models.__name__+".") if p]
    for m in tqdm(mods):
        mod=importlib.import_module(f"{m}.schema"); cfg=(WORKERS_PATH/m.split(".")[-1]/"config.yaml")
        if cfg.is_file(): MODELS_CONFIG.update((yaml.safe_load(cfg.open()) or {}).get("models",{}))
        for n,o in inspect.getmembers(mod,inspect.isclass):
            if issubclass(o,BaseModel) and o is not BaseModel: MODELS_REGISTRY[f"{m.split('.')[-1]}.{n}"]=o

    print("Dynamically loaded everything.")
    print(f"Discovered models: {list(MODELS_REGISTRY.keys())}")

discover_models()

# Classes definitions

## Job definition

@dataclass(slots=True)
class Payload:
    model_name: str
    config: dict[str, Any]

@dataclass(slots=True)
class Job:
    request_id: str
    client_id: bytes
    payload: Payload
    images: list[bytes] = field(default_factory=list)


## Worker definition

class WorkerStatus(Enum):
    IDLE = "IDLE"
    BUSY = "BUSY"    
    WAITING = "WAITING"

@dataclass(slots=True)
class Worker:
    model: str
    process: Any
    started_at: float
    status: WorkerStatus = WorkerStatus.WAITING


@dataclass(slots=True)
class WorkerPool:
    busy_workers: set[bytes] = field(default_factory=set)
    idle_workers: set[bytes] = field(default_factory=set)
    wait_workers: set[bytes] = field(default_factory=set)

    def add_waiting_worker(self, worker_id: bytes):
        self.wait_workers.add(worker_id)

    def set_idle(self, worker_id: bytes):
        self.wait_workers.discard(worker_id)
        self.busy_workers.discard(worker_id)
        self.idle_workers.add(worker_id)

    def set_busy(self, worker_id: bytes):
        self.wait_workers.discard(worker_id)
        self.idle_workers.discard(worker_id)
        self.busy_workers.add(worker_id)

    @property
    def total_count(self) -> int:
        return len(self.idle_workers + self.busy_workers + self.wait_workers)


# Broker state definition
@dataclass(slots=True)
class BrokerState:
    worker_registry: dict[str, WorkerPool] = field(default_factory=dict)
    # job_queue: deque[Job] = field(default_factory=deque)
    jobs_registry: dict[str, deque[Job]] = field(default_factory=dict)
    pending_jobs: dict[str, Job] = field(default_factory=dict) # request_id -> Job
    
    def has_active_worker_by_model(self, model_name: str) -> bool:
        """Checks if a worker exists and is actually populated."""
        return bool(self.worker_registry.get(model_name))

    def get_model_config(self, model_name: str) -> dict[str, Any] | None:
        return MODELS_CONFIG.get(model_name)

    def enqueue_job(self, job):  
        
        job_model = job.payload["model_name"]

        if job_model not in self.jobs_registry:
            self.jobs_registry[job_model] = deque()
        
        self.jobs_registry[job_model].append(job)        
        print("successfully added new job")
    
    def spawn_worker_for_model(self, model_name: str):
        model_config = self.get_model_config(model_name)
        if model_config is None:
            print(f"model {model_name} not found in config")
            return
        
        worker_id, command = build_command_for_model_bare(model_name, model_config) # bare metal without worrying about GPUs fill
        # worst but might work for now since gflow doesn't allow for shared multi-gpu allocation
        # worker_id, command = build_command_for_model_gflow(model_name, model_config) 
        print(f"spawning worker for model {model_name} with command: {' '.join(command)}")
        process = subprocess.Popen(command)

        self.worker_registry[model_name] = WorkerPool()
        print(f"worker for model {model_name} spawned with id {worker_id} and process id {process.pid}")
        self.worker_registry[model_name].add_waiting_worker(worker_id) # we can use the process id as the worker id for now, but we need to make sure that when the worker is

        # when does it become idle -> how to deal with difference from WAITING maybe there is no difference

    #TODO maybe do it better here as full function instead of the schifo there    
    # def assign_job_to_worker(self, model_name: str, worker_id: bytes, job_to_send: Job):

    #     # remove the job from the queue
    #     # job_to_send = self.jobs_registry[model_name].popleft() # remove the job from the queues        
    #     # append it to known registry to be able to forward the response to the correct client when the worker sends back the result
    #     self.pending_jobs[job_to_send.request_id] = job_to_send
    #     print(f"assigned job {job_to_send.request_id} to worker {worker_id} for model {model_name}")

# the queue shuould have a maximum number of jobs it can have I think right?

## UTILS

def build_command_for_model_gflow(model_name: str, model_config: dict[str, Any]) -> list[str]:
    worker_path = WORKERS_PATH / model_config["basefolder"]
    python_path = worker_path / ".venv/bin/python"
    worker_file = worker_path / "worker.py"
    gpu_memory = str(model_config.get("gpu_memory")) + "M"
    print(f"building command for model {model_name} with gpu memory {gpu_memory}")
    assert python_path.is_file(), python_path
    assert worker_file.is_file(), worker_file
    worker_id = str(uuid.uuid4())
    return worker_id, [
        "gbatch",
        "--gpus",
        GBATCH_GPUS,
        "--shared",
        "--gpu-memory",
        gpu_memory,
        "--time",
        GBATCH_TIME,
        str(python_path),
        str(worker_file),
        "--model-id",
        model_name,
        "--router-connect",
        ROUTER_ADDRESS,       
        "--worker-id",
        worker_id
    ]


def build_command_for_model_bare(model_name: str, model_config: dict[str, Any]) -> list[str]:
    worker_path = WORKERS_PATH / model_config["basefolder"]
    python_path = worker_path / ".venv/bin/python"
    worker_file = worker_path / "worker.py"
    gpu_memory = str(model_config.get("gpu_memory")) + "M"
    print(f"building command for model {model_name} with gpu memory {gpu_memory}")
    assert python_path.is_file(), python_path
    assert worker_file.is_file(), worker_file
    worker_id = str(uuid.uuid4())
    return worker_id, [      
        str(python_path),
        str(worker_file),
        "--model-id",
        model_name,
        "--router-connect",
        ROUTER_ADDRESS,       
        "--worker-id",
        worker_id
    ]

    # we need to also add optional args per model e.g.
    #     "--tp",


def send_client_payload(socket: zmq.Socket, client_id: bytes, payload: dict[str, Any]) -> None:
    socket.send_multipart([client_id, b"", json.dumps(payload).encode("utf-8")])

def receive_worker_payload(frames: list[bytes]) -> tuple[str, dict[str, Any]]:
    worker_id, raw_payload = frames
    worker_id = worker_id.decode("utf-8")
    payload = json.loads(raw_payload.decode("utf-8"))
    return worker_id, payload    


def handle_worker_message(
    socket: zmq.Socket, state: BrokerState, frames: list[bytes], now: float
) -> None:
    worker_id, raw_payload = frames
    worker_id, payload = receive_worker_payload(frames)
    
    type_answer = payload["type"]    

    # to double check
    if type_answer == "HEARTBEAT":
        model_name = payload["model_name"]
        worker_id = worker_id.replace(model_name+"-", "")
        print(f"Received heartbeat from worker {worker_id} for model {model_name}")
        # if the model is already loaded -> crashes but is this what we want? maybe we can just ignore the heartbeat if the worker is already in the registry, but if it's not in the registry we need to add it and set it to idle
        state.worker_registry[model_name].set_idle(worker_id)

    if type_answer == "SUCCESS":
        model_name = payload["model_name"]
        worker_id = worker_id.replace(model_name+"-", "")
        state.worker_registry[model_name].set_idle(worker_id)
        
        req_id = payload.get("req_id")
        print('req_id', req_id)
        job = state.pending_jobs.pop(req_id)
        client_id = job.client_id
        
        if client_id:            
            socket.send_multipart([client_id, b"", raw_payload])
            print(f"Forwarded result {req_id} to Client.")


def handle_client_message(socket: zmq.Socket, state: BrokerState, frames: list[bytes]) -> None:
    client_id = frames[0]
    payload = json.loads(frames[2].decode("utf-8"))

    request_id = payload.get("request_id")

    # here we need to check if the model has the correct structure!
    new_job = Job(request_id=request_id, client_id=client_id, payload=payload, images=frames[3:])

    model_name = new_job.payload["model_name"]
    model_path_config = state.get_model_config(model_name) 
    if model_path_config is None:
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

    state.enqueue_job(new_job)


def receive_message(socket, state):
    if not socket.poll(timeout=POLL_TIMEOUT_MS):
        return

    frames = socket.recv_multipart()
    now = time.monotonic()

    # message from worker
    if len(frames) == 2:
        handle_worker_message(socket, state, frames, now)
        return

    # message from client
    if len(frames) >= 3 and frames[1] == b"":
        handle_client_message(socket, state, frames)
        return
    LOGGER.warning("ignored malformed message with %s frames", len(frames))


def dispatch_jobs(socket, state: BrokerState):

    # even this one runs once every 100 milliseconds

    # read first job in the list

    for model_name, job_queue in state.jobs_registry.items():
        print(f"model {model_name} has {len(job_queue)} jobs in the queue")

        if len(job_queue) == 0:
            # remove the entry from the registry to avoid iterating over it in the future if there are no jobs for this model
            continue

        # is there an active worker for this model?
        active_workers = state.worker_registry.get(model_name) # should return a list of workers better if we it as a pool with idle and busy workers        
        
        if not active_workers:            
            state.spawn_worker_for_model(model_name) # this should spawn a worker for the model and add it to the registry with status loading
            
            # need to spawn a worker for this model, but before that we need to check if we have the model in the config file

        if active_workers and len(active_workers.idle_workers) == 0:            
            continue
        
        if active_workers and len(active_workers.wait_workers) > 0:            
            continue
        
        # TODO fix it and make it better -> cleaner
        if active_workers and len(active_workers.idle_workers) > 0:            

            chosen_worker_id = active_workers.idle_workers.pop()            
            active_workers.set_busy(chosen_worker_id) # here we need to assign the job to the worker and update the worker status to busy            
            job = job_queue.popleft() # remove the job from the queue
            state.pending_jobs[job.request_id] = job

            # send the job to the worker
            metadata_bytes = json.dumps(job.payload).encode('utf-8')
            destination_worker_id = f"{model_name}-{chosen_worker_id}".encode()
            frames_to_send = [destination_worker_id, b"", metadata_bytes] + job.images
            socket.send_multipart(frames_to_send)


def purge_dead_workers(state):
    pass


# Main loop

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    context = zmq.Context.instance()
    socket = context.socket(zmq.ROUTER)
    socket.bind(ROUTER_ADDRESS)
    LOGGER.info("broker listening on %s", ROUTER_ADDRESS)
    state = BrokerState() 

    while True:
        receive_message(socket, state)
    
        dispatch_jobs(socket, state)

        # purge_dead_workers(state) #optional because workers kill themeselves when they are idle for too long, but we can also add this function to be sure that we don't have any dead workers in the registry
        # only needed for high performance high concurrency with known ahead time processing



if __name__ == "__main__":
    main()
