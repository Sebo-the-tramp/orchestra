from orchestra.config import load_config
from orchestra.hardware import detect_hardware
from orchestra.nodes import local_node_status
from orchestra.process_manager import (
    GflowJob,
    gflow_job_id,
    gflow_queue_job_ids,
    missing_gflow_commands,
    process_manager_kind,
    safe_name,
    worker_gpus,
)
from orchestra.profiles import architecture_profile, default_llm_engine_runtime, setup_plan
from orchestra.registry import engine_for_model, load_registry, model_by_name, runtime_for_model
from orchestra.routing import candidate_status, candidates_for_task, choose_model
from orchestra.runtime import build_worker_command, env_status
from orchestra.scheduler import schedule_plan
from orchestra.snapshot import snapshot_data


def free_port() -> int:
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return int(port)


def test_registry_resolves_all_models() -> None:
    registry = load_registry()
    assert registry.models
    assert registry.runtimes
    assert registry.engines
    for model in registry.models.values():
        runtime_for_model(registry, model)
        engine_for_model(registry, model)


def test_qwen_llama_cpp_registered() -> None:
    registry = load_registry()
    model = registry.models["Qwen/Qwen2.5-7B-Instruct-GGUF"]
    assert model.default_engine == "llama_cpp"
    assert model.default_runtime == "llama_cpp_cuda"
    assert model.format == "gguf"


def test_requested_models_registered() -> None:
    registry = load_registry()
    expected = {
        "unsloth/Qwen3.6-35B-A3B-UD-Q4_K_S",
        "unsloth/Qwen3.5-9B-UD-Q4_K_XL",
        "unsloth/Qwen3.6-27B-Q4_1",
        "bartowski/Qwen_Qwen3-30B-A3B-Q4_K_M",
        "Qwen/Qwen3-8B-vllm-rocm",
        "BAAI/bge-m3-vllm-rocm",
        "google/translategemma-4b-it",
        "openai/whisper-large-v3-transformers",
        "Systran/faster-whisper-large-v3",
    }
    assert expected.issubset(registry.models)
    assert registry.models["google/translategemma-4b-it"].task == "translation"
    assert registry.models["BAAI/bge-m3-vllm-rocm"].task == "embedding"
    assert registry.models["Systran/faster-whisper-large-v3"].task == "speech_to_text"


def test_llama_server_command_keeps_model_defaults() -> None:
    registry = load_registry()
    model = registry.models["unsloth/Qwen3.5-9B-UD-Q4_K_XL"]
    runtime = runtime_for_model(registry, model)
    engine = engine_for_model(registry, model)
    command = build_worker_command(model, runtime, engine, "tcp://127.0.0.1:5556", "worker")
    assert "--llama-dir" in command
    assert "--ctx-size" in command
    assert "262144" in command


def test_model_aliases_and_worker_bool_args() -> None:
    registry = load_registry()
    model = model_by_name(registry, "qwen3.5-9b")
    runtime = runtime_for_model(registry, model)
    engine = engine_for_model(registry, model)
    command = build_worker_command(
        model,
        runtime,
        engine,
        "tcp://127.0.0.1:5556",
        "worker",
        {"isolate_gpu_devices": True},
    )
    assert model.name == "unsloth/Qwen3.5-9B-UD-Q4_K_XL"
    assert "--isolate-gpu-devices" in command
    assert "True" not in command


def test_process_manager_helpers() -> None:
    assert process_manager_kind("local") == "local"
    assert worker_gpus("cpu") == 0
    assert worker_gpus("cuda") == 1
    assert safe_name("orchestra/Qwen 35B!") == "orchestra-Qwen-35B"


def test_forced_gflow_fails_fast_when_missing(monkeypatch) -> None:
    import pytest

    monkeypatch.setattr("orchestra.process_manager.shutil.which", lambda _: None)
    assert "gflowd" in missing_gflow_commands()
    with pytest.raises(AssertionError, match="gflow process manager requested"):
        process_manager_kind("gflow")


def test_gflow_poll_marks_missing_job_failed(monkeypatch, tmp_path) -> None:
    import subprocess

    def fake_run(*_, **__) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(
            args=["gjob", "show", "40839"],
            returncode=0,
            stdout="Error: Job 40839 not found\n",
        )

    monkeypatch.setattr("orchestra.process_manager.subprocess.run", fake_run)
    job = GflowJob(job_id="40839", script_path=tmp_path / "worker.sh")
    assert job.poll() == 1
    assert job.poll() == 1


def test_gflow_job_id_ignores_uuid_digits() -> None:
    output = "Submitted job gjob-1-orchestra-974a376d-df3a-4aa6-b0b4-02eb74d40839\n"
    assert gflow_job_id(output) == "1"
    assert gflow_job_id("Submitted batch job 42\n") == "42"
    assert gflow_job_id("  7       gjob-7-orchestra-abc123\n") == "7"


def test_gflow_queue_job_ids_selects_orchestra_jobs() -> None:
    output = "\n".join(
        [
            "JOBID NAME ST TIME NODES NODELIST(REASON)",
            "1 gjob-1-orchestra-abc PD - 1 (Resources)",
            "2 gjob-2-other PD - 1 (Resources)",
            "3 gjob-3-orchestra-def R 00:01 1 localhost",
        ]
    )
    assert gflow_queue_job_ids(output) == ["1", "3"]


def test_vram_is_advisory_not_a_hard_route_block(monkeypatch) -> None:
    from orchestra.hardware import Gpu, HardwareReport

    config = load_config()
    registry = load_registry()
    model = model_by_name(registry, "qwen3.6-35b")
    monkeypatch.setattr("orchestra.routing.model_status", lambda *_: "downloaded")
    monkeypatch.setattr("orchestra.routing.env_status", lambda *_: "ready")
    monkeypatch.setattr("orchestra.routing.compatible", lambda *_: True)
    hardware = HardwareReport(
        os="linux",
        machine="x86_64",
        python="3.12",
        uv=True,
        cuda=True,
        rocm=False,
        mlx=False,
        gpus=[Gpu(vendor="nvidia", name="small", total_mb=16000, free_mb=16000)],
    )
    candidate = candidates_for_task(config, registry, hardware, "text_generation")[0]
    direct = candidate_status(config, registry, hardware, model)
    assert direct.ready
    assert "advisory" in direct.reason
    assert candidate.ready


def test_idle_worker_is_evicted_under_queued_model_pressure() -> None:
    import time

    from broker_core import (
        BrokerState,
        Worker,
        WorkerPool,
        WorkerStatus,
        has_stopping_worker_for_key,
        idle_pressure_blocks_spawn,
    )

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self) -> None:
            pass

    class FakeSocket:
        def __init__(self) -> None:
            self.frames = []

        def send_multipart(self, frames) -> None:
            self.frames.append(frames)

    state = BrokerState()
    socket = FakeSocket()
    state.worker_map["old-worker"] = Worker(
        key="old-key",
        model_name="old-model",
        process=FakeProcess(),
        started_at=time.monotonic() - 100,
        status=WorkerStatus.IDLE,
        idle_since=time.monotonic() - 4,
    )
    state.worker_registry["old-key"] = WorkerPool(idle_workers={"old-worker"})

    assert idle_pressure_blocks_spawn(socket, state, "new-key")
    assert state.worker_map["old-worker"].status == WorkerStatus.STOPPING
    assert not state.worker_registry["old-key"].idle_workers
    assert socket.frames[0][0] == b"old-model-old-worker"
    assert has_stopping_worker_for_key(state, "old-key")
    assert not has_stopping_worker_for_key(state, "new-key")


def test_stopping_worker_blocks_same_key_respawn() -> None:
    import time

    from broker_core import BrokerState, Worker, WorkerStatus, has_stopping_worker_for_key

    class FakeProcess:
        def poll(self):
            return None

    state = BrokerState()
    state.worker_map["worker"] = Worker(
        key="same-key",
        model_name="model",
        process=FakeProcess(),
        started_at=time.monotonic(),
        status=WorkerStatus.STOPPING,
        stopping_since=time.monotonic(),
    )
    assert has_stopping_worker_for_key(state, "same-key")


def test_text_generation_routing_returns_candidates() -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    candidates = candidates_for_task(config, registry, hardware, "text_generation")
    assert candidates
    candidate = choose_model(config, registry, hardware, "text_generation")
    assert candidate.model.task == "text_generation"


def test_snapshot_contains_registry_state() -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    snapshot = snapshot_data(config, registry, hardware)
    assert snapshot["models"]
    assert snapshot["runtimes"]
    assert snapshot["engines"]
    assert snapshot["broker_address"] == config.broker_address


def test_dummy_model_is_ready_without_external_artifacts() -> None:
    registry = load_registry()
    model = registry.models["orchestra/dummy-echo"]
    runtime = runtime_for_model(registry, model)
    engine = engine_for_model(registry, model)
    command = build_worker_command(model, runtime, engine, "tcp://127.0.0.1:5556", "worker")
    assert env_status(runtime) == "ready"
    assert command[1].endswith("models/generic/dummy/worker.py")


def test_local_node_inventory_contains_dummy() -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    statuses = local_node_status(config, registry, hardware)
    dummy = [status for status in statuses if status.model_name == "orchestra/dummy-echo"]
    assert dummy
    assert dummy[0].compatible


def test_http_api_health() -> None:
    from orchestra.api import api_setup_plan, health, nodes, profile, schedule

    assert health() == {"status": "ok"}
    assert schedule()
    assert isinstance(nodes(), list)
    assert profile()["name"]
    assert api_setup_plan()


def test_scheduler_plan_contains_dummy() -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    plan = schedule_plan(config, registry, hardware, {"text_generation"})
    dummy = [item for item in plan if item.model_name == "orchestra/dummy-echo"]
    assert dummy
    assert dummy[0].action == "keep_warm"


def test_architecture_profile_selects_llm_runtime() -> None:
    hardware = detect_hardware()
    profile = architecture_profile(hardware)
    engine, runtime = default_llm_engine_runtime(hardware, "gguf")
    assert profile.name
    assert engine == "llama_cpp"
    assert runtime.startswith("llama_cpp_")


def test_setup_plan_contains_runtime_and_model_actions() -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    plan = setup_plan(config, registry, hardware)
    targets = {item.target for item in plan}
    assert "orchestra/dummy-echo" in targets
    assert "dummy_system" in targets


def test_broker_dummy_end_to_end() -> None:
    import json
    import os
    import subprocess
    import sys
    import time
    import uuid

    import zmq

    address = f"tcp://127.0.0.1:{free_port()}"
    env = os.environ | {
        "ORCHESTRA_BROKER_ADDRESS": address,
        "ORCHESTRA_BROKER_BIND_ADDRESS": address,
        "ORCHESTRA_PROCESS_MANAGER": "local",
    }
    process = subprocess.Popen(
        [sys.executable, "broker_core.py"],
        cwd=os.getcwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        socket = zmq.Context.instance().socket(zmq.REQ)
        socket.connect(address)
        payload = {
            "request_id": str(uuid.uuid4()),
            "model_name": "orchestra/dummy-echo",
            "prompt": "e2e",
            "config": {},
        }
        deadline = time.time() + 10
        response = None
        while time.time() < deadline:
            socket.send_multipart([json.dumps(payload).encode("utf-8")])
            if socket.poll(1000):
                response = json.loads(socket.recv_multipart()[-1].decode("utf-8"))
                break
            socket.close()
            socket = zmq.Context.instance().socket(zmq.REQ)
            socket.connect(address)
        assert response is not None
        assert response["type"] == "SUCCESS"
        assert response["answer"] == "echo: e2e"
    finally:
        process.terminate()
        process.wait(timeout=5)
