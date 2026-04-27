from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from orchestra.config import CONFIG_PATH, init_dirs, load_config
from orchestra.hardware import as_json, detect_hardware
from orchestra.job_store import tail_jobs
from orchestra.metrics import tail_events
from orchestra.model_store import (
    download_model,
    download_plan,
    model_file_path,
    model_status,
    remote_files,
    remove_model,
)
from orchestra.nodes import NodeSpec, load_node_specs, local_node_status, write_node_spec
from orchestra.profiles import architecture_profile, default_llm_engine_runtime, setup_plan
from orchestra.registry import engine_for_model, load_registry, model_by_name, runtime_for_model
from orchestra.routing import candidate_status, candidates_for_task, choose_model
from orchestra.runtime import (
    build_worker_command,
    compatible,
    env_path,
    env_status,
    setup_env,
    setup_env_commands,
)
from orchestra.scheduler import schedule_plan
from orchestra.settings import CONFIG_KEYS, config_data, set_config_value
from orchestra.snapshot import snapshot_data, write_snapshot

app = typer.Typer(no_args_is_help=True)
env_app = typer.Typer(no_args_is_help=True)
models_app = typer.Typer(no_args_is_help=True)
engines_app = typer.Typer(no_args_is_help=True)
broker_app = typer.Typer(no_args_is_help=True)
api_app = typer.Typer(no_args_is_help=True)
worker_app = typer.Typer(no_args_is_help=True)
route_app = typer.Typer(no_args_is_help=True)
request_app = typer.Typer(no_args_is_help=True)
snapshot_app = typer.Typer(no_args_is_help=True)
nodes_app = typer.Typer(no_args_is_help=True)
metrics_app = typer.Typer(no_args_is_help=True)
schedule_app = typer.Typer(no_args_is_help=True)
jobs_app = typer.Typer(no_args_is_help=True)
profile_app = typer.Typer(no_args_is_help=True)
setup_app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
app.add_typer(env_app, name="env")
app.add_typer(models_app, name="models")
app.add_typer(engines_app, name="engines")
app.add_typer(broker_app, name="broker")
app.add_typer(api_app, name="api")
app.add_typer(worker_app, name="worker")
app.add_typer(route_app, name="route")
app.add_typer(request_app, name="request")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(nodes_app, name="nodes")
app.add_typer(metrics_app, name="metrics")
app.add_typer(schedule_app, name="schedule")
app.add_typer(jobs_app, name="jobs")
app.add_typer(profile_app, name="profile")
app.add_typer(setup_app, name="setup")
app.add_typer(config_app, name="config")
console = Console()


def _relative(path: Path) -> str:
    return str(path.relative_to(Path.cwd()))


def _compatible_text(ok: bool) -> str:
    return "yes" if ok else "no"


def _engine_worker_dir(engine_name: str) -> Path:
    generic = Path.cwd() / "models" / "generic" / engine_name
    if generic.is_dir():
        return generic
    registry = load_registry()
    dirs = sorted(
        {engine.worker_dir for engine in registry.engines.values() if engine.name == engine_name}
    )
    assert len(dirs) == 1, f"Engine '{engine_name}' needs an explicit generic adapter"
    return dirs[0]


def _model_yaml_path(engine_name: str) -> Path:
    return _engine_worker_dir(engine_name) / "model.yaml"


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {"family": path.parent.relative_to(Path.cwd() / "models").as_posix(), "models": []}
    return yaml.safe_load(path.read_text()) or {}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        answer = input(f"{path} exists. Update it? [y/N]: ").strip().lower()
        assert answer in {"y", "yes"}, f"Refusing to update {path}"
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _parse_worker_args(worker_arg: list[str] | None) -> dict:
    values = {}
    for item in worker_arg or []:
        if "=" in item:
            key, value = item.split("=", 1)
            values[key.replace("-", "_")] = value
        else:
            values[item.replace("-", "_")] = True
    return values


def _worker_arg_items(args: dict) -> list[str]:
    items = []
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            items.append(flag)
        else:
            items += [flag, str(value)]
    return items


@app.command()
def init(dry_run: bool = False) -> None:
    config = load_config()
    if dry_run:
        table = Table(title="Init Dry Run")
        table.add_column("Path")
        table.add_column("Action")
        paths = [
            config.root,
            config.model_cache,
            config.engine_cache,
            config.env_cache,
            config.logs,
        ]
        for path in paths:
            table.add_row(str(path), "create if missing")
        table.add_row(str(CONFIG_PATH), "write default config if missing")
        console.print(table)
        return

    init_dirs(config)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            "paths:\n"
            f"  root: {config.root}\n"
            f"  model_cache: {config.model_cache}\n"
            f"  engine_cache: {config.engine_cache}\n"
            f"  env_cache: {config.env_cache}\n"
            f"  logs: {config.logs}\n"
            "broker:\n"
            f"  address: {config.broker_address}\n"
        )
    console.print(f"Initialized ORCHESTRA at {config.root}")


@app.command()
def doctor(json: bool = False, fix_plan: bool = False) -> None:
    import shutil

    report = detect_hardware()
    if json:
        console.print(as_json(report))
        return

    table = Table(title="ORCHESTRA Doctor")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("OS", report.os)
    table.add_row("Machine", report.machine)
    table.add_row("Python", report.python)
    table.add_row("uv", "yes" if report.uv else "missing")
    table.add_row("CUDA", "yes" if report.cuda else "no")
    table.add_row("ROCm/HIP", "yes" if report.rocm else "no")
    table.add_row("MLX/MPS", "yes" if report.mlx else "no")
    for tool in ["curl", "sudo", "unshare", "wget", "tmux"]:
        table.add_row(tool, "yes" if shutil.which(tool) else "missing")
    for index, gpu in enumerate(report.gpus):
        table.add_row(f"GPU {index}", f"{gpu.name} {gpu.free_mb}/{gpu.total_mb} MB free")
    console.print(table)

    if fix_plan and not report.uv:
        console.print("Install uv first: https://docs.astral.sh/uv/getting-started/installation/")


@app.command("dry-run")
def dry_run() -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    console.print("[bold]ORCHESTRA dry run:[/bold] no files, envs, downloads, sockets,")
    console.print("or workers are created.")
    doctor()

    summary = Table(title="System Plan")
    summary.add_column("Area")
    summary.add_column("Status")
    summary.add_column("Details")
    summary.add_row("Config", "ok" if CONFIG_PATH.exists() else "missing", str(CONFIG_PATH))
    summary.add_row("Models", str(len(registry.models)), "registry entries")
    summary.add_row("Runtimes", str(len(registry.runtimes)), "runtime entries")
    summary.add_row("Engines", str(len(registry.engines)), "engine entries")
    console.print(summary)

    table = Table(title="Default Model Execution Plan")
    table.add_column("Model")
    table.add_column("Engine")
    table.add_column("Runtime")
    table.add_column("Artifact")
    table.add_column("Env")
    table.add_column("Worker")
    table.add_column("Result")
    for model in registry.models.values():
        runtime = runtime_for_model(registry, model)
        engine = engine_for_model(registry, model)
        artifact_state = model_status(config, model)
        runtime_state = env_status(runtime)
        worker_path = model.worker_dir / engine.entrypoint
        free_vram = max((gpu.free_mb for gpu in hardware.gpus), default=0)
        result = "ready"
        if artifact_state != "downloaded":
            result = "blocked: model missing"
        elif runtime_state != "ready":
            result = "blocked: env missing"
        elif not compatible(runtime, hardware):
            result = "blocked: incompatible runtime"
        elif hardware.gpus and free_vram < model.min_vram_mb:
            result = "blocked: insufficient VRAM"
        elif not worker_path.is_file():
            result = "blocked: worker missing"
        table.add_row(
            model.name,
            model.default_engine,
            model.default_runtime,
            artifact_state,
            runtime_state,
            "ok" if worker_path.is_file() else "missing",
            result,
        )
    console.print(table)

    env_plan = Table(title="Environment Setup Commands")
    env_plan.add_column("Runtime")
    env_plan.add_column("Worker")
    env_plan.add_column("Command")
    for runtime in registry.runtimes.values():
        for command in setup_env_commands(runtime):
            env_plan.add_row(runtime.name, _relative(runtime.worker_dir), " ".join(command))
    console.print(env_plan)

    download_table = Table(title="Model Download Plan")
    download_table.add_column("Model")
    download_table.add_column("Source")
    download_table.add_column("Plan")
    for model in registry.models.values():
        download_table.add_row(model.name, model.source, download_plan(config, model))
    console.print(download_table)


@app.command()
def validate() -> None:
    registry = load_registry()
    for model in registry.models.values():
        runtime_for_model(registry, model)
        engine_for_model(registry, model)
    console.print(
        f"Registry valid: {len(registry.models)} models, "
        f"{len(registry.runtimes)} runtimes, {len(registry.engines)} engines"
    )


@config_app.command("show")
def config_show() -> None:
    config = load_config()
    table = Table(title="ORCHESTRA Config")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("paths.root", str(config.root))
    table.add_row("paths.model_cache", str(config.model_cache))
    table.add_row("paths.engine_cache", str(config.engine_cache))
    table.add_row("paths.env_cache", str(config.env_cache))
    table.add_row("paths.logs", str(config.logs))
    table.add_row("broker.address", config.broker_address)
    for node in config_data().get("nodes", []):
        table.add_row(f"nodes.{node['name']}", node["address"])
    console.print(table)


@config_app.command("set")
def config_set(key: str, value: str, force: bool = False) -> None:
    assert key in CONFIG_KEYS, f"Allowed keys: {', '.join(sorted(CONFIG_KEYS))}"
    set_config_value(key, value, force=force)
    console.print(f"Set {key}={value}")


@env_app.command("list")
def env_list() -> None:
    registry = load_registry()
    hardware = detect_hardware()
    table = Table(title="Runtime Environments")
    table.add_column("Runtime")
    table.add_column("Worker")
    table.add_column("Accelerator")
    table.add_column("Status")
    table.add_column("Compatible")
    for runtime in registry.runtimes.values():
        table.add_row(
            runtime.name,
            _relative(runtime.worker_dir),
            runtime.accelerator,
            env_status(runtime),
            _compatible_text(compatible(runtime, hardware)),
        )
    console.print(table)


@env_app.command("status")
def env_status_command() -> None:
    env_list()


@env_app.command("setup")
def env_setup(runtime_name: str, force: bool = False, dry_run: bool = False) -> None:
    registry = load_registry()
    hardware = detect_hardware()
    runtimes = [runtime for runtime in registry.runtimes.values() if runtime.name == runtime_name]
    assert runtimes, runtime_name
    for runtime in runtimes:
        if dry_run:
            table = Table(title=f"Env Setup Dry Run: {runtime.name}")
            table.add_column("Field")
            table.add_column("Value")
            table.add_row("Worker", _relative(runtime.worker_dir))
            table.add_row("Env", str(env_path(runtime)))
            table.add_row("Status", env_status(runtime))
            table.add_row("Compatible", _compatible_text(compatible(runtime, hardware)))
            for command in setup_env_commands(runtime):
                table.add_row("Would run", " ".join(command))
            console.print(table)
            continue
        assert compatible(runtime, hardware), (
            f"Runtime is not compatible with this machine: {runtime.name}"
        )
        setup_env(runtime, force=force)
        console.print(f"Ready: {runtime.worker_dir / '.venv'}")


@models_app.command("list")
def models_list() -> None:
    registry = load_registry()
    table = Table(title="Model Catalog")
    table.add_column("Model")
    table.add_column("Task")
    table.add_column("Aliases")
    table.add_column("Engine")
    table.add_column("Runtime")
    table.add_column("VRAM")
    for model in registry.models.values():
        table.add_row(
            model.name,
            model.task,
            ", ".join(model.aliases),
            model.default_engine,
            model.default_runtime,
            f"{model.min_vram_mb} MB",
        )
    console.print(table)


@models_app.command("add")
def models_add(
    name: str,
    artifact: str,
    engine: str,
    runtime: str,
    format: str,
    artifact_file: str | None = None,
    task: str = "text_generation",
    source: str = "huggingface",
    min_vram_mb: int = 0,
    recommended_vram_mb: int = 0,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    path = _model_yaml_path(engine)
    data = _read_yaml(path)
    models = list(data.get("models", []))
    existing = [item for item in models if item["name"] == name]
    if existing and not force:
        raise AssertionError(f"Model already exists: {name}")

    registry = load_registry()
    worker_dir = _engine_worker_dir(engine)
    runtime_key = f"{worker_dir}:{runtime}"
    engine_key = f"{worker_dir}:{engine}"
    assert runtime_key in registry.runtimes, f"Unknown runtime for {engine}: {runtime}"
    assert engine_key in registry.engines, f"Unknown engine adapter: {engine}"

    new_model = {
        "name": name,
        "source": source,
        "artifact": artifact,
        "artifact_file": artifact_file,
        "format": format,
        "task": task,
        "min_vram_mb": min_vram_mb,
        "recommended_vram_mb": recommended_vram_mb or min_vram_mb,
        "default_engine": engine,
        "supported_engines": [engine],
        "default_runtime": runtime,
    }
    if dry_run:
        table = Table(title="Model Add Dry Run")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Manifest", str(path))
        table.add_row("Name", name)
        table.add_row("Artifact", artifact)
        table.add_row("Artifact file", artifact_file or "")
        table.add_row("Format", format)
        table.add_row("Engine", engine)
        table.add_row("Runtime", runtime)
        table.add_row("Task", task)
        table.add_row("Source", source)
        console.print(table)
        return

    if existing:
        answer = input(f"Replace existing model {name}? [y/N]: ").strip().lower()
        assert answer in {"y", "yes"}, f"Refusing to replace {name}"
        models = [item for item in models if item["name"] != name]
    models.append(new_model)
    data["models"] = models
    _write_yaml(path, data)
    console.print(f"Added {name} to {path}")


@models_app.command("add-llm")
def models_add_llm(
    name: str,
    artifact: str,
    format: str = "gguf",
    artifact_file: str | None = None,
    engine: str | None = None,
    runtime: str | None = None,
    min_vram_mb: int = 0,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    hardware = detect_hardware()
    selected_engine, selected_runtime = default_llm_engine_runtime(hardware, format)
    models_add(
        name=name,
        artifact=artifact,
        engine=engine or selected_engine,
        runtime=runtime or selected_runtime,
        format=format,
        artifact_file=artifact_file,
        task="text_generation",
        source="huggingface",
        min_vram_mb=min_vram_mb,
        recommended_vram_mb=min_vram_mb,
        dry_run=dry_run,
        force=force,
    )


@models_app.command("status")
def models_status() -> None:
    registry = load_registry()
    config = load_config()
    hardware = detect_hardware()
    table = Table(title="Model Status")
    table.add_column("Model")
    table.add_column("Engine")
    table.add_column("Runtime")
    table.add_column("Model")
    table.add_column("Env")
    table.add_column("Compatible")
    table.add_column("Location")
    for model in registry.models.values():
        runtime = runtime_for_model(registry, model)
        free_vram = max((gpu.free_mb for gpu in hardware.gpus), default=0)
        enough_vram = not hardware.gpus or free_vram >= model.min_vram_mb
        table.add_row(
            model.name,
            model.default_engine,
            model.default_runtime,
            model_status(config, model),
            env_status(runtime),
            _compatible_text(compatible(runtime, hardware) and enough_vram),
            str(model_file_path(config, model)),
        )
    console.print(table)


@models_app.command("download")
def models_download(model_name: str, force: bool = False, dry_run: bool = False) -> None:
    registry = load_registry()
    config = load_config()
    model = model_by_name(registry, model_name)
    if dry_run:
        table = Table(title=f"Model Download Dry Run: {model.name}")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Status", model_status(config, model))
        table.add_row("Source", model.source)
        table.add_row("Artifact", model.artifact)
        table.add_row("Would do", download_plan(config, model))
        console.print(table)
        return
    path = download_model(config, model, force=force)
    console.print(f"Downloaded {model.name} to {path}")


@models_app.command("files")
def models_files(repo_id: str, pattern: str = "*.gguf", limit: int = 50) -> None:
    table = Table(title=f"Remote Files: {repo_id}")
    table.add_column("File")
    for path in remote_files(repo_id, pattern)[:limit]:
        table.add_row(path)
    console.print(table)


@models_app.command("remove")
def models_remove(model_name: str) -> None:
    registry = load_registry()
    config = load_config()
    remove_model(config, model_by_name(registry, model_name))


@engines_app.command("list")
def engines_list() -> None:
    registry = load_registry()
    table = Table(title="Inference Engines")
    table.add_column("Engine")
    table.add_column("Worker")
    table.add_column("Install")
    table.add_column("Entrypoint")
    for engine in registry.engines.values():
        table.add_row(engine.name, _relative(engine.worker_dir), engine.install, engine.entrypoint)
    console.print(table)


@engines_app.command("setup")
def engines_setup(engine_name: str, runtime_name: str, dry_run: bool = False) -> None:
    registry = load_registry()
    hardware = detect_hardware()
    engines = [engine for engine in registry.engines.values() if engine.name == engine_name]
    assert engines, engine_name
    worker_dirs = sorted({engine.worker_dir for engine in engines})
    runtimes = [
        runtime
        for runtime in registry.runtimes.values()
        if runtime.worker_dir in worker_dirs and runtime.name == runtime_name
    ]
    assert runtimes, runtime_name
    runtime = runtimes[0]

    if dry_run:
        table = Table(title=f"Engine Setup Dry Run: {engine_name}")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Runtime", runtime.name)
        table.add_row("Worker", _relative(runtime.worker_dir))
        table.add_row("Compatible", _compatible_text(compatible(runtime, hardware)))
        table.add_row("Env", str(env_path(runtime)))
        table.add_row("Env status", env_status(runtime))
        for command in setup_env_commands(runtime):
            table.add_row("Would run", " ".join(command))
        console.print(table)
        return

    assert compatible(runtime, hardware), f"Runtime is not compatible: {runtime.name}"
    setup_env(runtime)
    console.print(f"Ready: {engine_name} {runtime.name}")


@route_app.command("task")
def route_task(task: str, engine: str | None = None, runtime: str | None = None) -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    table = Table(title=f"Route Candidates: {task}")
    table.add_column("Model")
    table.add_column("Engine")
    table.add_column("Runtime")
    table.add_column("VRAM")
    table.add_column("Ready")
    table.add_column("Reason")
    for candidate in candidates_for_task(config, registry, hardware, task, engine, runtime):
        model = candidate.model
        table.add_row(
            model.name,
            model.default_engine,
            model.default_runtime,
            f"{model.min_vram_mb} MB",
            _compatible_text(candidate.ready),
            candidate.reason,
        )
    console.print(table)


def _request_payload(
    prompt: str,
    model_name: str | None,
    task: str,
    engine: str | None,
    runtime: str | None,
    max_tokens: int,
    temperature: float,
    worker_arg: list[str] | None = None,
) -> tuple[dict, str]:
    import uuid

    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    if model_name is None:
        candidate = choose_model(config, registry, hardware, task, engine, runtime)
        model = candidate.model
    else:
        model = model_by_name(registry, model_name)
        candidate = candidate_status(config, registry, hardware, model)
    payload = {
        "request_id": str(uuid.uuid4()),
        "model_name": model.name,
        "engine": engine or model.default_engine,
        "runtime": runtime or model.default_runtime,
        "prompt": prompt,
        "config": {"max_tokens": max_tokens, "temperature": temperature},
    }
    args_per_model = _parse_worker_args(worker_arg)
    if args_per_model:
        payload["args_per_model"] = args_per_model
    return payload, candidate.reason


def _task_payload(
    model_name: str | None,
    task: str,
    fields: dict,
    worker_arg: list[str] | None = None,
) -> tuple[dict, str]:
    import uuid

    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    if model_name is None:
        candidate = choose_model(config, registry, hardware, task)
        model = candidate.model
    else:
        model = model_by_name(registry, model_name)
        candidate = candidate_status(config, registry, hardware, model)
    payload = {
        "request_id": str(uuid.uuid4()),
        "model_name": model.name,
        "engine": model.default_engine,
        "runtime": model.default_runtime,
    } | fields
    args_per_model = _parse_worker_args(worker_arg)
    if args_per_model:
        payload["args_per_model"] = args_per_model
    return payload, candidate.reason


def _send_or_print(payload: dict, reason: str, timeout_ms: int, dry_run: bool) -> None:
    import json

    import zmq

    config = load_config()
    if dry_run:
        console.print(f"Would connect to {config.broker_address}")
        console.print(json.dumps(payload, indent=2))
        console.print(f"Route status: {reason}")
        return
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect(config.broker_address)
    socket.send_multipart([json.dumps(payload).encode("utf-8")])
    assert socket.poll(timeout_ms), f"Broker response timed out after {timeout_ms} ms"
    console.print(socket.recv_multipart()[-1].decode("utf-8"))


@request_app.command("build")
def request_build(
    prompt: str,
    model_name: str | None = None,
    task: str = "text_generation",
    engine: str | None = None,
    runtime: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    worker_arg: list[str] | None = None,
) -> None:
    import json

    payload, reason = _request_payload(
        prompt, model_name, task, engine, runtime, max_tokens, temperature, worker_arg
    )
    console.print(json.dumps(payload, indent=2))
    console.print(f"Route status: {reason}")


@request_app.command("send")
def request_send(
    prompt: str,
    model_name: str | None = None,
    task: str = "text_generation",
    engine: str | None = None,
    runtime: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout_ms: int = 60000,
    dry_run: bool = False,
    worker_arg: list[str] | None = None,
) -> None:
    payload, reason = _request_payload(
        prompt, model_name, task, engine, runtime, max_tokens, temperature, worker_arg
    )
    _send_or_print(payload, reason, timeout_ms, dry_run)


@request_app.command("translate")
def request_translate(
    text: str,
    source: str,
    target: str,
    model_name: str | None = "google/translategemma-4b-it",
    timeout_ms: int = 600000,
    dry_run: bool = False,
    worker_arg: list[str] | None = None,
) -> None:
    payload, reason = _task_payload(
        model_name,
        "translation",
        {"text": text, "source": source, "target": target},
        worker_arg,
    )
    _send_or_print(payload, reason, timeout_ms, dry_run)


@request_app.command("embed")
def request_embed(
    text: str,
    model_name: str | None = "BAAI/bge-m3-vllm-rocm",
    timeout_ms: int = 60000,
    dry_run: bool = False,
    worker_arg: list[str] | None = None,
) -> None:
    payload, reason = _task_payload(model_name, "embedding", {"input": text}, worker_arg)
    _send_or_print(payload, reason, timeout_ms, dry_run)


@request_app.command("transcribe")
def request_transcribe(
    audio_path: str,
    model_name: str | None = "Systran/faster-whisper-large-v3",
    language: str | None = None,
    timeout_ms: int = 600000,
    dry_run: bool = False,
    worker_arg: list[str] | None = None,
) -> None:
    fields = {"audio_path": audio_path}
    if language is not None:
        fields["language"] = language
    payload, reason = _task_payload(model_name, "speech_to_text", fields, worker_arg)
    _send_or_print(payload, reason, timeout_ms, dry_run)


@snapshot_app.command("export")
def snapshot_export(path: Path = Path("orchestra.lock"), dry_run: bool = False) -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    data = snapshot_data(config, registry, hardware)
    if dry_run:
        console.print(f"Would write snapshot to {path}")
        console.print(f"Models: {len(data['models'])}")
        console.print(f"Runtimes: {len(data['runtimes'])}")
        console.print(f"Engines: {len(data['engines'])}")
        return
    write_snapshot(path, data)
    console.print(f"Wrote snapshot to {path}")


@nodes_app.command("local")
def nodes_local() -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    table = Table(title="Local Node Inventory")
    table.add_column("Model")
    table.add_column("Engine")
    table.add_column("Runtime")
    table.add_column("Model")
    table.add_column("Env")
    table.add_column("Compatible")
    for status in local_node_status(config, registry, hardware):
        table.add_row(
            status.model_name,
            status.engine,
            status.runtime,
            status.model_status,
            status.env_status,
            _compatible_text(status.compatible),
        )
    console.print(table)


@nodes_app.command("list")
def nodes_list() -> None:
    table = Table(title="Configured Nodes")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Address")
    table.add_column("Labels")
    for node in load_node_specs():
        labels = " ".join(f"{key}={value}" for key, value in node.labels.items())
        table.add_row(node.name, node.role, node.address, labels)
    console.print(table)


@nodes_app.command("add")
def nodes_add(
    name: str,
    address: str,
    role: str = "broker",
    label: list[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    labels = dict(item.split("=", 1) for item in label or [])
    spec = NodeSpec(name=name, address=address, role=role, labels=labels)
    if dry_run:
        table = Table(title="Node Add Dry Run")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Name", spec.name)
        table.add_row("Role", spec.role)
        table.add_row("Address", spec.address)
        table.add_row("Labels", " ".join(f"{key}={value}" for key, value in labels.items()))
        console.print(table)
        return
    write_node_spec(spec, force=force)
    console.print(f"Configured node {name}")


@metrics_app.command("tail")
def metrics_tail(limit: int = 20) -> None:
    table = Table(title="ORCHESTRA Events")
    table.add_column("Event")
    table.add_column("Fields")
    for event in tail_events(limit):
        event_name = str(event.pop("event"))
        event.pop("time_ns", None)
        table.add_row(event_name, " ".join(f"{key}={value}" for key, value in event.items()))
    console.print(table)


@jobs_app.command("tail")
def jobs_tail(limit: int = 20) -> None:
    table = Table(title="ORCHESTRA Jobs")
    table.add_column("Event")
    table.add_column("Request")
    table.add_column("Fields")
    for event in tail_jobs(limit):
        event_name = str(event.pop("event"))
        request_id = str(event.pop("request_id"))
        event.pop("time_ns", None)
        fields = " ".join(f"{key}={value}" for key, value in event.items())
        table.add_row(event_name, request_id, fields)
    console.print(table)


@profile_app.command("show")
def profile_show() -> None:
    hardware = detect_hardware()
    profile = architecture_profile(hardware)
    table = Table(title="Architecture Profile")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Profile", profile.name)
    table.add_row("Accelerator", profile.accelerator)
    table.add_row("Text engine", profile.preferred_text_engine)
    table.add_row("GGUF runtime", profile.preferred_gguf_runtime)
    table.add_row("HF runtime", profile.preferred_hf_runtime)
    console.print(table)


@setup_app.command("plan")
def setup_plan_command() -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    table = Table(title="Setup Plan")
    table.add_column("Target")
    table.add_column("Status")
    table.add_column("Action")
    table.add_column("Command/Artifact")
    for action in setup_plan(config, registry, hardware):
        table.add_row(action.target, action.status, action.action, action.command)
    console.print(table)


@schedule_app.command("plan")
def schedule_plan_command(warm_task: list[str] | None = None) -> None:
    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    table = Table(title="Scheduler Plan")
    table.add_column("Model")
    table.add_column("Action")
    table.add_column("Reason")
    table.add_column("VRAM")
    for decision in schedule_plan(config, registry, hardware, set(warm_task or [])):
        table.add_row(
            decision.model_name,
            decision.action,
            decision.reason,
            f"{decision.min_vram_mb} MB",
        )
    console.print(table)


@app.command()
def benchmark(model_name: str, dry_run: bool = False) -> None:
    import time

    config = load_config()
    registry = load_registry()
    hardware = detect_hardware()
    model = model_by_name(registry, model_name)
    if dry_run:
        console.print(f"Would benchmark cold start, warm latency, throughput, VRAM: {model.name}")
        return
    start = time.perf_counter()
    candidate = candidate_status(config, registry, hardware, model)
    elapsed = time.perf_counter() - start
    table = Table(title=f"Benchmark: {model.name}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("validation_seconds", f"{elapsed:.6f}")
    table.add_row("ready", _compatible_text(candidate.ready))
    table.add_row("reason", candidate.reason)
    console.print(table)


@broker_app.command("start")
def broker_start(dry_run: bool = False) -> None:
    if dry_run:
        config = load_config()
        registry = load_registry()
        console.print(f"Would bind broker to {config.broker_address}")
        console.print(f"Would load {len(registry.models)} models from registry")
        return
    from broker_core import main

    main()


@api_app.command("start")
def api_start(host: str = "127.0.0.1", port: int = 8000, dry_run: bool = False) -> None:
    if dry_run:
        console.print(f"Would serve ORCHESTRA API at http://{host}:{port}")
        return
    import uvicorn

    uvicorn.run("orchestra.api:app", host=host, port=port)


@worker_app.command("start")
def worker_start(
    model_name: str,
    dry_run: bool = False,
    worker_arg: list[str] | None = None,
) -> None:
    import subprocess
    import sys
    import uuid

    config = load_config()
    registry = load_registry()
    model = model_by_name(registry, model_name)
    runtime = runtime_for_model(registry, model)
    engine = engine_for_model(registry, model)
    worker_id = str(uuid.uuid4())
    args_per_model = _parse_worker_args(worker_arg)
    if dry_run:
        python = (
            Path(sys.executable)
            if runtime.python == "system"
            else env_path(runtime) / "bin" / "python"
        )
        worker = model.worker_dir / engine.entrypoint
        command = [
            str(python),
            str(worker),
            "--model-id",
            model.name,
            "--router-connect",
            config.broker_address,
            "--worker-id",
            worker_id,
            *model.default_args,
            *_worker_arg_items(args_per_model),
        ]
        console.print("Would run worker command:")
        console.print(" ".join(command))
        console.print(f"Env status: {env_status(runtime)}")
        console.print(f"Worker exists: {_compatible_text(worker.is_file())}")
        return
    command = build_worker_command(
        model,
        runtime,
        engine,
        config.broker_address,
        worker_id,
        args_per_model,
    )
    subprocess.run(command, check=True)
