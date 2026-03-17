# orchestra

Minimal broker-worker scaffold for serving models with very little orchestration logic.

## Status

🚧 This repo is under construction 🚧

Current tested baseline:

- `2x NVIDIA RTX 5090`
- `CUDA 12.8`

The core flow exists, but startup and worker packaging are still being cleaned up.

## What It Does

`orchestra` is a small ZeroMQ-based dispatcher:

- `broker_core.py` accepts client requests, keeps per-model queues, and spawns workers on demand.
- Workers live under `models/<lab>/<family>/worker.py`.
- Model availability is declared in `models/*/config.yaml`.
- Shared transport and image helpers live in `utils/`.

The intended shape is simple:

1. A client sends a request with a `model_name`.
2. The broker queues the job.
3. If no worker is alive for that model, the broker starts one.
4. The worker runs the job and sends the result back through the broker.
5. Idle workers exit on their own.

## Repo Layout

```text
broker_core.py              Main broker loop
models/*/config.yaml        Model registry and worker mapping
models/*/*/worker.py        Worker entrypoints
utils/transport.py          ZeroMQ worker connection helper
utils/image_io.py           Image decoding helpers
testsuite/                  Early test scripts
```

Current model families in the repo include `OpenGVLab/InternVL` and `facebook/sam3`.

## Examples

`recipes/` is the place for usage examples and runnable patterns. Much more is coming.

## Setup

Install the root environment:

```bash
uv sync
```

Optional dev tools:

```bash
uv sync --group dev
uv run pre-commit install
```

## Running

Start the broker from the repository root:

```bash
uv run python broker_core.py
```

Important: the broker currently expects each worker folder declared in `config.yaml` to have its own `.venv` and `worker.py`. If that environment is missing, worker spawning will fail fast.

Just run 
```bash
start_orchestra.sh
```


## 