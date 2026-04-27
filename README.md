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

## Docs

Docs are deployed through GitHub Pages with `.github/workflows/docs.yml`.

One-time GitHub setup:

1. Push the docs workflow and `docs/` content to `main`.
2. Open `Settings -> Pages`.
3. Set `Build and deployment -> Source` to `GitHub Actions`.

Published URL:

```text
https://sebo-the-tramp.github.io/orchestra/
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

## ORCHESTRA CLI

The new control surface is the `orchestra` CLI:

```bash
uv run orchestra doctor
uv run orchestra config show
uv run orchestra env list
uv run orchestra models list
uv run orchestra models status
uv run orchestra engines list
uv run orchestra broker start
```

Use explicit commands for mutating operations:

```bash
uv run orchestra env setup <runtime>
uv run orchestra models download <model-name>
uv run orchestra models remove <model-name>
```

The CLI reads `orchestra.yaml` plus per-worker `model.yaml` and `runtime.yaml` manifests.
It reports missing CUDA/HIP/MLX support, model cache status, runtime env status, and broker readiness without installing system-level components automatically.

Change default paths or broker address without editing YAML manually:

```bash
uv run orchestra config set paths.model_cache /mnt/models --force
uv run orchestra config set broker.address tcp://10.10.151.14:5556 --force
```

### Full Dry Run

On machines that cannot run the target CUDA workers, validate the full control plane without mutating state:

```bash
uv run orchestra dry-run
uv run orchestra init --dry-run
uv run orchestra env setup cuda128_torch28 --dry-run
uv run orchestra models download facebook/sam3 --dry-run
uv run orchestra broker start --dry-run
uv run orchestra worker start facebook/sam3 --dry-run
```

Dry runs do not create folders, virtualenvs, sockets, model downloads, or worker processes. They print the exact commands and blockers that will matter on the target machine.

### Adding Generic Models

New text models should not require a custom worker when an engine adapter already exists.
The generic shape is:

```text
model = artifact + format + engine + runtime + task
```

Example: Qwen GGUF through `llama.cpp` with CUDA:

```bash
uv run orchestra models add \
  Qwen/Qwen2.5-7B-Instruct-GGUF \
  unsloth/Qwen2.5-7B-Instruct-GGUF \
  llama_cpp \
  llama_cpp_cuda \
  gguf \
  --min-vram-mb 8000

uv run orchestra engines setup llama_cpp llama_cpp_cuda --dry-run
uv run orchestra models download Qwen/Qwen2.5-7B-Instruct-GGUF --dry-run
uv run orchestra worker start Qwen/Qwen2.5-7B-Instruct-GGUF --dry-run
```

Or let ORCHESTRA pick the generic engine/runtime from the detected architecture:

```bash
uv run orchestra profile show
uv run orchestra models add-llm \
  Qwen/Qwen2.5-7B-Instruct-Q4_K_M \
  unsloth/Qwen2.5-7B-Instruct-GGUF \
  --format gguf \
  --artifact-file qwen2.5-7b-instruct-q4_k_m.gguf \
  --min-vram-mb 8000 \
  --dry-run
```

Generic engine adapters currently exist for:

| Engine | Runtimes | Model Format |
| --- | --- | --- |
| `llama_cpp` | `llama_cpp_cuda`, `llama_cpp_metal`, `llama_cpp_hip`, `llama_cpp_cpu` | `gguf` |
| `llama_server` | `llama_server_system`, `llama_server_cuda`, `llama_server_hip` | external `llama-server` GGUF |
| `vllm` | `vllm_cuda` | Hugging Face model repos |
| `vllm_serve` | `vllm_serve_rocm`, `vllm_serve_cuda` | OpenAI-compatible vLLM server |
| `transformers` | `transformers_cuda`, `transformers_cpu` | Hugging Face model repos |

Pre-registered local lab models include:

| Model | Engine | Default Local Path |
| --- | --- | --- |
| `unsloth/Qwen3.6-35B-A3B-UD-Q4_K_S` | `llama_server` | `/home/kilolab/big-storage/llms/models/unsloth/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` |
| `unsloth/Qwen3.5-9B-UD-Q4_K_XL` | `llama_server` | `/home/kilolab/big-storage/llms/models/unsloth/Qwen3.5-9B-UD-Q4_K_XL.gguf` |
| `unsloth/Qwen3.6-27B-Q4_1` | `llama_server` | `/home/kilolab/big-storage/llms/models/unsloth/Qwen3.6-27B-Q4_1.gguf` |
| `bartowski/Qwen_Qwen3-30B-A3B-Q4_K_M` | `llama_server` | `/home/kilolab/big-storage/llms/models/qwen3/Qwen_Qwen3-30B-A3B-Q4_K_M.gguf` |
| `Qwen/Qwen3-8B-vllm-rocm` | `vllm_serve` | ORCHESTRA model cache |
| `BAAI/bge-m3-vllm-rocm` | `vllm_serve` | ORCHESTRA model cache |
| `google/translategemma-4b-it` | `translategemma` | `/home/kilolab/big-storage/llms/models/qwen3/translategemma-4b-it` |
| `openai/whisper-large-v3-transformers` | `whisper_transformers` | ORCHESTRA model cache |
| `Systran/faster-whisper-large-v3` | `faster_whisper` | ORCHESTRA model cache |

Provider-specific workers still exist for special model families like SAM3, DINOv3, InternVL, and LTX.

For GGUF repos with multiple quantizations, pin the exact file:

```bash
uv run orchestra models files unsloth/Qwen2.5-7B-Instruct-GGUF --pattern "*.gguf"

uv run orchestra models add \
  Qwen/Qwen2.5-7B-Instruct-Q4_K_M \
  unsloth/Qwen2.5-7B-Instruct-GGUF \
  llama_cpp \
  llama_cpp_cuda \
  gguf \
  --artifact-file qwen2.5-7b-instruct-q4_k_m.gguf \
  --min-vram-mb 8000 \
  --dry-run
```

### Routing, Scheduler, Jobs, Metrics

Dry-run a client request without contacting the broker:

```bash
uv run orchestra request send "ping" --model-name orchestra/dummy-echo --dry-run
uv run orchestra request translate "ciao mondo" it en --dry-run
uv run orchestra request embed "test embedding" --dry-run
uv run orchestra request transcribe /path/to/audio.wav --language it --dry-run
```

Use model aliases for shorter commands:

```bash
uv run orchestra worker start qwen3.5-9b --dry-run
uv run orchestra request translate "ciao mondo" it en --model-name translategemma --dry-run
uv run orchestra request embed "test embedding" --model-name bge-m3 --dry-run
```

GPU device isolation is explicit. It masks NVIDIA device nodes inside a mount namespace for the spawned vLLM server, matching the manual `sudo unshare -m --fork` pattern used on ROCm machines:

```bash
uv run orchestra request send "ping" \
  --model-name qwen3-8b-vllm \
  --worker-arg isolate-gpu-devices \
  --dry-run

uv run orchestra worker start qwen3-8b-vllm \
  --worker-arg isolate-gpu-devices \
  --dry-run
```

The isolation path requires `sudo`, `unshare`, and `curl`; `orchestra doctor` reports these explicitly. Server-style workers wait for their local `/health` endpoint before advertising HEARTBEAT to the broker.

Inspect task routing and local node readiness:

```bash
uv run orchestra route task text_generation
uv run orchestra nodes local
uv run orchestra nodes list
uv run orchestra schedule plan --warm-task text_generation
```

Register a remote broker/router explicitly:

```bash
uv run orchestra nodes add lab-a tcp://10.10.151.14:5556 --role broker --label gpu=4090 --dry-run
```

Export a reproducibility snapshot and inspect broker/job events:

```bash
uv run orchestra snapshot export --dry-run
uv run orchestra setup plan
uv run orchestra metrics tail
uv run orchestra jobs tail
```

`orchestra/dummy-echo` is a builtin CPU model used to validate routing, worker command generation, and request construction without external model downloads.

### HTTP API

The broker remains the core runtime. A small FastAPI surface can be started for HTTP clients:

```bash
uv run orchestra api start --host 0.0.0.0 --port 8000 --dry-run
uv run orchestra api start --host 0.0.0.0 --port 8000
```

Endpoint shape:

```bash
curl -X POST http://localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"model_name":"orchestra/dummy-echo","prompt":"ping","config":{}}'
```

Operational introspection endpoints:

```bash
curl http://localhost:8000/schedule
curl http://localhost:8000/profile
curl http://localhost:8000/setup-plan
curl http://localhost:8000/events
curl http://localhost:8000/jobs
curl http://localhost:8000/nodes
```
