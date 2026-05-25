# Getting Started

Run commands from the repository root unless noted.

## Requirements

| Requirement | Notes |
| --- | --- |
| Python `>=3.12` | Root project metadata |
| `uv` | Package and command runner |
| Broker host | Current code expects `tcp://10.10.151.14:5556` |
| Worker venvs | Broker launches `models/<lab>/<family>/.venv/bin/python` |

## Docs

Preview:

```bash
uv run --no-project --with mkdocs-material mkdocs serve
```

Build:

```bash
uv run --no-project --with mkdocs-material mkdocs build --strict
```

## Runtime

Install:

```bash
uv sync
```

Start broker:

```bash
uv run python broker_core.py
```

Start tmux view:

```bash
./start_orchestra.sh
```

## Worker Layout

```text
models/<lab>/
  config.yaml
  schema.py
  <family>/
    worker.py
    .venv/
```

If `worker.py` or `.venv/bin/python` is missing, spawn fails.

## Test Requests

| Model | Script | Status |
| --- | --- |
| DINOv3 | Add a client payload matching `models/facebook/dinov3/worker.py` | 🟢 Active worker |
| InternVL | `tests/test_intern.py` | 🟡 Worker file is `worker_tofix.py` |
| SAM3 | `tests/sam.py` | 🟡 Worker file is `worker_tofix.py` |
