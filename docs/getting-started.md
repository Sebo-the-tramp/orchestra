# Getting Started

## What You Need

| Requirement | Why |
| --- | --- |
| Python `>=3.12` | Declared in the root project metadata |
| `uv` | Used for local commands and docs preview |
| A broker host reachable at `tcp://10.10.151.14:5556` | Hardcoded in the current runtime |
| Per-worker virtual environments | The broker spawns `models/<lab>/<family>/.venv/bin/python` directly |

## Preview The Docs Locally

The docs toolchain is isolated from the project install for now:

```bash
uv run --no-project --with mkdocs-material mkdocs serve
```

Build the static site without starting a local server:

```bash
uv run --no-project --with mkdocs-material mkdocs build --strict
```

## Run Orchestra

Install the root environment:

```bash
uv sync
```

Start the broker from the repository root:

```bash
uv run python broker_core.py
```

Or start the tmux dashboard:

```bash
./start_orchestra.sh
```

## Worker Assumptions

The current broker is strict about worker layout:

```text
models/<lab>/
  config.yaml
  schema.py
  <family>/
    worker.py
    .venv/
```

If a model is declared in `config.yaml` and the worker path does not contain both `worker.py`
and `.venv/bin/python`, worker spawn fails immediately.

## Send A Test Request

Two small request examples already exist:

| Model | File |
| --- | --- |
| InternVL | `tests/test_intern.py` |
| SAM3 | `tests/sam.py` |

Those scripts show the current multipart request format used by the broker.
