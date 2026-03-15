# 🎶 orchestra 🎶

## ATTENTION, the repo is heavily under construction ETA 1 week with first working models

ORCHESTRA is a barebone autonomous dispatch layer for serious model serving 🎼: one broker, many stupid workers, minimal maintenance, minimal downtime, and clean routing of any well-formed request to the right compute.

The design is intentionally simple 🎻. `broker.py` is the central mind that tracks queues, workers, and resources. Each worker in `./workers/` does one thing only: take assigned work, execute it, report back, and die when idle. `example_client.py` is the seed for the external interface that will eventually grow into the public-facing API layer 🎹. The goal is being able to run natively *any* model supported in huggingface, with their own implementation and speed, with orchestra being the middle layer with zero overhead. Similar to having all models on the GPU at the *same time*.

Think of the broker as a conductor with a magician's baton ✨: one precise gesture, and the right section starts playing at the right time on the right machine. You are the magician now!

## Requirements

- `uv`

## Install the repo

```bash
uv sync
```

This installs the repo together with:

- `pyzmq`
- `tqdm`
- `Pillow`

Note: the Python import is `zmq`, but the package dependency name is `pyzmq`.

## Install dependencies only

If you want the environment without installing the project itself:

```bash
uv sync --no-install-project
```

## Dev tooling

Install the dev tools:

```bash
uv sync --group dev
```

Install the git hooks:

```bash
uv run pre-commit install
```

This step requires the folder to be a Git repository.

Run the checks manually:

```bash
uv run pre-commit run --all-files
```

## Start orchestra

From the repository root:

```bash
./start_orchestra.sh
```

Raise the baton 🎶 and let the system play.
