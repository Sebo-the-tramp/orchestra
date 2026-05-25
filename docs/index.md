---
hide:
  - toc
---

# Orchestra

Small broker-worker runtime for model jobs.

The broker owns queues, starts workers when needed, and routes replies back to
clients. Workers run in their own environments and exit after idle time.

## Current State

| Area | Status |
| --- | --- |
| Broker | 🟢 `broker_core.py` queues jobs, starts missing workers, forwards replies |
| Workers | 🟢 `models/<lab>/<family>/worker.py` processes one model family |
| Registry | 🟢 Broker discovers `models/**/worker.py` and reads `models_available` |
| Models | 🟢 DINOv3 has an active `worker.py` |
| Legacy workers | 🟡 InternVL, SAM3, and LTX live as `worker_tofix.py` |
| Schemas | 🟡 Some schemas and payloads still lag active workers |
| Deployment | 🟡 Broker address is still hardcoded around `tcp://10.10.151.14:5556` |

## Shape

```text
client -> broker_core.py -> worker -> broker_core.py -> client
```

## Files

| Path | Role |
| --- | --- |
| `broker_core.py` | Queues, worker state, routing |
| `models/<lab>/config.yaml` | Legacy model config |
| `models/<lab>/schema.py` | Request and response schema target |
| `models/<lab>/<family>/worker.py` | Inference process |
| `utils/transport.py` | Worker socket helper |
| `tests/` | Current runnable client examples |

## Read Next

| Page | Use |
| --- | --- |
| [Getting Started](getting-started.md) | Run docs, broker, and example clients |
| [Architecture](architecture.md) | Broker state and worker lifecycle |
| [Models](models.md) | Active and legacy model layout |
| [Request Flow](request-flow.md) | ZeroMQ frames |
| [Publishing](publishing.md) | GitHub Pages build |
| [Roadmap](roadmap.md) | Known cleanup |
