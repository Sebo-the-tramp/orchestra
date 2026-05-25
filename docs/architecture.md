# Architecture

## Runtime

```text
client REQ
  -> broker ROUTER
      -> validate model_name against discovered workers
      -> append job to a per-model queue
      -> start worker if needed
      -> send queued job to an idle worker
      -> forward worker reply
  -> worker DEALER
      -> send HEARTBEAT
      -> receive work
      -> run inference
      -> return SUCCESS or ERROR
```

## Main Files

| Path | Role |
| --- | --- |
| `broker_core.py` | Main loop, queues, workers, replies |
| `models/**/worker.py` | Active worker files discovered at broker startup |
| `models_available` | Static dictionary inside each active worker |
| `models/*/config.yaml` | Legacy model config files still in the tree |
| `models/*/schema.py` | Intended validation layer |
| `models/*/*/worker.py` | Model-family worker |
| `utils/transport.py` | Worker socket setup |
| `utils/image_io.py` | Image decode helpers |

## Broker Memory

| Structure | Use |
| --- | --- |
| `jobs_registry` | One `deque` per model |
| `worker_registry` | Waiting, idle, and busy workers per model |
| `pending_jobs` | Original client for each request |
| `inflight_by_worker` | Running job for each busy worker |

## Worker Lifecycle

1. Client sends a request with `model_name`.
2. Broker rejects unknown models.
3. Broker queues the job.
4. Broker starts a worker if no pool exists.
5. Worker sends `HEARTBEAT`.
6. Broker marks the worker idle.
7. Broker sends the next queued job.
8. Worker returns `SUCCESS` or `ERROR`.
9. Broker forwards the payload to the client.
10. Broker marks the worker idle again.

## Constraints

| Area | Status |
| --- | --- |
| Broker address | 🟡 Hardcoded around `tcp://10.10.151.14:5556` |
| Socket topology | 🟡 One `ROUTER` handles clients and workers |
| Registry source | 🟡 Code uses `worker.py`, not the older `config.yaml` path |
| Idle timeout | 🟡 Base worker default is `60s`; older workers have local values |
| Docs | 🟢 Describe current code, not target design |
