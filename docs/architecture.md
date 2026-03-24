# Architecture

## Runtime Shape

```text
client (REQ)
  -> broker_core.py (ROUTER)
      -> validate model_name against models/*/config.yaml
      -> append Job to the per-model deque
      -> spawn a worker if no pool exists for that model
      -> hand the next queued Job to an idle worker
      -> forward SUCCESS / ERROR back to the original client
  -> worker (DEALER)
      -> send HEARTBEAT
      -> poll for work
      -> run inference
      -> return SUCCESS / ERROR
```

## Main Pieces

| Path | Role |
| --- | --- |
| `broker_core.py` | Main event loop, job queues, worker spawn, result forwarding |
| `models/*/config.yaml` | Declares which model names exist and where their worker lives |
| `models/*/schema.py` | Intended request/response validation surface |
| `models/*/*/worker.py` | Actual inference worker entrypoints |
| `utils/transport.py` | Shared worker socket helper |
| `utils/image_io.py` | Shared image decoding helpers |

## Broker State

The broker keeps three core structures in memory:

| Structure | Purpose |
| --- | --- |
| `jobs_registry` | Per-model `deque` of queued jobs |
| `worker_registry` | Per-model `WorkerPool` split into waiting, idle, and busy worker ids |
| `pending_jobs` / `inflight_by_worker` | Maps running work back to the original client id |

## Worker Lifecycle

1. A client submits a multipart request with a `model_name`.
2. The broker rejects unknown models immediately.
3. The job is appended to that model's queue.
4. If no worker pool exists for the model, the broker spawns one from the registered worker path.
5. Workers announce themselves with `HEARTBEAT`.
6. Once a worker is idle, the broker sends the queued job as multipart frames.
7. The worker returns either `SUCCESS` or `ERROR`.
8. The broker forwards that payload to the original client and marks the worker idle again.

## Current Constraints

- The runtime host and broker address are still hardcoded around `tcp://10.10.151.14:5556`.
- The broker uses a single `ROUTER` socket for both client and worker traffic.
- Idle shutdown is model-specific in the worker code today: InternVL uses `10s`, SAM3 uses `30s`.
- The docs reflect the current code, not an idealized target architecture.
