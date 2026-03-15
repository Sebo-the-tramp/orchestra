# Worker Spec

## Purpose

A worker is a single long-lived process that:

- loads exactly one model or capability
- connects to the broker over ZeroMQ `DEALER`
- advertises availability with heartbeats
- receives one job at a time
- returns either `SUCCESS` or `ERROR`
- can terminate on idle timeout or explicit shutdown message

## Characteristics

- Single responsibility: one worker process should serve one model family or one task.
- Long-lived: model loading happens once at startup, not per request.
- Stateless per job: a job should not depend on previous jobs unless explicitly designed that way.
- Broker-driven: workers do not pull from storage or discover work themselves.
- Bare protocol: JSON metadata plus optional binary frames.
- Predictable exit: workers should shut down cleanly on idle timeout or control message.
- Observable: use structured logging, not `print`.
- Minimal shared surface: only truly generic code belongs in `workers/_utils`.

## Runtime Contract

At startup a worker should:

1. parse runtime args
2. configure logging
3. load the model
4. connect to the broker
5. enter the work loop

Inside the loop a worker should:

- send heartbeat messages periodically
- block on broker input with a poll timeout
- decode the incoming payload
- handle control messages before inference
- run inference
- send back a `SUCCESS` or `ERROR` payload

## Message Shape

### Worker -> broker

Heartbeat:

```json
{"type": "HEARTBEAT", "model": "<model-id>"}
```

Success:

```json
{"type": "SUCCESS", "req_id": "<request-id>", "answer": "..."}
```

Error:

```json
{"type": "ERROR", "req_id": "<request-id>", "message": "..."}
```

### Broker -> worker

Job metadata is JSON in frame `1`, with binary attachments in frames `2+`.

Example control message:

```json
{"type": "SHUTDOWN"}
```

## Code Guidelines

- Keep the worker entrypoint small.
- Put generic transport helpers in `workers/_utils/transport.py`.
- Put generic image decoding helpers in `workers/_utils/image_io.py`.
- Keep model-specific preprocessing and postprocessing inside the worker or its model package.
- Avoid comments that explain obvious code.
- Avoid dead constants, unused imports, and debug blocks.
- Prefer constants or CLI args for runtime knobs.

## Non-Goals

- No business logic in the broker.
- No worker-specific hacks in shared utils.
- No implicit cross-worker state.
