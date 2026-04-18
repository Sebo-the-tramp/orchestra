# Request Flow

## Client To Broker

The current clients use a ZeroMQ `REQ` socket and send multipart frames:

```python
socket.send_multipart(
    [json.dumps(payload).encode("utf-8"), image_0, image_1, ...]
)
```

At the broker boundary, that arrives as:

```text
[client_id, b"", payload_json, image_0, image_1, ...]
```

## Broker To Worker

When a worker becomes idle, the broker sends:

```text
[f"{model_name}-{worker_id}".encode(), b"", payload_json, image_0, image_1, ...]
```

The worker connects with a `DEALER` identity built from the model name and worker id, so the broker can route a job to one concrete worker instance.

## Worker Replies

Workers send plain JSON payloads back to the broker. The broker forwards those payloads directly to the original client.

### Success

```json
{
  "type": "SUCCESS",
  "request_id": "uuid",
  "answer": "...",
  "model_name": "OpenGVLab/InternVL3-2B-AWQ"
}
```

### Error

```json
{
  "type": "ERROR",
  "request_id": "uuid",
  "message": "Job processing failed",
  "model_name": "facebook/sam3"
}
```

## Heartbeats

Workers send:

```json
{
  "type": "HEARTBEAT",
  "model_name": "facebook/sam3"
}
```

The broker uses those heartbeats to move a worker from the waiting set to the idle set.

## Current Cleanups Worth Tracking

- Request schemas and worker payloads are not fully aligned yet, especially for SAM3.
- Address defaults inside worker files still show older fallback values, even though spawned workers receive the live broker address from the broker itself.
- The broker currently forwards worker payloads mostly as-is, so the wire format is defined by the worker implementations.
