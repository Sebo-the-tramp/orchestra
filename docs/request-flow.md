# Request Flow

## Client To Broker

Clients use ZeroMQ `REQ` and send multipart frames:

```python
socket.send_multipart(
    [json.dumps(payload).encode("utf-8"), image_0, image_1]
)
```

Broker receives:

```text
[client_id, b"", payload_json, image_0, image_1]
```

## Broker To Worker

Broker sends work to one concrete `DEALER` identity:

```text
[f"{model_name}-{worker_id}".encode(), b"", payload_json, image_0, image_1]
```

## Worker Replies

Workers send JSON. Broker forwards it to the original client.

Success:

```json
{
  "type": "SUCCESS",
  "request_id": "uuid",
  "answer": "...",
  "model_name": "OpenGVLab/InternVL3-2B-AWQ"
}
```

Error:

```json
{
  "type": "ERROR",
  "request_id": "uuid",
  "message": "Job processing failed",
  "model_name": "facebook/sam3"
}
```

## Heartbeat

```json
{
  "type": "HEARTBEAT",
  "model_name": "facebook/sam3"
}
```

The broker uses this to move workers from waiting to idle.

## Known Gaps

| Area | Status |
| --- | --- |
| Schemas vs payloads | 🟡 Not fully aligned, especially SAM3 |
| Worker address defaults | 🟡 Some files still show old fallback addresses |
| Reply format | 🟡 Mostly defined by workers today |
