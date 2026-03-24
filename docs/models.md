# Models

## Registry Layout

Each lab folder is expected to provide three things:

| File | Purpose |
| --- | --- |
| `models/<lab>/config.yaml` | Maps public `model_name` values to worker folders and resource hints |
| `models/<lab>/schema.py` | Pydantic request/response models |
| `models/<lab>/<family>/worker.py` | The executable worker entrypoint |

## Current Families

| Family | Registry Pattern | Worker Path | Notes |
| --- | --- | --- | --- |
| InternVL | `OpenGVLab/InternVL*` | `models/OpenGVLab/InternVL/worker.py` | VLM worker using `lmdeploy` |
| SAM3 | `facebook/sam3` | `models/facebook/sam3/worker.py` | Image segmentation worker built on the local `sam3` checkout |

## Request Payloads In Practice

### InternVL

The current InternVL path takes a single text prompt plus one or more image frames:

```json
{
  "request_id": "uuid",
  "model_name": "OpenGVLab/InternVL3-2B-AWQ",
  "prompt": "Classify the object in <IMAGE_TOKEN> ...",
  "config": {
    "tp": 1
  }
}
```

### SAM3

The live worker currently expects prompt and threshold lists:

```json
{
  "request_id": "uuid",
  "model_name": "facebook/sam3",
  "prompt": ["flower"],
  "confidence_threshold": [0.5],
  "config": {}
}
```

!!! note

    `models/facebook/schema.py` is still narrower than the worker implementation. The docs describe the
    current wire format used by `tests/sam.py`, which is what the worker actually consumes today.

## Resource Hints

`config.yaml` also carries lightweight scheduling hints:

| Field | Meaning |
| --- | --- |
| `basefolder` | Relative path from `models/` to the worker directory |
| `gpu_memory` | Current rough GPU memory requirement in MiB |
| `tp` | Tensor parallelism hint used by some workers |

These are enough for the current broker, which only needs to find the worker code and build a launch command.
