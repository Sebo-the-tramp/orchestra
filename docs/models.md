# Models

## Registry

The live broker scans for `worker.py` files and reads a static
`models_available` dictionary from each one.

| File | Use |
| --- | --- |
| `models/<lab>/<family>/worker.py` | Active worker entrypoint |
| `models_available` | Public names, worker folder, resource hints |
| `models/<lab>/config.yaml` | Older config path, still present |
| `models/<lab>/schema.py` | Shared schema target |

## Current Families

| Family | Public Name | Worker | Status |
| --- | --- | --- | --- |
| DINOv3 | `facebook/dinov3-*` | `models/facebook/dinov3/worker.py` | 🟢 Active |
| InternVL | `OpenGVLab/InternVL*` | `models/OpenGVLab/InternVL/worker_tofix.py` | 🟡 Not active |
| SAM3 | `facebook/sam3` | `models/facebook/sam3/worker_tofix.py` | 🟡 Not active |
| LTX | `Lightricks/LTX-2.3` | `models/lightricks/LTX-2/worker_tofix.py` | 🟡 Not active |

## DINOv3 Payload

Image frames plus model args.

```json
{
  "type": "REQUEST",
  "request_id": "uuid",
  "model_name": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
  "num_images": 1,
  "args_per_model": {
    "image_size": 224,
    "device_map": "auto",
    "torch_dtype": "float16"
  }
}
```

## InternVL Payload

Legacy payload from the old test script.

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

## SAM3 Payload

Legacy payload from the old test script.

```json
{
  "request_id": "uuid",
  "model_name": "facebook/sam3",
  "prompt": ["flower"],
  "confidence_threshold": [0.5],
  "config": {}
}
```

`models/facebook/schema.py` is narrower than the SAM3 worker payload. The
payload above matches `tests/sam.py`.

## Resource Hints

| Field | Meaning |
| --- | --- |
| `basefolder` | Worker path relative to `models/` |
| `gpu_memory` / `gpu_mem` | Rough GPU memory need in MiB |
| `tp` | Tensor parallelism hint |
