import io
import json
import sys
import uuid

from pathlib import Path
from typing import Any

import zmq
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.image_io import image_bytes

BROKER_CLIENT_ADDRESS = "tcp://10.10.151.14:5556"
SAM_MODEL = "facebook/sam3"
VLM_MODEL = "OpenGVLab/InternVL3-2B-AWQ"
IMAGE_PATH = Path("/data0/sebastian.cavada/datasets/PlantCLEF2026/PlantCLEF2025test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/CBN-can-A1-20230705.jpg")
SAM_PROMPT = "flower"
SAM_CONFIDENCE_THRESHOLD = 0.5
VLM_PROMPT = "Describe the main object in <IMAGE_TOKEN> in one short sentence."
OUTPUT_DIR = PROJECT_ROOT / "recipes" / "outputs"
OUTPUT_CROP_PATH = OUTPUT_DIR / f"{IMAGE_PATH.stem}_sam_crop.png"


def request(
    socket: zmq.Socket,
    model_name: str,
    prompt: str | list[str],
    images: list[bytes],
    config: dict[str, Any],
    confidence_threshold: list[float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": str(uuid.uuid4()),
        "model_name": model_name,
        "prompt": prompt,
        "config": config,
    }
    if confidence_threshold is not None:
        payload["confidence_threshold"] = confidence_threshold
    socket.send_multipart([json.dumps(payload).encode("utf-8"), *images])
    response = json.loads(socket.recv_multipart()[-1].decode("utf-8"))
    assert response["type"] == "SUCCESS", response
    return response


def pil_image_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def confirm_override(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return
    answer = input(f"{path} exists. Override it? [y/N]: ").strip().lower()
    assert answer in {"y", "yes"}, f"Refusing to override {path}"


def crop_best_box(image: Image.Image, answer: dict[str, Any]) -> tuple[Image.Image, list[float], float]:
    boxes = answer["boxes"]
    scores = answer["scores"]
    assert boxes and scores, answer
    index = max(range(len(scores)), key=scores.__getitem__)
    left, top, right, bottom = [int(value) for value in boxes[index]]
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, image.width)
    bottom = min(bottom, image.height)
    assert right > left and bottom > top, boxes[index]
    return image.crop((left, top, right, bottom)), boxes[index], float(scores[index])


def main() -> None:
    assert IMAGE_PATH.is_file(), IMAGE_PATH
    socket = zmq.Context.instance().socket(zmq.REQ)
    socket.connect(BROKER_CLIENT_ADDRESS)
    original = Image.open(IMAGE_PATH).convert("RGB")

    sam_response = request(
        socket=socket,
        model_name=SAM_MODEL,
        prompt=[SAM_PROMPT],
        images=[image_bytes(IMAGE_PATH)],
        config={},
        confidence_threshold=[SAM_CONFIDENCE_THRESHOLD],
    )
    crop, box, score = crop_best_box(original, sam_response["answer"])
    confirm_override(OUTPUT_CROP_PATH)
    crop.save(OUTPUT_CROP_PATH)

    vlm_response = request(
        socket=socket,
        model_name=VLM_MODEL,
        prompt=VLM_PROMPT,
        images=[pil_image_bytes(crop)],
        config={"tp": 1},
    )
    print(
        json.dumps(
            {
                "sam_box": box,
                "sam_score": score,
                "crop_path": str(OUTPUT_CROP_PATH),
                "vlm_answer": vlm_response["answer"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
