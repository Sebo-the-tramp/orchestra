import io

from PIL import Image
from pathlib import Path


def decode_images(image_frames):
    return [Image.open(io.BytesIO(image_bytes)).convert("RGB") for image_bytes in image_frames]


def image_bytes(path: Path) -> bytes:
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()