import zmq
import sys
import uuid
import json

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.image_io import image_bytes


BROKER_CLIENT_ADDRESS = "tcp://10.10.151.14:5556"

MODEL = "facebook/sam3"

socket = zmq.Context.instance().socket(zmq.REQ)
socket.connect(BROKER_CLIENT_ADDRESS)
print("socket connected")

path = "/data0/sebastian.cavada/datasets/PlantCLEF2026/PlantCLEF2025test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/CBN-can-A1-20230705.jpg"

payload = {
    "request_id": str(uuid.uuid4()),
    "model_name": MODEL,
    "prompt": ["flower"],
    "confidence_threshold": [0.5],
    "config": {},
}

socket.send_multipart(
    [json.dumps(payload).encode("utf-8"), image_bytes(path)]
)
print("Sent request 1")
response = json.loads(socket.recv_multipart()[-1].decode("utf-8"))
print(response)
if response.get("type") == "ERROR":
    raise RuntimeError(response.get("message", "Unknown VLM error"))

print("Received response")

# socket.send_multipart(
#     [json.dumps(payload).encode("utf-8"), image_bytes(REFERENCE_IMAGES[0]), *ref_images]
# )
# print("Sent request 2")

# socket.send_multipart(
#     [json.dumps(payload).encode("utf-8"), image_bytes(REFERENCE_IMAGES[0]), *ref_images]
# )
# print("Sent request 3")
