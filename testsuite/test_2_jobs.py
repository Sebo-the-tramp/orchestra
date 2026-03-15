import zmq
import sys
import uuid
import json

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from utils.image_io import image_bytes


BROKER_CLIENT_ADDRESS = "tcp://10.10.151.14:5556"

MODEL = "OpenGVLab/InternVL3-2B-AWQ"
PROMPT = ("Can you tell to what class does the in the image <IMAGE_TOKEN> belong to A) bobcat <IMAGE_TOKEN> B) asphalt_paving_machine <IMAGE_TOKEN> C) asphalt_roller_machine <IMAGE_TOKEN> D) none of the above. Just reply with A-B-C or D and nothing else.")

REFERENCE_IMAGES = [
    Path("/home/cavadalab/Documents/scsv/covision/datasets/curated_db/test_set_machinery_balanced/bobcat/WH5bmcD8bnE_frame_003966_0.png"),
    Path("/home/cavadalab/Documents/scsv/covision/datasets/curated_db/test_set_machinery_balanced/paver/jPNAfMWMtNA_frame_000218_0.png"),
    Path("/home/cavadalab/Documents/scsv/covision/datasets/curated_db/test_set_machinery_balanced/roller/yxzzNUDUWDM_frame_000075_1.png"),
]

socket = zmq.Context.instance().socket(zmq.REQ)
socket.connect(BROKER_CLIENT_ADDRESS)
print("socket connected")

ref_images = [image_bytes(path) for path in REFERENCE_IMAGES]

payload = {
    "request_id": str(uuid.uuid4()),
    "model_name": MODEL,
    "prompt": PROMPT,
    "config": {
        "tp":1
    }
}

socket.send_multipart(
    [json.dumps(payload).encode("utf-8"), image_bytes(REFERENCE_IMAGES[0]), *ref_images]
)
print("Sent request 1")
response = json.loads(socket.recv_multipart()[-1].decode("utf-8"))
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
