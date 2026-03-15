import os
import sys
import json
import time
import logging
import argparse

from pathlib import Path

from lmdeploy import ChatTemplateConfig, TurbomindEngineConfig, pipeline

current_dir = Path(__file__).resolve().parent
print(f"Current directory: {current_dir}")
sys.path.append(str(current_dir.parent.parent.parent))

from utils.image_io import decode_images
from utils.transport import connect_to_router

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_MODEL_ID = "OpenGVLab/InternVL3-38B-AWQ"
DEFAULT_CHAT_TEMPLATE = "internvl2_5"
DEFAULT_SESSION_LEN = 8192 * 2
DEFAULT_TP = 2
IDLE_SHUTDOWN_SECONDS = 10
POLL_TIMEOUT_MS = 1000

ROUTER_CONNECT = "tcp://10.10.151.14:5555"
SHUTDOWN_MESSAGE_TYPE = "SHUTDOWN"

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--router-connect", default=ROUTER_CONNECT)
    parser.add_argument("--session-len", type=int, default=DEFAULT_SESSION_LEN)
    parser.add_argument("--tp", type=int, default=DEFAULT_TP)
    parser.add_argument("--worker-id", default=None, required=True, help="Unique identifier for the worker, used in logging and router identity")
    return parser.parse_args()


def load_model(model_id, session_len, tp):
    return pipeline(
        model_id,
        backend_config=TurbomindEngineConfig(session_len=session_len, tp=tp),
        chat_template_config=ChatTemplateConfig(model_name=DEFAULT_CHAT_TEMPLATE),
    )


def main():    
    args = parse_args()
    model_name = args.model_id
    pipe = load_model(args.model_id, args.session_len, args.tp)
    logger.info("Model loaded")

    socket = connect_to_router(model_name, args.router_connect, worker_id=args.worker_id)
    last_work_time = time.monotonic()

    while True:
        print(f"Sending heartbeat {model_name}")
        socket.send_json({"type": "HEARTBEAT", "model_name": model_name})

        if not socket.poll(timeout=POLL_TIMEOUT_MS):
            if time.monotonic() - last_work_time >= IDLE_SHUTDOWN_SECONDS:
                logger.info("No work received for %s seconds, shutting down", IDLE_SHUTDOWN_SECONDS)
                return
            continue

        last_work_time = time.monotonic()
        req_id = None

        try:
            frames = socket.recv_multipart()
            payload = json.loads(frames[1].decode("utf-8"))
            if payload.get("type") == SHUTDOWN_MESSAGE_TYPE:
                logger.info("Received %s message, shutting down", SHUTDOWN_MESSAGE_TYPE)
                return
            req_id = payload.get("request_id")
            response = pipe((payload.get("prompt"), decode_images(frames[2:])))
            answer = response.text if hasattr(response, "text") else str(response)
            logger.info("Processed request_id=%s successfully", req_id)
            socket.send_json({"type": "SUCCESS", "req_id": req_id, "answer": answer, "model_name": args.model_id})
        except Exception:
            logger.exception("Job processing failed")
            socket.send_json({"type": "ERROR", "req_id": req_id, "message": "Job processing failed", "model_name": args.model_id})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
