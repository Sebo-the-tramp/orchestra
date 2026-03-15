import os
import sys
import json
import time
import torch
import logging
import argparse

from PIL import Image
from pathlib import Path

import sam3.sam3
from sam3.sam3 import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent.parent))

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

from utils.image_io import decode_images
from utils.transport import connect_to_router

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_MODEL_ID = "facebook/sam3"
DEFAULT_CHAT_TEMPLATE = "internvl2_5"
IDLE_SHUTDOWN_SECONDS = 30
POLL_TIMEOUT_MS = 1000

ROUTER_CONNECT = "tcp://10.10.151.14:5555"
SHUTDOWN_MESSAGE_TYPE = "SHUTDOWN"

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--router-connect", default=ROUTER_CONNECT)    
    parser.add_argument("--worker-id", default=None, required=True, help="Unique identifier for the worker, used in logging and router identity")
    return parser.parse_args()


def load_model():
    device = "cuda"    
    sam3_root = os.path.join(os.path.dirname(sam3.sam3.__file__))
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"    
    model = build_sam3_image_model(bpe_path=bpe_path, device=device)
    processor = Sam3Processor(model=model, device=device)
    return processor


@torch.inference_mode()
def single_image_model_inference(processor, image: Image.Image, prompt: str, confidence_threshold: float = 0.5) -> dict:

    result = single_image_multi_prompt_model_inference(processor=processor, image=image, prompt=[prompt], confidence_threshold=[confidence_threshold])
    return result[prompt]

@torch.inference_mode()
def single_image_multi_prompt_model_inference(processor, image: Image.Image, prompt: list[str], confidence_threshold: list[float]) -> dict:

    inference_state_by_prompt = {}
    
    for p,c in zip(prompt, confidence_threshold):        
        inference_state = processor.set_confidence_threshold(c)
        inference_state = processor.set_image(image)
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=p)        
        inference_state_by_prompt[p] = inference_state.copy()

    return inference_state_by_prompt

def prepare_for_json(output):    
    data = {}
    data["boxes"] = output['boxes'].cpu().tolist()
    data["masks"] = output['masks'].cpu().numpy().tolist()
    data["scores"] = output['scores'].cpu().tolist()
    return data


def main():    
    args = parse_args()
    model_name = args.model_id
    processor = load_model()
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
            prompt = payload.get("prompt")
            confidence_threshold = payload.get("confidence_threshold")
            pil_images = decode_images(frames[2:])

            response = single_image_model_inference(processor, pil_images[0], prompt=prompt[0], confidence_threshold=confidence_threshold[0])                
            clean_response = prepare_for_json(response)            
            logger.info("Processed request_id=%s successfully", req_id)
            socket.send_json({"type": "SUCCESS", "req_id": payload.get("request_id"), "answer": clean_response, "model_name": DEFAULT_MODEL_ID})
        except Exception:
            logger.exception("Job processing failed")
            socket.send_json({"type": "ERROR", "req_id": req_id, "message": "Job processing failed", "model_name": DEFAULT_MODEL_ID})


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
