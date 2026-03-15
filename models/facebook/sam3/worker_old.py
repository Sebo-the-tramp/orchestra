import os
import io
import zmq
import sys
import uuid
import json
import time
import torch

from pathlib import Path
from PIL import Image

import sam3.sam3
from sam3.sam3 import build_sam3_image_model
from sam3.sam3.model.sam3_image_processor import Sam3Processor

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent.parent))

torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# ---- METADATA VARIABLES ----
VRAM_REQUIRED = 9000
LOAD_TIME_SECONDS = 2
BATCH_SIZE = 1
MODEL_NAME = "sam3"
ROUTER_CONNECT = "tcp://10.10.151.14:5555"
# ----------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def connect_to_router():
    context = zmq.Context()        
    socket = context.socket(zmq.DEALER)
    worker_id = str(MODEL_NAME + "-" + str(uuid.uuid4())).encode('utf-8')    
    socket.setsockopt(zmq.IDENTITY, worker_id)    
    socket.connect(ROUTER_CONNECT)
    return socket


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
    
    processor = load_model()
    print("Model loaded and ready for jobs!")

    socket = connect_to_router()

    while True:
        # We can safely send a heartbeat every loop cycle
        socket.send_json({"type": "HEARTBEAT", "model": MODEL_NAME})
        
        # Poll for 1000ms to see if the router sent us a JOB
        if socket.poll(timeout=1000):
            print("I am alive")

            try:
                # If poll is true, there is a message waiting to be received
                frames = socket.recv_multipart()
                print("Received message with {} frames.".format(len(frames)))

                # # --- DEBUGGING BLOCK ---
                # print(f"\n--- {time.ctime()} - INCOMING MESSAGE ({len(frames)} frames) ---")
                # for i, frame in enumerate(frames):
                #     # We truncate the output so huge images don't flood your terminal
                #     safe_print = frame[:10] + b"... (truncated)" if len(frame) > 50 else frame
                #     print(f"Frame {i}: {safe_print}")
                # print("--------------------------------------\n")            

                # prompt = message.get("prompt").replace("<IMAGE_TOKEN>", IMAGE_TOKEN)
                # images = message.get("images", [])
                # client_id = frames[0]
                # frames[1] is the empty delimiter
                raw_metadata = frames[1]                      # The JSON bytes
                # here we should dynamically check from the metatdata how many images there are, 
                # but for simplicity we just take all remaining frames as images
                image_frames = frames[2:]                     # All attached images (0 to N)
                payload = json.loads(raw_metadata.decode('utf-8'))

                prompt = payload.get("prompt")
                confidence_threshold = payload.get("confidence_threshold")
                # print(f"Decoded payload: {payload}")
                # print(f"request_id: {payload.get('request_id')}, model: {payload.get('model')}, prompt: {payload.get('prompt')}, attached_images: {len(image_frames)}")                
                pil_images = []  # This is where you'd convert the byte frames back into PIL images or tensors as needed by your model.       
                for img_bytes in image_frames:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    pil_images.append(img)            

                print(pil_images[0].size)

                print(f"Running inference on prompt: {prompt} with {len(pil_images)} images...")    

                response = single_image_model_inference(processor, pil_images[0], prompt=prompt[0], confidence_threshold=confidence_threshold[0])                
                print("Inference complete, preparing response...")
                print("Response size in MB JSON :", sys.getsizeof(response) / (1024 * 1024))
                time_before_json = time.perf_counter()
                clean_response = prepare_for_json(response)
                time_after_json = time.perf_counter()
                print(f"Time to prepare JSON response: {time_after_json - time_before_json:.2f} seconds")
                print("Response size in MB CLEANED:", sys.getsizeof(clean_response) / (1024 * 1024))
                print("Response prepared, sending back to broker...")
                socket.send_json({"type": "SUCCESS", "req_id": payload.get("request_id"), "answer": clean_response})
                print("Finished")
            except Exception as e:
                print(f"Error processing job: {e}")
                socket.send_json({"type": "ERROR", "req_id": payload.get("request_id"), "message": str(e)})
            
        else:
            pass
            # If poll times out, we just loop around and ping again.
            # No crash will happen because DEALER doesn't require a recv() first!
            print("No jobs yet, sending next ping...")

if __name__ == "__main__":
    main()

# source /home/cavadalab/Documents/scsv/the_thing/backend/models/segmentation/sam3/.venv/bin/activate