import os
import sys
import json
import time
import logging

from pathlib import Path
import torch

current_dir = Path(__file__).resolve().parent
repo_root = current_dir / "LTX-2"
print(f"Current directory: {current_dir}")
sys.path.insert(0, str(repo_root / "packages" / "ltx-core" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "ltx-pipelines" / "src"))

sys.path.append(str(current_dir.parent.parent.parent))

from ltx_pipelines import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.utils.args import default_2_stage_arg_parser

from utils.image_io import decode_images
from utils.transport import connect_to_router

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_MODEL_ID = "Lightricks/LTX-2.3"
IDLE_SHUTDOWN_SECONDS = 30
POLL_TIMEOUT_MS = 1000

ROUTER_CONNECT = "tcp://10.10.151.14:5555"
SHUTDOWN_MESSAGE_TYPE = "SHUTDOWN"

logger = logging.getLogger(__name__)


def load_model(args):
    return TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
    )


@torch.inference_mode()
def run_model(args, pipeline):
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )

    return args.output_path


def main():    
    parser = default_2_stage_arg_parser()
    args = parser.parse_args()
    model_name = DEFAULT_MODEL_ID
    pipe = load_model(args)
    logger.info("Model loaded")

    test = run_model(args, pipe)
    logger.info("Test run completed, output saved to %s", test)

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
            # response = pipe((payload.get("prompt"), decode_images(frames[2:])))
            response = run_model(args, pipe) # should not be args lol!
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
