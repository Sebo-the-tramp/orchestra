# source .venv/bin/activate

ROOT="/data0/.cache/hub/models--Lightricks--LTX-2.3/snapshots/5a9c1c680bc66c159f708143bf274739961ecd08"

# Required inputs
CHECKPOINT_PATH="${ROOT}/ltx-2.3-22b-dev.safetensors"
DISTILLED_LORA_PATH="${ROOT}/ltx-2.3-22b-distilled-lora-384.safetensors"
DISTILLED_LORA_STRENGTH="0.8"
SPATIAL_UPSAMPLER_PATH="${ROOT}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT="/data0/.cache/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482"
IMAGE_PATH="/home/cavadalab/Documents/scsv/orchestra/models/Lightricks/LTX-2/IMG_1097.png"
# IMAGE_PATH="/data0/sebastian.cavada/diffusion/flux2/outputs_yaml/asphalt_paver_real2real/asphalt_paver_real2real_run_02/images/flux2_klein_1770808116_0001.png"
# IMAGE_PATH="/home/cavadalab/Documents/scsv/covision/image-models/LightX2V/save_results/rain.png"
# IMAGE_PATH=""
IMAGE_FRAME_IDX=0
IMAGE_STRENGTH=0.8

python worker.py \
    --checkpoint-path $CHECKPOINT_PATH \
    --distilled-lora $DISTILLED_LORA_PATH 0.8 \
    --spatial-upsampler-path $SPATIAL_UPSAMPLER_PATH \
    --gemma-root $GEMMA_ROOT \
    --prompt "This person running forward in the same style of the image saying sono un missileee" \
    --output-path output.mp4 \
    --quantization fp8-cast \
    --image $IMAGE_PATH $IMAGE_FRAME_IDX $IMAGE_STRENGTH \
    --height 1024 \
    --width 704 \
    --num-frames 242
    # we might have to add
    # worker_id 
    # model_id
    # and do a bit different input especially prompt, output and image
    # as those should come from zmq
