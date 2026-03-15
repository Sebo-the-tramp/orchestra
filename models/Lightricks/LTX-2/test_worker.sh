source .venv/bin/activate



# Required inputs
CHECKPOINT_PATH="/home/cavadalab/Documents/dvsv/LTX-2/ltx-2-19b-dev-fp8.safetensors"
DISTILLED_LORA_PATH="/home/cavadalab/Documents/dvsv/LTX-2/ltx-2-19b-distilled-lora-384.safetensors"
DISTILLED_LORA_STRENGTH="0.8"
SPATIAL_UPSAMPLER_PATH="/home/cavadalab/Documents/dvsv/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT="/data0/.cache/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/snapshots/68f7ee4fbd59087436ada77ed2d62f373fdd4482"
IMAGE_PATH="/home/cavadalab/Documents/scsv/covision/data/original_clip/0000000003.png"
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
    --prompt "Create a photorealistic video of the scene as it is now, where the paver moves slowly and some human workers help in the costruction work." \
    --output-path output.mp4 \
    --quantization fp8-cast \
    --image $IMAGE_PATH $IMAGE_FRAME_IDX $IMAGE_STRENGTH
    # we might have to add
    # worker_id 
    # model_id
    # and do a bit different input especially prompt, output and image
    # as those should come from zmq
