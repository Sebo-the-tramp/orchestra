uv venv --python 3.12

source .venv/bin/activate

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install matplotlib zmq pydantic einops decord pycocotools psutil

git clone https://github.com/facebookresearch/sam3.git
cd sam3
uv pip install -e .