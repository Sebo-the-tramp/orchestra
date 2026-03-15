# Installing InternLM Deploy with CUDA 12.8 support
uv venv --python python3.12

source .venv/bin/activate

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt

LMDEPLOY_VERSION=0.12.0
PYTHON_VERSION=312
CUDA_VERSION=cu128

uv pip install https://github.com/InternLM/lmdeploy/releases/download/v${LMDEPLOY_VERSION}/lmdeploy-${LMDEPLOY_VERSION}+${CUDA_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux2014_x86_64.whl --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}