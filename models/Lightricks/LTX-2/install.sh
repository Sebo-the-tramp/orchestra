uv venv --python=3.12

cd LTX-2
uv sync --frozen

uv pip install zmq

uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128