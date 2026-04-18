import sys
import torch
import argparse

from pathlib import Path
current_dir = Path(__file__).resolve()
sys.path.append(str(current_dir.parent.parent.parent.parent))

from typing import Literal
from models.base_worker import BaseWorker, BaseModel

from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

##########################################
#              Model definition          #
##########################################

models_available = {
    "facebook/dinov3-vits16-pretrain-lvd1689m": {"gpu_mem": 4000, "basefolder": "facebook/dinov3"}, # to fix memory requirements
    "facebook/dinov3-vits16plus-pretrain-lvd1689m": {"gpu_mem": 4000, "basefolder": "facebook/dinov3"}, # to fix memory requirements
    "facebook/dinov3-vitb16-pretrain-lvd1689m": {"gpu_mem": 4000, "basefolder": "facebook/dinov3"}, # to fix memory requirements
    "facebook/dinov3-vitl16-pretrain-lvd1689m": {"gpu_mem": 4000, "basefolder": "facebook/dinov3"}, # to fix memory requirements
    "facebook/dinov3-vith16-pretrain-lvd1689m": {"gpu_mem": 4000, "basefolder": "facebook/dinov3"}, # to fix memory requirements
    "facebook/dinov3-vith16plus-pretrain-lvd1689m": {"gpu_mem": 4000, "basefolder": "facebook/dinov3"}, # to fix memory requirements
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": {"gpu_mem": 4000, "basefolder": "facebook/dinov3"}, # to fix memory requirements
}

##########################################
#            Model input/output          #
##########################################

class DinoV3Config(BaseModel):
    image_size: int
    device_map: str
    torch_dtype: Literal["float32", "float16", "bfloat16", "int8"]

class Request(BaseWorker.BaseRequest):
    type: str
    request_id: str
    model_name: str
    num_images: int
    args_per_model: DinoV3Config

class NumpyArrayModel(BaseModel):
    dtype: str
    shape: tuple
    frame_index: int

class AnswerDinoV3(BaseModel):
    patch_features: NumpyArrayModel
    cls_token_feature: NumpyArrayModel
    register_token_feature: NumpyArrayModel

class Response(BaseWorker.BaseResponse):
    type: str
    request_id: str
    answer: AnswerDinoV3
    model_name: str


##########################################
#          Model methods override        #
##########################################

class Worker(BaseWorker):

    model = None
    processor = None

    @property
    def request_model(self):
        return Request

    @property
    def response_model(self):
        return Response

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-id", default="")
        parser.add_argument("--router-connect", default=BaseWorker.ROUTER_CONNECT)
        parser.add_argument("--worker-id", default=None, required=True, help="Unique identifier for the worker, used in logging and router identity")
        parser.add_argument("--device-map", default="auto", help="Device map to use when loading the model, e.g. 'auto' or 'cuda:0'")
        parser.add_argument("--torch-dtype", default="float16", help="Torch dtype to use when loading the model, e.g. 'float16' or 'float32'")
        parser.add_argument("--image-size", default=224, type=int, help="Image size to use when loading the model, e.g. 224")
        return parser.parse_args()

    def load_model(self, args):
        print(f"Received args for model loading: {args}")
        self.processor = AutoImageProcessor.from_pretrained(args.model_id, size={"height": args.image_size, "width": args.image_size})

        self.model = AutoModel.from_pretrained(
            args.model_id,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            image_size=args.image_size,
        ).eval()

    def inference(self, _: Request, frames: list) -> dict:

        patch_size = self.model.config.patch_size
        inputs = self.processor(images=frames, return_tensors="pt").to(self.model.device)
        batch_size, _, img_height, img_width = inputs.pixel_values.shape
        num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
        num_patches_flat = num_patches_height * num_patches_width

        with torch.inference_mode():
            outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        assert last_hidden_states.shape == (batch_size, 1 + self.model.config.num_register_tokens + num_patches_flat, self.model.config.hidden_size)

        cls_token_features = last_hidden_states[:, 0, :]
        patch_features_flat = last_hidden_states[:, 1 + self.model.config.num_register_tokens:, :]
        patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))

        register_token_features = last_hidden_states[:, 1:1 + self.model.config.num_register_tokens, :]

        return {
            "patch_features": patch_features.detach().cpu(),
            "cls_token_feature": cls_token_features.detach().cpu(),
            "register_token_feature": register_token_features.detach().cpu(),
        }


if __name__ == "__main__":
    worker = Worker()
    worker.run()