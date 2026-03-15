from pydantic import BaseModel, Field


class LTX2Config(BaseModel):
    checkpoint_path: str
    distilled_lora_path: str
    distilled_lora_strength: float = Field(default=0.5, gt=0.0, lt=1.0)
    spatial_upsampler_path: str     
    gemma_root: str # path to the gemma root folder, used for loading the gemma model and processor    
    enable_fp8: bool = Field(default=False, description="Whether to use FP8 precision for the model inference, default is False which means using the original precision of the model")

    class Config:
        extra = "forbid"

class InterVLRequest(BaseModel):
    request_id: str
    prompt: str
    image_path: str # base64 encoded image 
    image_frame_idx: int = Field(default=0, description="The frame number to use for inference in case the input image is a video, default is 0 which means using the first frame")
    image_strength: float = Field(default=0.5, gt=0.0, lt=1.0, description="The strength of the image prompt, default is 0.5 which means giving equal importance to the image and text prompts, higher values mean giving more importance to the image prompt")
    output_path: str
    config: LTX2Config

class InternVLResponse(BaseModel):
    request_id: str
    answer: str