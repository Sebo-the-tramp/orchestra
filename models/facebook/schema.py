from pydantic import BaseModel, Field


class Sam3Config(BaseModel):    

    class Config:
        extra = "forbid"

class Sam3Request(BaseModel):
    request_id: str
    prompt: str
    confidence_threshold: float = Field(default=0.93, gt=0.0, lt=1.0)
    config: Sam3Config

class Sam3Response(BaseModel):
    request_id: str
    answer: str