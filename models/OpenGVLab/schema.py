from pydantic import BaseModel, Field


class InterVLConfig(BaseModel):
    tp: int = Field(default=1, gt=0)
    chat_template: str | None = None
    session_len: int = Field(default=4096, gt=0)

    class Config:
        extra = "forbid"

class InterVLRequest(BaseModel):
    request_id: str
    prompt: str
    config: InterVLConfig

class InternVLResponse(BaseModel):
    request_id: str
    answer: str