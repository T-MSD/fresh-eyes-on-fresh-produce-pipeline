from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FrameResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: int
    id: int
    camera_id: str
    filename: str
    ts: datetime
    width: int | None
    height: int | None
