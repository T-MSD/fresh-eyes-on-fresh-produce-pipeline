from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, computed_field
from sqlalchemy import Connection

from config.db.connection import get_connection
from config.db.queries import CameraNotFound, FrameNotFound, get_frame_areas
from config.db.schema import STOCK_VALUES, StockLevel

router = APIRouter()


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    planogram_compliant: bool
    stock_level: StockLevel
    detected_product_id: str | None
    compliance_confidence: float | None = Field(ge=0, le=1)
    stock_confidence: float | None = Field(ge=0, le=1)
    is_fallback: bool

    @computed_field
    @property
    def stock_value(self) -> float:
        return STOCK_VALUES[self.stock_level]


class AreaResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    area_id: str
    number: int
    product_id: str
    product_name: str
    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_right: tuple[int, int]
    bottom_left: tuple[int, int]
    prediction: PredictionResponse | None


class FrameAreasResponse(BaseModel):
    run_id: int
    camera_id: str
    filename: str
    ts: AwareDatetime
    areas: list[AreaResponse]


@router.get(
    "/cameras/{camera}/frames/{filename}/areas",
    responses={404: {"description": "No completed runs available, camera or frame not found"}},
)
def frame_areas(
    camera: str,
    filename: str,
    connection: Annotated[Connection, Depends(get_connection)],
) -> FrameAreasResponse:
    try:
        result = get_frame_areas(connection, camera, filename)
    except CameraNotFound as error:
        raise HTTPException(status_code=404, detail="Camera not found") from error
    except FrameNotFound as error:
        raise HTTPException(status_code=404, detail="Frame not found") from error
    if result is None:
        raise HTTPException(status_code=404, detail="No completed runs available")
    return FrameAreasResponse.model_validate(result, from_attributes=True)
