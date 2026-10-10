from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field

from config.db.schema import STOCK_VALUES, StockLevel


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
    frame_id: int
    camera_id: str
    filename: str
    ts: datetime
    areas: list[AreaResponse]
