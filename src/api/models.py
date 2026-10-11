from pydantic import AwareDatetime, BaseModel, computed_field

from config.db.schema import STOCK_VALUES, StockLevel


class CornersResponse(BaseModel):
    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_right: tuple[int, int]
    bottom_left: tuple[int, int]


class PredictionResponse(BaseModel):
    planogram_compliant: bool
    stock_level: StockLevel
    detected_product_id: str | None
    detected_product_name: str | None
    compliance_confidence: float | None
    stock_confidence: float | None

    @computed_field
    @property
    def stock_value(self) -> float:
        return STOCK_VALUES[self.stock_level]


class CameraAreaResponse(BaseModel):
    area_id: str
    number: int
    product_id: str
    product_name: str
    corners: CornersResponse


class CameraResponse(BaseModel):
    camera: str
    camera_name: str | None
    areas: list[CameraAreaResponse]


class CamerasResponse(BaseModel):
    cameras: list[CameraResponse]


class FrameAreaResponse(BaseModel):
    area_id: str
    number: int
    product_id: str
    product_name: str
    corners: CornersResponse
    prediction: PredictionResponse | None


class FrameResponse(BaseModel):
    camera: str
    camera_name: str | None
    filename: str
    ts: AwareDatetime
    width: int | None
    height: int | None
    areas: list[FrameAreaResponse]


class AreaStockResponse(BaseModel):
    ts: AwareDatetime
    stock_level: StockLevel
    stock_confidence: float | None

    @computed_field
    @property
    def stock_value(self) -> float:
        return STOCK_VALUES[self.stock_level]
