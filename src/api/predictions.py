from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import AwareDatetime, BaseModel, computed_field

from config.db.queries import AreaStatus, get_predictions
from config.db.schema import STOCK_VALUES, StockLevel
from src.api.dependencies import CompletedRun, DatabaseConnection

router = APIRouter()


class StoreAreaResponse(BaseModel):
    area_id: str
    product_id: str
    product_name: str
    detected_product_id: str | None
    status: AreaStatus
    planogram_compliant: bool
    stock_level: StockLevel
    compliance_confidence: float | None
    stock_confidence: float | None

    @computed_field
    @property
    def stock_value(self) -> float:
        return STOCK_VALUES[self.stock_level]


class StoreFrameResponse(BaseModel):
    filename: str
    ts: AwareDatetime


class StoreCameraResponse(BaseModel):
    camera: str
    frame: StoreFrameResponse
    areas: list[StoreAreaResponse]


class PredictionsResponse(BaseModel):
    cameras: list[StoreCameraResponse]


@router.get("/predictions", responses={404: {"description": "No completed runs available"}})
def predictions(
    connection: DatabaseConnection,
    run: CompletedRun,
    at: Annotated[AwareDatetime | None, Query()] = None,
    camera: Annotated[str | None, Query()] = None,
) -> PredictionsResponse:
    result = get_predictions(connection, run, at, camera)
    return PredictionsResponse.model_validate(result, from_attributes=True)
