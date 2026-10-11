from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import AwareDatetime, BaseModel

from config.db.queries import AreaStatus, get_predictions, has_camera
from src.api.dependencies import CompletedRun, DatabaseConnection
from src.api.models import PredictionResponse

router = APIRouter()


class StoreAreaResponse(BaseModel):
    area_id: str
    product_id: str
    product_name: str
    status: AreaStatus
    prediction: PredictionResponse | None


class StoreFrameResponse(BaseModel):
    filename: str
    ts: AwareDatetime


class StoreCameraResponse(BaseModel):
    camera: str
    frame: StoreFrameResponse
    areas: list[StoreAreaResponse]


class PredictionsResponse(BaseModel):
    cameras: list[StoreCameraResponse]


@router.get("/predictions", responses={404: {"description": "No completed run or camera"}})
def predictions(
    connection: DatabaseConnection,
    run: CompletedRun,
    at: Annotated[AwareDatetime | None, Query()] = None,
    camera: Annotated[str | None, Query()] = None,
) -> PredictionsResponse:
    if camera is not None and not has_camera(connection, run, camera):
        raise HTTPException(status_code=404, detail="Camera not found in the latest completed run")
    result = get_predictions(connection, run, at, camera)
    return PredictionsResponse.model_validate(result, from_attributes=True)
