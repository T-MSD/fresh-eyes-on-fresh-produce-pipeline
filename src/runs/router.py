from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime, BaseModel

from src.api.dependencies import DatabaseConnection
from src.runs.queries import SelectedRun, get_hour_range, get_latest_run

router = APIRouter()


def completed_run(connection: DatabaseConnection) -> SelectedRun:
    run = get_latest_run(connection)
    if run is None:
        raise HTTPException(status_code=404, detail="No completed runs available")
    return run


CompletedRun = Annotated[SelectedRun, Depends(completed_run)]


class HourRangeResponse(BaseModel):
    start: AwareDatetime | None
    end: AwareDatetime | None


@router.get("/hour-range", responses={404: {"description": "No completed runs available"}})
def hour_range(connection: DatabaseConnection, run: CompletedRun) -> HourRangeResponse:
    return HourRangeResponse.model_validate(get_hour_range(connection, run), from_attributes=True)
