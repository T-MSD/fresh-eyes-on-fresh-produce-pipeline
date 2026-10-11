from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime, BaseModel
from sqlalchemy import Connection

from config.db.connection import get_connection
from config.db.queries import get_hour_range

router = APIRouter()


class HourRangeResponse(BaseModel):
    start: AwareDatetime | None
    end: AwareDatetime | None


@router.get("/hour-range", responses={404: {"description": "No completed runs available"}})
def hour_range(connection: Annotated[Connection, Depends(get_connection)]) -> HourRangeResponse:
    result = get_hour_range(connection)
    if result is None:
        raise HTTPException(status_code=404, detail="No completed runs available")
    return HourRangeResponse.model_validate(result, from_attributes=True)
