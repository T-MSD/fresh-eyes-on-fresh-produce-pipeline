from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime, BaseModel
from sqlalchemy import Connection

from config.db.connection import get_connection
from config.db.queries import get_pending_tasks
from config.db.schema import TaskKind

router = APIRouter()


class TaskFrameResponse(BaseModel):
    filename: str
    ts: AwareDatetime


class TaskResponse(BaseModel):
    camera: str
    area_id: str
    product_id: str
    product_name: str
    kind: TaskKind
    opened: TaskFrameResponse
    closed: TaskFrameResponse | None


class TasksResponse(BaseModel):
    tasks: list[TaskResponse]


@router.get("/tasks", responses={404: {"description": "No completed runs available"}})
def tasks(connection: Annotated[Connection, Depends(get_connection)]) -> TasksResponse:
    result = get_pending_tasks(connection)
    if result is None:
        raise HTTPException(status_code=404, detail="No completed runs available")
    return TasksResponse.model_validate(result, from_attributes=True)
