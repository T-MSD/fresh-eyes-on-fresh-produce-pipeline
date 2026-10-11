from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime, BaseModel
from sqlalchemy import Connection

from config.db.connection import get_connection
from config.db.queries import get_latest_run, get_pending_tasks, mark_task_done
from config.db.schema import TaskKind
from src.api.dependencies import Transaction

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


class TaskDoneRequest(BaseModel):
    area_id: str
    kind: TaskKind
    opened_filename: str


@router.get("/tasks", responses={404: {"description": "No completed runs available"}})
def tasks(connection: Annotated[Connection, Depends(get_connection)]) -> TasksResponse:
    result = get_pending_tasks(connection)
    if result is None:
        raise HTTPException(status_code=404, detail="No completed runs available")
    return TasksResponse.model_validate(result, from_attributes=True)


@router.post(
    "/tasks/done",
    status_code=204,
    responses={404: {"description": "No completed run or task"}},
)
def mark_done(task: TaskDoneRequest, connection: Transaction) -> None:
    run = get_latest_run(connection)
    if run is None:
        raise HTTPException(status_code=404, detail="No completed runs available")
    if not mark_task_done(connection, run, task.area_id, task.kind, task.opened_filename):
        raise HTTPException(status_code=404, detail="Task not found in the latest completed run")
