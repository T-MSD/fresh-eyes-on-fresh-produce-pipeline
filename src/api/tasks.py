from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime, BaseModel
from sqlalchemy import Connection

from config.db.connection import get_connection
from config.db.queries import TaskFrame, get_pending_tasks
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
    run_id: int
    tasks: list[TaskResponse]


@router.get("/tasks", responses={404: {"description": "No runs available"}})
def tasks(connection: Annotated[Connection, Depends(get_connection)]) -> TasksResponse:
    result = get_pending_tasks(connection)
    if result is None:
        raise HTTPException(status_code=404, detail="No runs available")
    return TasksResponse(
        run_id=result.run_id,
        tasks=[
            TaskResponse(
                camera=task.camera,
                area_id=task.area_id,
                product_id=task.product_id,
                product_name=task.product_name,
                kind=task.kind,
                opened=frame_response(task.opened),
                closed=frame_response(task.closed) if task.closed is not None else None,
            )
            for task in result.tasks
        ],
    )


def frame_response(frame: TaskFrame) -> TaskFrameResponse:
    return TaskFrameResponse(filename=frame.filename, ts=frame.ts)
