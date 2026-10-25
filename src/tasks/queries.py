from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Connection, case, func, select, update
from sqlalchemy.orm import aliased

from config.db.schema import Area, Camera, Frame, Product, Task
from src.runs.queries import SelectedRun
from src.tasks.kind import TaskKind

TASK_PRIORITY: dict[TaskKind, int] = {"restock": 1, "move_product": 2, "top_up": 3}


@dataclass(frozen=True)
class TaskFrame:
    filename: str
    ts: datetime


@dataclass(frozen=True)
class PendingTask:
    camera: str
    area_id: str
    product_id: str
    product_name: str
    kind: TaskKind
    opened: TaskFrame
    closed: TaskFrame | None


@dataclass(frozen=True)
class PendingTasks:
    tasks: list[PendingTask]


def get_pending_tasks(connection: Connection, run: SelectedRun) -> PendingTasks:
    """Tasks of the run not marked done, ordered by priority then opening time."""
    opened, closed = aliased(Frame), aliased(Frame)
    statement = (
        select(
            Camera.external_id,
            Area.external_id,
            Product.external_id,
            Product.name,
            Task.kind,
            opened.filename,
            opened.ts,
            closed.filename,
            closed.ts,
        )
        .select_from(Task)
        .join(Area, Area.id == Task.area_id)
        .join(Camera, Camera.id == Area.camera_id)
        .join(Product, Product.id == Area.product_id)
        .join(opened, opened.id == Task.opened_frame_id)
        .outerjoin(closed, closed.id == Task.closed_frame_id)
        .where(Task.run_id == run.id, Task.done_at.is_(None))
        .order_by(
            case(*((Task.kind == kind, priority) for kind, priority in TASK_PRIORITY.items())),
            opened.ts,
            Camera.external_id,
            Area.number,
            Task.id,
        )
    )
    tasks = [
        PendingTask(
            camera=row[0],
            area_id=row[1],
            product_id=row[2],
            product_name=row[3],
            kind=row[4],
            opened=TaskFrame(filename=row[5], ts=row[6]),
            closed=TaskFrame(filename=row[7], ts=row[8]) if row[7] is not None else None,
        )
        for row in connection.execute(statement)
    ]
    return PendingTasks(tasks=tasks)


def mark_task_done(
    connection: Connection, run: SelectedRun, area_id: str, kind: TaskKind, opened_filename: str
) -> bool:
    """Mark the run's task done; a task already done keeps its first `done_at`.

    Returns whether the task exists.
    """
    task_id = connection.scalar(
        update(Task)
        .where(
            Task.run_id == run.id,
            Task.kind == kind,
            Task.area_id == Area.id,
            Area.dataset_id == run.dataset_id,
            Area.external_id == area_id,
            Task.opened_frame_id == Frame.id,
            Frame.filename == opened_filename,
        )
        .values(done_at=func.coalesce(Task.done_at, func.now()))
        .returning(Task.id)
    )
    return task_id is not None
