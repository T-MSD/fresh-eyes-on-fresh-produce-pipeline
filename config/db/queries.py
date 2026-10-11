from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Connection, Subquery, and_, case, func, select
from sqlalchemy.orm import Session, aliased, joinedload

from config.db.schema import Area, Camera, Frame, Prediction, Product, Run, Task, TaskKind

TASK_PRIORITY: dict[TaskKind, int] = {"restock": 1, "move_product": 2, "top_up": 3}


def latest_run_id(session: Session) -> int | None:
    return session.scalar(select(Run.id).order_by(Run.started_at.desc(), Run.id.desc()).limit(1))


def areas_for_frame(
    session: Session, camera_id: int, frame_id: int, run_id: int
) -> list[tuple[Area, Prediction | None]]:
    statement = (
        select(Area, Prediction)
        .outerjoin(
            Prediction,
            and_(
                Prediction.area_id == Area.id,
                Prediction.frame_id == frame_id,
                Prediction.run_id == run_id,
            ),
        )
        .options(joinedload(Area.product), joinedload(Prediction.detected_product))
        .where(Area.camera_id == camera_id)
        .order_by(Area.number)
    )
    return [(area, prediction) for area, prediction in session.execute(statement)]


@dataclass(frozen=True)
class HourRange:
    run_id: int
    start: datetime | None
    end: datetime | None


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


def latest_run() -> Subquery:
    return (
        select(Run.id)
        .where(Run.status == "done")
        .order_by(Run.started_at.desc(), Run.id.desc())
        .limit(1)
        .subquery()
    )


def get_hour_range(connection: Connection) -> HourRange | None:
    run = latest_run()
    statement = (
        select(run.c.id, func.min(Frame.ts), func.max(Frame.ts))
        .select_from(run)
        .outerjoin(Prediction, Prediction.run_id == run.c.id)
        .outerjoin(Frame, Frame.id == Prediction.frame_id)
        .group_by(run.c.id)
    )
    row = connection.execute(statement).one_or_none()
    if row is None:
        return None
    return HourRange(run_id=row[0], start=row[1], end=row[2])


def get_pending_tasks(connection: Connection) -> PendingTasks | None:
    """Pending tasks of the latest completed run, ordered by priority then opening time."""
    run_id = connection.scalar(select(latest_run().c.id))
    if run_id is None:
        return None
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
        .where(Task.run_id == run_id, Task.done_at.is_(None))
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
