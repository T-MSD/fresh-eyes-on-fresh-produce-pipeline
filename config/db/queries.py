from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Connection, Subquery, and_, case, func, select
from sqlalchemy.orm import aliased

from config.db.schema import (
    Area,
    Camera,
    Frame,
    Prediction,
    Product,
    Run,
    StockLevel,
    Task,
    TaskKind,
)

TASK_PRIORITY: dict[TaskKind, int] = {"restock": 1, "move_product": 2, "top_up": 3}


class CameraNotFound(LookupError):
    pass


class FrameNotFound(LookupError):
    pass


@dataclass(frozen=True)
class FrameAreaPrediction:
    planogram_compliant: bool
    stock_level: StockLevel
    detected_product_id: str | None
    compliance_confidence: float | None
    stock_confidence: float | None
    is_fallback: bool


@dataclass(frozen=True)
class FrameArea:
    area_id: str
    number: int
    product_id: str
    product_name: str
    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_right: tuple[int, int]
    bottom_left: tuple[int, int]
    prediction: FrameAreaPrediction | None


@dataclass(frozen=True)
class FrameAreas:
    run_id: int
    camera_id: str
    filename: str
    ts: datetime
    areas: list[FrameArea]


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


def get_frame_areas(connection: Connection, camera: str, filename: str) -> FrameAreas | None:
    selected_run = latest_run()
    run = connection.execute(
        select(Run.id, Run.dataset_id).join(selected_run, selected_run.c.id == Run.id)
    ).one_or_none()
    if run is None:
        return None
    camera_id = connection.scalar(
        select(Camera.id).where(Camera.dataset_id == run.dataset_id, Camera.external_id == camera)
    )
    if camera_id is None:
        raise CameraNotFound
    frame = connection.execute(
        select(Frame.id, Frame.filename, Frame.ts).where(
            Frame.camera_id == camera_id, Frame.filename == filename
        )
    ).one_or_none()
    if frame is None:
        raise FrameNotFound
    detected = aliased(Product)
    statement = (
        select(
            Area.external_id.label("area_id"),
            Area.number,
            Product.external_id.label("product_id"),
            Product.name.label("product_name"),
            Area.top_left,
            Area.top_right,
            Area.bottom_right,
            Area.bottom_left,
            Prediction.id.label("prediction_id"),
            Prediction.planogram_compliant,
            Prediction.stock_level,
            detected.external_id.label("detected_product_id"),
            Prediction.compliance_confidence,
            Prediction.stock_confidence,
            Prediction.is_fallback,
        )
        .select_from(Area)
        .join(Product, Product.id == Area.product_id)
        .outerjoin(
            Prediction,
            and_(
                Prediction.area_id == Area.id,
                Prediction.frame_id == frame.id,
                Prediction.run_id == run.id,
            ),
        )
        .outerjoin(detected, detected.id == Prediction.detected_product_id)
        .where(Area.camera_id == camera_id)
        .order_by(Area.number)
    )
    areas = [
        FrameArea(
            area_id=row.area_id,
            number=row.number,
            product_id=row.product_id,
            product_name=row.product_name,
            top_left=tuple(row.top_left),
            top_right=tuple(row.top_right),
            bottom_right=tuple(row.bottom_right),
            bottom_left=tuple(row.bottom_left),
            prediction=FrameAreaPrediction(
                planogram_compliant=row.planogram_compliant,
                stock_level=row.stock_level,
                detected_product_id=row.detected_product_id,
                compliance_confidence=row.compliance_confidence,
                stock_confidence=row.stock_confidence,
                is_fallback=row.is_fallback,
            )
            if row.prediction_id is not None
            else None,
        )
        for row in connection.execute(statement)
    ]
    return FrameAreas(
        run_id=run.id, camera_id=camera, filename=frame.filename, ts=frame.ts, areas=areas
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
