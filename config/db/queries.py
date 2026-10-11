from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Connection, Subquery, case, func, select
from sqlalchemy.orm import aliased

from config.db.schema import (
    Area,
    Camera,
    Dataset,
    Frame,
    Prediction,
    Product,
    Run,
    StockLevel,
    Task,
    TaskKind,
)

TASK_PRIORITY: dict[TaskKind, int] = {"restock": 1, "move_product": 2, "top_up": 3}


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


@dataclass(frozen=True)
class SelectedRun:
    id: int
    dataset_id: int


def get_latest_run(connection: Connection) -> SelectedRun | None:
    latest = latest_run()
    row = connection.execute(
        select(Run.id, Run.dataset_id).join(latest, Run.id == latest.c.id)
    ).one_or_none()
    return SelectedRun(*row) if row is not None else None


@dataclass(frozen=True)
class StoredFrame:
    id: int
    camera_id: int
    camera: str
    camera_name: str | None
    filename: str
    ts: datetime
    width: int | None
    height: int | None
    images_dir: str


def get_stored_frame(
    connection: Connection, run: SelectedRun, camera: str, filename: str
) -> StoredFrame | None:
    row = connection.execute(
        select(
            Frame.id,
            Camera.id,
            Camera.external_id,
            Camera.name,
            Frame.filename,
            Frame.ts,
            Frame.width,
            Frame.height,
            Dataset.images_dir,
        )
        .join(Camera, Camera.id == Frame.camera_id)
        .join(Dataset, Dataset.id == Camera.dataset_id)
        .where(
            Camera.dataset_id == run.dataset_id,
            Camera.external_id == camera,
            Frame.filename == filename,
        )
    ).one_or_none()
    return StoredFrame(*row) if row is not None else None


def get_camera_frame(
    connection: Connection, run: SelectedRun, camera: str, at: datetime | None
) -> StoredFrame | None:
    statement = (
        select(Frame.filename)
        .join(Camera, Camera.id == Frame.camera_id)
        .where(Camera.dataset_id == run.dataset_id, Camera.external_id == camera)
        .order_by(Frame.ts.desc(), Frame.filename.desc())
        .limit(1)
    )
    if at is not None:
        statement = statement.where(Frame.ts <= at)
    filename = connection.scalar(statement)
    return get_stored_frame(connection, run, camera, filename) if filename is not None else None


@dataclass(frozen=True)
class AreaPrediction:
    planogram_compliant: bool
    stock_level: StockLevel
    detected_product_id: str | None
    detected_product_name: str | None
    compliance_confidence: float | None
    stock_confidence: float | None


@dataclass(frozen=True)
class FrameArea:
    area_id: str
    number: int
    product_id: str
    product_name: str
    top_left: list[int]
    top_right: list[int]
    bottom_right: list[int]
    bottom_left: list[int]
    prediction: AreaPrediction | None


def get_frame_areas(
    connection: Connection, run: SelectedRun, frame: StoredFrame
) -> list[FrameArea]:
    detected = aliased(Product)
    rows = connection.execute(
        select(
            Area.external_id.label("area_id"),
            Area.number,
            Product.external_id.label("product_id"),
            Product.name.label("product_name"),
            Area.top_left,
            Area.top_right,
            Area.bottom_right,
            Area.bottom_left,
            Prediction.planogram_compliant,
            Prediction.stock_level,
            detected.external_id.label("detected_product_id"),
            detected.name.label("detected_product_name"),
            Prediction.compliance_confidence,
            Prediction.stock_confidence,
        )
        .join(Product, Product.id == Area.product_id)
        .outerjoin(
            Prediction,
            (Prediction.area_id == Area.id)
            & (Prediction.frame_id == frame.id)
            & (Prediction.run_id == run.id),
        )
        .outerjoin(detected, detected.id == Prediction.detected_product_id)
        .where(Area.camera_id == frame.camera_id, Area.dataset_id == run.dataset_id)
        .order_by(Area.number, Area.external_id)
    ).all()
    return [
        FrameArea(
            area_id=row.area_id,
            number=row.number,
            product_id=row.product_id,
            product_name=row.product_name,
            top_left=row.top_left,
            top_right=row.top_right,
            bottom_right=row.bottom_right,
            bottom_left=row.bottom_left,
            prediction=AreaPrediction(
                planogram_compliant=row.planogram_compliant,
                stock_level=row.stock_level,
                detected_product_id=row.detected_product_id,
                detected_product_name=row.detected_product_name,
                compliance_confidence=row.compliance_confidence,
                stock_confidence=row.stock_confidence,
            )
            if row.stock_level is not None
            else None,
        )
        for row in rows
    ]


@dataclass(frozen=True)
class AreaStock:
    ts: datetime
    stock_level: StockLevel
    stock_confidence: float | None


def get_area_stock(
    connection: Connection, run: SelectedRun, area_id: str
) -> list[AreaStock] | None:
    area = connection.scalar(
        select(Area.id).where(Area.dataset_id == run.dataset_id, Area.external_id == area_id)
    )
    if area is None:
        return None
    rows = connection.execute(
        select(Frame.ts, Prediction.stock_level, Prediction.stock_confidence)
        .join(Prediction, Prediction.frame_id == Frame.id)
        .where(Prediction.area_id == area, Prediction.run_id == run.id)
        .order_by(Frame.ts, Frame.filename)
    )
    return [AreaStock(*row) for row in rows]


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
