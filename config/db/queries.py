from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import Connection, Row, Subquery, case, func, select, update
from sqlalchemy.dialects.postgresql import distinct_on
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
from ingest.planogram import Corners

AreaStatus = Literal["empty", "wrong_product", "half", "ok", "unknown"]

TASK_PRIORITY: dict[TaskKind, int] = {"restock": 1, "move_product": 2, "top_up": 3}
URGENT: tuple[AreaStatus, ...] = ("empty", "wrong_product", "half", "unknown")


@dataclass(frozen=True)
class HourRange:
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


def latest_frames(run: SelectedRun, at: datetime | None, camera: str | None = None) -> Subquery:
    """One row per camera of the run's dataset: its newest frame at or before `at`.

    Every frame of the dataset is eligible, with or without predictions in the run. Frames tied
    on time resolve to the highest filename.
    """
    statement = (
        select(Frame.camera_id, Frame.id.label("frame_id"))
        .join(Camera, Camera.id == Frame.camera_id)
        .where(Camera.dataset_id == run.dataset_id)
        .ext(distinct_on(Frame.camera_id))
        .order_by(Frame.camera_id, Frame.ts.desc(), Frame.filename.desc())
    )
    if at is not None:
        statement = statement.where(Frame.ts <= at)
    if camera is not None:
        statement = statement.where(Camera.external_id == camera)
    return statement.subquery()


def get_camera_frame(
    connection: Connection, run: SelectedRun, camera: str, at: datetime | None
) -> StoredFrame | None:
    latest = latest_frames(run, at, camera)
    filename = connection.scalar(select(Frame.filename).join(latest, latest.c.frame_id == Frame.id))
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
class CameraArea:
    area_id: str
    number: int
    product_id: str
    product_name: str
    corners: Corners


@dataclass(frozen=True)
class CameraFrameArea(CameraArea):
    prediction: AreaPrediction | None


def area_corners(row: Row[Any]) -> Corners:
    return Corners(
        top_left=row.top_left,
        top_right=row.top_right,
        bottom_right=row.bottom_right,
        bottom_left=row.bottom_left,
    )


def get_camera_frame_areas(
    connection: Connection, run: SelectedRun, frame: StoredFrame
) -> list[CameraFrameArea]:
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
        CameraFrameArea(
            area_id=row.area_id,
            number=row.number,
            product_id=row.product_id,
            product_name=row.product_name,
            corners=area_corners(row),
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
        select(func.min(Frame.ts), func.max(Frame.ts))
        .select_from(run)
        .outerjoin(Prediction, Prediction.run_id == run.c.id)
        .outerjoin(Frame, Frame.id == Prediction.frame_id)
        .group_by(run.c.id)
    )
    row = connection.execute(statement).one_or_none()
    if row is None:
        return None
    return HourRange(start=row[0], end=row[1])


@dataclass(frozen=True)
class StoreArea:
    area_id: str
    product_id: str
    product_name: str
    status: AreaStatus
    prediction: AreaPrediction | None


@dataclass(frozen=True)
class StoreFrame:
    filename: str
    ts: datetime


@dataclass(frozen=True)
class StoreCamera:
    camera: str
    frame: StoreFrame
    areas: list[StoreArea]

    @property
    def urgency(self) -> tuple[int, ...]:
        return tuple(-sum(area.status == status for area in self.areas) for status in URGENT)


@dataclass(frozen=True)
class StorePredictions:
    cameras: list[StoreCamera]


@dataclass(frozen=True)
class PlanogramCamera:
    camera: str
    camera_name: str | None
    areas: list[CameraArea]


def get_cameras(connection: Connection, run: SelectedRun) -> list[PlanogramCamera]:
    """Every camera of the run's dataset by camera id, each with its areas by area number."""
    areas: dict[int, list[CameraArea]] = {}
    for row in connection.execute(
        select(
            Area.camera_id,
            Area.external_id.label("area_id"),
            Area.number,
            Product.external_id.label("product_id"),
            Product.name.label("product_name"),
            Area.top_left,
            Area.top_right,
            Area.bottom_right,
            Area.bottom_left,
        )
        .join(Product, Product.id == Area.product_id)
        .where(Area.dataset_id == run.dataset_id)
        .order_by(Area.number, Area.external_id)
    ):
        areas.setdefault(row.camera_id, []).append(
            CameraArea(
                area_id=row.area_id,
                number=row.number,
                product_id=row.product_id,
                product_name=row.product_name,
                corners=area_corners(row),
            )
        )
    cameras = connection.execute(
        select(Camera.id, Camera.external_id, Camera.name)
        .where(Camera.dataset_id == run.dataset_id)
        .order_by(Camera.external_id)
    )
    return [
        PlanogramCamera(camera=external_id, camera_name=name, areas=areas.get(camera_id, []))
        for camera_id, external_id, name in cameras
    ]


def has_camera(connection: Connection, run: SelectedRun, camera: str) -> bool:
    return (
        connection.scalar(
            select(Camera.id).where(
                Camera.dataset_id == run.dataset_id, Camera.external_id == camera
            )
        )
        is not None
    )


def get_predictions(
    connection: Connection,
    run: SelectedRun,
    at: datetime | None = None,
    camera: str | None = None,
) -> StorePredictions:
    """Every area of each camera's `latest_frames` frame, most urgent camera first.

    A camera without a frame by `at` is left out. An area is `unknown` without a prediction in
    the run, else `empty`, else `wrong_product` when not compliant, else `half`, else `ok`.
    Cameras with more empty areas come first, then more wrong products, then more half areas,
    then more unknown areas, then by camera id. `camera` keeps only that camera.
    """
    latest = latest_frames(run, at, camera)
    detected = aliased(Product)
    status = case(
        (Prediction.id.is_(None), "unknown"),
        (Prediction.stock_level == "empty", "empty"),
        (Prediction.planogram_compliant.is_(False), "wrong_product"),
        (Prediction.stock_level == "half", "half"),
        else_="ok",
    )
    rows = connection.execute(
        select(
            Camera.external_id.label("camera"),
            Frame.filename,
            Frame.ts,
            Area.external_id.label("area_id"),
            Product.external_id.label("product_id"),
            Product.name.label("product_name"),
            status.label("status"),
            Prediction.planogram_compliant,
            Prediction.stock_level,
            detected.external_id.label("detected_product_id"),
            detected.name.label("detected_product_name"),
            Prediction.compliance_confidence,
            Prediction.stock_confidence,
        )
        .select_from(Area)
        .join(latest, latest.c.camera_id == Area.camera_id)
        .join(Frame, Frame.id == latest.c.frame_id)
        .join(Camera, Camera.id == Area.camera_id)
        .join(Product, Product.id == Area.product_id)
        .outerjoin(
            Prediction,
            (Prediction.area_id == Area.id)
            & (Prediction.frame_id == Frame.id)
            & (Prediction.run_id == run.id),
        )
        .outerjoin(detected, detected.id == Prediction.detected_product_id)
        .order_by(Camera.external_id, Area.number, Area.external_id)
    )
    cameras: dict[str, StoreCamera] = {}
    for row in rows:
        if row.camera not in cameras:
            cameras[row.camera] = StoreCamera(
                camera=row.camera, frame=StoreFrame(filename=row.filename, ts=row.ts), areas=[]
            )
        cameras[row.camera].areas.append(
            StoreArea(
                area_id=row.area_id,
                product_id=row.product_id,
                product_name=row.product_name,
                status=row.status,
                prediction=AreaPrediction(
                    planogram_compliant=row.planogram_compliant,
                    stock_level=row.stock_level,
                    detected_product_id=row.detected_product_id,
                    detected_product_name=row.detected_product_name,
                    compliance_confidence=row.compliance_confidence,
                    stock_confidence=row.stock_confidence,
                )
                if row.status != "unknown"
                else None,
            )
        )
    return StorePredictions(cameras=sorted(cameras.values(), key=lambda c: c.urgency))


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
