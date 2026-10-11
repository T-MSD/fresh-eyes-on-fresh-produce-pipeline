import subprocess
from collections.abc import Collection
from dataclasses import asdict, dataclass, replace
from itertools import groupby
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter
from sqlalchemy import Connection, func, insert, select

from config.db.schema import Area, Camera, Frame, Prediction, Run, StockLevel, Task, TaskKind
from config.db.schema import Dataset as DatasetLoad

ROOT = Path(__file__).resolve().parent.parent.parent
SHOWN_PROBLEMS = 10


class AreaPrediction(BaseModel):
    planogram_compliant: bool
    stock_level: StockLevel


SubmissionShape = TypeAdapter(dict[str, dict[str, dict[str, AreaPrediction]]])


class UnusableRun(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


def fallback_keys(path: Path) -> set[tuple[str, str, str]]:
    """The (camera, frame, area) triples of a fallbacks.json file written by crop_areas."""
    return {
        (camera, frame, area)
        for camera, by_frame in SubmissionShape.validate_json(path.read_bytes()).items()
        for frame, by_area in by_frame.items()
        for area in by_area
    }


def git_sha() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return done.stdout.strip() or None


def latest_dataset_id(conn: Connection) -> int:
    dataset_id = conn.scalar(select(func.max(DatasetLoad.id)))
    if dataset_id is None:
        raise UnusableRun(["no dataset is loaded: run `cli.py load` first"])
    return dataset_id


def save_run(
    conn: Connection,
    dataset_id: int,
    predictions: dict[str, Any],
    *,
    method: str,
    params: dict[str, Any] | None = None,
    fallback: Collection[tuple[str, str, str]] = (),
) -> int:
    """Store one run, a prediction for every area of every frame and the run's tasks, all or
    nothing.

    `predictions` is the submission shape (camera -> frame -> area -> label). `fallback` holds the
    (camera, frame, area) triples that are a fallback, not a model answer. A run missing an area,
    or naming a frame or area the dataset does not have, is refused: a gap costs -5 points. Tasks
    come from `derive_tasks`.
    """
    parsed = SubmissionShape.validate_python(predictions)
    if not conn.scalar(select(DatasetLoad.id).where(DatasetLoad.id == dataset_id)):
        raise UnusableRun([f"dataset {dataset_id} does not exist"])

    camera_of = {
        id_: ext
        for id_, ext in conn.execute(
            select(Camera.id, Camera.external_id).where(Camera.dataset_id == dataset_id)
        )
    }
    areas = conn.execute(
        select(Area.id, Area.camera_id, Area.external_id).where(Area.dataset_id == dataset_id)
    ).all()
    frames = conn.execute(
        select(Frame.id, Frame.camera_id, Frame.filename)
        .join(Camera, Frame.camera_id == Camera.id)
        .where(Camera.dataset_id == dataset_id)
    ).all()

    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for frame_id, camera_id, filename in frames:
        camera = camera_of[camera_id]
        for area_id, area_camera_id, area_ext in areas:
            if area_camera_id != camera_id:
                continue
            rows[(camera, filename, area_ext)] = {
                "dataset_id": dataset_id,
                "camera_id": camera_id,
                "frame_id": frame_id,
                "area_id": area_id,
            }

    given = {
        (camera, frame, area): label
        for camera, by_frame in parsed.items()
        for frame, by_area in by_frame.items()
        for area, label in by_area.items()
    }
    problems = [f"missing {'/'.join(key)}" for key in sorted(rows.keys() - given.keys())]
    problems += [f"unknown {'/'.join(key)}" for key in sorted(given.keys() - rows.keys())]
    if problems:
        shown = problems[:SHOWN_PROBLEMS]
        if len(problems) > len(shown):
            shown.append(f"... and {len(problems) - len(shown)} more")
        raise UnusableRun([f"{len(problems)} area(s) do not match dataset {dataset_id}", *shown])

    with conn.begin_nested():
        run_id = conn.scalar(
            insert(Run)
            .values(
                dataset_id=dataset_id,
                method=method,
                git_sha=git_sha(),
                params=params or {},
                status="done",
                finished_at=func.now(),
            )
            .returning(Run.id)
        )
        conn.execute(
            insert(Prediction),
            [
                {
                    **row,
                    "run_id": run_id,
                    "planogram_compliant": given[key].planogram_compliant,
                    "stock_level": given[key].stock_level,
                    "is_fallback": key in fallback,
                }
                for key, row in rows.items()
            ],
        )
        derive_tasks(conn, run_id)
    return run_id


@dataclass(frozen=True)
class DerivedTask:
    area_id: int
    kind: TaskKind
    opened_frame_id: int
    closed_frame_id: int | None = None


def task_kind(stock_level: StockLevel, planogram_compliant: bool) -> TaskKind | None:
    if stock_level == "empty":
        return "restock"
    if not planogram_compliant:
        return "move_product"
    if stock_level == "half":
        return "top_up"
    return None


def derive_tasks(conn: Connection, run_id: int) -> int:
    """Store the tasks of a run without any, from its predictions; returns how many.

    Each area's frames are walked in time order. A task opens on the frame where its problem
    appears and closes on the first later frame showing a different problem or none; a different
    problem opens its own task on that frame. Fallback predictions neither open nor close a task.
    """
    if conn.scalar(select(Run.id).where(Run.id == run_id)) is None:
        raise UnusableRun([f"run {run_id} does not exist"])
    if conn.scalar(select(Task.id).where(Task.run_id == run_id).limit(1)) is not None:
        raise UnusableRun([f"run {run_id} already has tasks"])

    rows = conn.execute(
        select(
            Prediction.area_id,
            Prediction.frame_id,
            Prediction.stock_level,
            Prediction.planogram_compliant,
        )
        .join(Frame, Frame.id == Prediction.frame_id)
        .where(Prediction.run_id == run_id, Prediction.is_fallback.is_(False))
        .order_by(Prediction.area_id, Frame.ts, Frame.filename)
    ).all()

    tasks: list[DerivedTask] = []
    for area_id, frames in groupby(rows, key=lambda row: row.area_id):
        open_task: DerivedTask | None = None
        for row in frames:
            kind = task_kind(row.stock_level, row.planogram_compliant)
            if open_task is not None and open_task.kind == kind:
                continue
            if open_task is not None:
                tasks.append(replace(open_task, closed_frame_id=row.frame_id))
            open_task = DerivedTask(area_id, kind, row.frame_id) if kind is not None else None
        if open_task is not None:
            tasks.append(open_task)
    if tasks:
        conn.execute(insert(Task), [{"run_id": run_id, **asdict(task)} for task in tasks])
    return len(tasks)
