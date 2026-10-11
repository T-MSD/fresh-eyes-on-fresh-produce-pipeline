import subprocess
from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter
from sqlalchemy import Connection, func, insert, select

from config.db.schema import Area, Camera, Frame, Prediction, Run, StockLevel
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
    """Store one run and a prediction for every area of every frame, all or nothing.

    `predictions` is the submission shape (camera -> frame -> area -> label). `fallback` holds the
    (camera, frame, area) triples that are a fallback, not a model answer. A run missing an area,
    or naming a frame or area the dataset does not have, is refused: a gap costs -5 points.
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
    return run_id
