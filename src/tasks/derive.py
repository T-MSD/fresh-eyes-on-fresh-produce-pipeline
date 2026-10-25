from dataclasses import asdict, dataclass, replace
from itertools import groupby

from sqlalchemy import Connection, insert, select

from config.db.schema import Frame, Prediction, Run, Task
from src.runs.errors import UnusableRun
from src.tasks.kind import TaskKind, task_kind


@dataclass(frozen=True)
class DerivedTask:
    area_id: int
    kind: TaskKind
    opened_frame_id: int
    closed_frame_id: int | None = None


def steady_kinds(kinds: list[TaskKind | None]) -> list[TaskKind | None]:
    """`kinds` with each one-frame change that the next frame undoes replaced by the kind before
    it, so one misread frame neither opens nor closes a task. The first and last kinds stay."""
    steady = list(kinds)
    for i in range(1, len(kinds) - 1):
        if steady[i - 1] == kinds[i + 1] != kinds[i]:
            steady[i] = steady[i - 1]
    return steady


def derive_tasks(conn: Connection, run_id: int) -> int:
    """Store the tasks of a run without any, from its predictions; returns how many.

    Each area's frames are walked in time order, skipping fallback predictions. A frame whose
    problem differs from both the previous frame (after this rule) and the next frame, when those
    two agree, takes their problem. A task opens on the frame where its problem appears and closes
    on the first later frame showing a different problem or none; a different problem opens its
    own task on that frame.
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
    for area_id, area_rows in groupby(rows, key=lambda row: row.area_id):
        frames = list(area_rows)
        kinds = steady_kinds(
            [task_kind(row.stock_level, row.planogram_compliant) for row in frames]
        )
        open_task: DerivedTask | None = None
        for row, kind in zip(frames, kinds, strict=True):
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
