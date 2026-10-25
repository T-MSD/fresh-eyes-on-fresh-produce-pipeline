from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Connection, Subquery, func, select

from config.db.schema import Frame, Prediction, Run


@dataclass(frozen=True)
class SelectedRun:
    id: int
    dataset_id: int


@dataclass(frozen=True)
class HourRange:
    start: datetime | None
    end: datetime | None


def latest_run() -> Subquery:
    return (
        select(Run.id)
        .where(Run.status == "done")
        .order_by(Run.started_at.desc(), Run.id.desc())
        .limit(1)
        .subquery()
    )


def get_latest_run(connection: Connection) -> SelectedRun | None:
    latest = latest_run()
    row = connection.execute(
        select(Run.id, Run.dataset_id).join(latest, Run.id == latest.c.id)
    ).one_or_none()
    return SelectedRun(*row) if row is not None else None


def get_hour_range(connection: Connection, run: SelectedRun) -> HourRange:
    start, end = connection.execute(
        select(func.min(Frame.ts), func.max(Frame.ts))
        .join(Prediction, Prediction.frame_id == Frame.id)
        .where(Prediction.run_id == run.id)
    ).one()
    return HourRange(start=start, end=end)
