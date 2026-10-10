from sqlalchemy import select
from sqlalchemy.orm import Session

from config.db.schema import Frame, Prediction, Run


def latest_run_id(session: Session) -> int | None:
    return session.scalar(select(Run.id).order_by(Run.started_at.desc(), Run.id.desc()).limit(1))


def frame_for_prediction(
    session: Session, run_id: int, frame_id: int, area_id: str
) -> Frame | None:
    return session.scalar(
        select(Frame)
        .join(
            Prediction,
            (Prediction.frame_id == Frame.id) & (Prediction.camera_id == Frame.camera_id),
        )
        .where(
            Prediction.run_id == run_id,
            Prediction.frame_id == frame_id,
            Prediction.area_id == area_id,
        )
    )
