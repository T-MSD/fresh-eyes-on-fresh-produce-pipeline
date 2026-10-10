from sqlalchemy import and_, select
from sqlalchemy.orm import Session, joinedload

from config.db.schema import Area, Prediction, Run


def latest_run_id(session: Session) -> int | None:
    return session.scalar(select(Run.id).order_by(Run.started_at.desc(), Run.id.desc()).limit(1))


def areas_for_frame(
    session: Session, camera_id: str, frame_id: int, run_id: int
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
        .options(joinedload(Area.product))
        .where(Area.camera_id == camera_id)
        .order_by(Area.number)
    )
    return [(area, prediction) for area, prediction in session.execute(statement)]
