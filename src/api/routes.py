from typing import Annotated

from fastapi import APIRouter, HTTPException, Path

from config.db.queries import frame_for_prediction
from src.api.dependencies import DatabaseSession, MostRecentRun
from src.api.models import FrameResponse

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get(
    "/{frame_id}/{area_id}/frame",
    response_model=FrameResponse,
    responses={404: {"description": "No run or no matching prediction in the most recent run"}},
)
def get_prediction_frame(
    frame_id: Annotated[int, Path(gt=0)],
    area_id: Annotated[str, Path(min_length=1)],
    session: DatabaseSession,
    run_id: MostRecentRun,
) -> FrameResponse:
    frame = frame_for_prediction(session, run_id, frame_id, area_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="Prediction not found in the most recent run")
    return FrameResponse(
        run_id=run_id,
        id=frame.id,
        camera_id=frame.camera_id,
        filename=frame.filename,
        ts=frame.ts,
        width=frame.width,
        height=frame.height,
    )
