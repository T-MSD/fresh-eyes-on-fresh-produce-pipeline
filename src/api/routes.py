from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from config.db.queries import areas_for_frame
from config.db.schema import Frame
from src.api.dependencies import DatabaseSession, LatestRunId
from src.api.models import AreaResponse, FrameAreasResponse, PredictionResponse

router = APIRouter()


def get_frame(frame_id: Annotated[int, Path(gt=0)], session: DatabaseSession) -> Frame:
    frame = session.get(Frame, frame_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    return frame


@router.get("/frames/{frame_id}/areas", response_model=FrameAreasResponse)
def get_frame_areas(
    frame: Annotated[Frame, Depends(get_frame)],
    session: DatabaseSession,
    run_id: LatestRunId,
) -> FrameAreasResponse:
    if run_id is None:
        raise HTTPException(status_code=503, detail="No runs available")
    areas = [
        AreaResponse(
            area_id=area.external_id,
            number=area.number,
            product_id=area.product.external_id,
            product_name=area.product.name,
            top_left=tuple(area.top_left),
            top_right=tuple(area.top_right),
            bottom_right=tuple(area.bottom_right),
            bottom_left=tuple(area.bottom_left),
            prediction=PredictionResponse(
                planogram_compliant=prediction.planogram_compliant,
                stock_level=prediction.stock_level,
                detected_product_id=(
                    prediction.detected_product.external_id if prediction.detected_product else None
                ),
                compliance_confidence=prediction.compliance_confidence,
                stock_confidence=prediction.stock_confidence,
                is_fallback=prediction.is_fallback,
            )
            if prediction
            else None,
        )
        for area, prediction in areas_for_frame(session, frame.camera_id, frame.id, run_id)
    ]
    return FrameAreasResponse(
        run_id=run_id,
        frame_id=frame.id,
        camera_id=frame.camera.external_id,
        filename=frame.filename,
        ts=frame.ts,
        areas=areas,
    )
