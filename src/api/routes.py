from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import AwareDatetime

from config.db.queries import get_area_stock, get_camera_frame, get_camera_frame_areas, get_cameras
from src.api.dependencies import CompletedRun, DatabaseConnection, RequestedFrame
from src.api.models import (
    AreaStockResponse,
    CameraAreaResponse,
    CameraResponse,
    CamerasResponse,
    CornersResponse,
    FrameAreaResponse,
    FrameResponse,
    PredictionResponse,
)

router = APIRouter(tags=["frames"])


@router.get("/cameras", responses={404: {"description": "No completed runs available"}})
def cameras(connection: DatabaseConnection, run: CompletedRun) -> CamerasResponse:
    return CamerasResponse(
        cameras=[
            CameraResponse(
                camera=camera.camera,
                camera_name=camera.camera_name,
                areas=[
                    CameraAreaResponse(
                        area_id=area.area_id,
                        number=area.number,
                        product_id=area.product_id,
                        product_name=area.product_name,
                        corners=CornersResponse(
                            top_left=area.top_left,
                            top_right=area.top_right,
                            bottom_right=area.bottom_right,
                            bottom_left=area.bottom_left,
                        ),
                    )
                    for area in camera.areas
                ],
            )
            for camera in get_cameras(connection, run)
        ]
    )


@router.get(
    "/cameras/{camera}/frame", responses={404: {"description": "No completed run or frame"}}
)
def get_frame(
    camera: str,
    connection: DatabaseConnection,
    run: CompletedRun,
    at: Annotated[AwareDatetime | None, Query()] = None,
) -> FrameResponse:
    frame = get_camera_frame(connection, run, camera, at)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found in the latest completed run")
    areas = []
    for row in get_camera_frame_areas(connection, run, frame):
        prediction = None
        if row.prediction is not None:
            prediction = PredictionResponse.model_validate(row.prediction, from_attributes=True)
        areas.append(
            FrameAreaResponse(
                area_id=row.area_id,
                number=row.number,
                product_id=row.product_id,
                product_name=row.product_name,
                corners=CornersResponse(
                    top_left=row.top_left,
                    top_right=row.top_right,
                    bottom_right=row.bottom_right,
                    bottom_left=row.bottom_left,
                ),
                prediction=prediction,
            )
        )
    return FrameResponse(
        camera=frame.camera,
        camera_name=frame.camera_name,
        filename=frame.filename,
        ts=frame.ts,
        width=frame.width,
        height=frame.height,
        areas=areas,
    )


@router.get("/areas/{area_id}/stock", responses={404: {"description": "No completed run or area"}})
def area_stock(
    area_id: str, connection: DatabaseConnection, run: CompletedRun
) -> list[AreaStockResponse]:
    stock = get_area_stock(connection, run, area_id)
    if stock is None:
        raise HTTPException(status_code=404, detail="Area not found in the latest completed run")
    return [AreaStockResponse.model_validate(item, from_attributes=True) for item in stock]


@router.get(
    "/images/{camera}/{filename}",
    response_class=FileResponse,
    responses={
        200: {"content": {"image/jpeg": {}}},
        404: {"description": "No completed run, frame or image"},
    },
)
def get_image(frame: RequestedFrame) -> FileResponse:
    try:
        root = Path(frame.images_dir).resolve(strict=True)
        camera_dir = (root / frame.camera).resolve(strict=True)
        image = (camera_dir / frame.filename).resolve(strict=True)
        if (
            Path(frame.camera).name != frame.camera
            or Path(frame.filename).name != frame.filename
            or not camera_dir.is_relative_to(root)
            or not image.is_relative_to(camera_dir)
            or not image.is_file()
        ):
            raise ValueError("Invalid image path")
    except (OSError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=404, detail="Frame image not found") from error
    return FileResponse(image, media_type="image/jpeg")
