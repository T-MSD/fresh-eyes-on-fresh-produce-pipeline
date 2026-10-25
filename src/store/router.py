from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import AwareDatetime

from src.api.dependencies import DatabaseConnection
from src.runs.router import CompletedRun
from src.store.models import (
    AreaStockResponse,
    CameraResponse,
    CamerasResponse,
    FrameAreaResponse,
    FrameResponse,
    PredictionsResponse,
)
from src.store.queries import (
    StoredFrame,
    get_area_stock,
    get_camera_frame,
    get_camera_frame_areas,
    get_cameras,
    get_predictions,
    get_stored_frame,
    has_camera,
)

router = APIRouter()


def stored_frame(
    camera: str, filename: str, connection: DatabaseConnection, run: CompletedRun
) -> StoredFrame:
    frame = get_stored_frame(connection, run, camera, filename)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found in the latest completed run")
    return frame


RequestedFrame = Annotated[StoredFrame, Depends(stored_frame)]


@router.get(
    "/cameras", tags=["frames"], responses={404: {"description": "No completed runs available"}}
)
def cameras(connection: DatabaseConnection, run: CompletedRun) -> CamerasResponse:
    return CamerasResponse(
        cameras=[
            CameraResponse.model_validate(camera, from_attributes=True)
            for camera in get_cameras(connection, run)
        ]
    )


@router.get(
    "/cameras/{camera}/frame",
    tags=["frames"],
    responses={404: {"description": "No completed run or frame"}},
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
    areas = [
        FrameAreaResponse.model_validate(area, from_attributes=True)
        for area in get_camera_frame_areas(connection, run, frame)
    ]
    return FrameResponse(
        camera=frame.camera,
        camera_name=frame.camera_name,
        filename=frame.filename,
        ts=frame.ts,
        width=frame.width,
        height=frame.height,
        areas=areas,
    )


@router.get(
    "/areas/{area_id}/stock",
    tags=["frames"],
    responses={404: {"description": "No completed run or area"}},
)
def area_stock(
    area_id: str, connection: DatabaseConnection, run: CompletedRun
) -> list[AreaStockResponse]:
    stock = get_area_stock(connection, run, area_id)
    if stock is None:
        raise HTTPException(status_code=404, detail="Area not found in the latest completed run")
    return [AreaStockResponse.model_validate(item, from_attributes=True) for item in stock]


@router.get(
    "/images/{camera}/{filename}",
    tags=["frames"],
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


@router.get("/predictions", responses={404: {"description": "No completed run or camera"}})
def predictions(
    connection: DatabaseConnection,
    run: CompletedRun,
    at: Annotated[AwareDatetime | None, Query()] = None,
    camera: Annotated[str | None, Query()] = None,
) -> PredictionsResponse:
    if camera is not None and not has_camera(connection, run, camera):
        raise HTTPException(status_code=404, detail="Camera not found in the latest completed run")
    result = get_predictions(connection, run, at, camera)
    return PredictionsResponse.model_validate(result, from_attributes=True)
