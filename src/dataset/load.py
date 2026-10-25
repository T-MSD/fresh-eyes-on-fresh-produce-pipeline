from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Connection, insert

from config.db.schema import Area, Camera, Frame, Product
from config.db.schema import Dataset as DatasetLoad
from src.dataset.read import Dataset


@dataclass(frozen=True)
class LoadReport:
    dataset_id: int
    cameras: int
    products: int
    areas: int
    frames: int
    unreadable_frames: int
    loaded_at: datetime


def load_dataset(conn: Connection, dataset: Dataset) -> LoadReport:
    """Add `dataset` as a new load, all or nothing; earlier loads and their runs stay."""
    with conn.begin_nested():
        dataset_id, loaded_at = conn.execute(
            insert(DatasetLoad)
            .values(planogram_path=str(dataset.planogram_path), images_dir=str(dataset.images_dir))
            .returning(DatasetLoad.id, DatasetLoad.loaded_at)
        ).one()

        planogram = dataset.planogram
        camera_ids = insert_returning_ids(
            conn,
            Camera,
            [{"dataset_id": dataset_id, "external_id": camera} for camera in planogram.cameras],
        )
        product_ids = insert_returning_ids(
            conn,
            Product,
            [
                {"dataset_id": dataset_id, "external_id": p.external_id, "name": p.name}
                for p in planogram.products
            ],
        )
        conn.execute(
            insert(Area),
            [
                {
                    "dataset_id": dataset_id,
                    "external_id": area.external_id,
                    "camera_id": camera_ids[area.camera],
                    "number": area.number,
                    "product_id": product_ids[area.product],
                    **{c: list(point) for c, point in area.corners.model_dump().items()},
                }
                for area in planogram.areas
            ],
        )
        conn.execute(
            insert(Frame),
            [
                {
                    "camera_id": camera_ids[frame.camera],
                    "filename": frame.filename,
                    "ts": frame.ts,
                    "width": frame.size[0] if frame.size else None,
                    "height": frame.size[1] if frame.size else None,
                }
                for frame in dataset.frames
            ],
        )

    return LoadReport(
        dataset_id=dataset_id,
        cameras=len(camera_ids),
        products=len(product_ids),
        areas=len(planogram.areas),
        frames=len(dataset.frames),
        unreadable_frames=sum(frame.size is None for frame in dataset.frames),
        loaded_at=loaded_at,
    )


def insert_returning_ids(
    conn: Connection, table: type[Camera] | type[Product], rows: list[dict[str, Any]]
) -> dict[str, int]:
    returned = conn.execute(insert(table).returning(table.external_id, table.id), rows)
    return {external_id: id_ for external_id, id_ in returned}
