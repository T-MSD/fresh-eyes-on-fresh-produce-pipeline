from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Connection, insert, select

from config.db.schema import Area, Camera, Dataset, Frame, Prediction, Product, Run, Task
from tests.builders import CORNERS


def add_dataset(db: Connection, **overrides: Any) -> int:
    values = {"planogram_path": "/data/planogram.json", "images_dir": "/data/training", **overrides}
    return db.execute(insert(Dataset).values(**values).returning(Dataset.id)).scalar_one()


def add_camera(db: Connection, dataset_id: int, external_id: str) -> int:
    return db.execute(
        insert(Camera).values(dataset_id=dataset_id, external_id=external_id).returning(Camera.id)
    ).scalar_one()


def add_product(db: Connection, dataset_id: int, external_id: str = "11908") -> int:
    return db.execute(
        insert(Product)
        .values(dataset_id=dataset_id, external_id=external_id, name="BANANA CAT I CNT KG")
        .returning(Product.id)
    ).scalar_one()


def add_area(db: Connection, camera_id: int, number: int, **overrides: Any) -> int:
    dataset_id, camera = db.execute(
        select(Camera.dataset_id, Camera.external_id).where(Camera.id == camera_id)
    ).one()
    product_id = db.scalar(
        select(Product.id).where(Product.dataset_id == dataset_id, Product.external_id == "11908")
    )
    values = {
        "dataset_id": dataset_id,
        "external_id": f"{camera}_{number}",
        "camera_id": camera_id,
        "number": number,
        "product_id": product_id,
        **CORNERS,
        **overrides,
    }
    return db.execute(insert(Area).values(**values).returning(Area.id)).scalar_one()


def add_frame(db: Connection, camera_id: int, ts: int = 1751052241, **overrides: Any) -> int:
    values = {
        "camera_id": camera_id,
        "filename": f"{ts}.jpg",
        "ts": datetime.fromtimestamp(ts, UTC),
        "width": 1280,
        "height": 720,
        **overrides,
    }
    return db.execute(insert(Frame).values(**values).returning(Frame.id)).scalar_one()


def add_run(db: Connection, dataset_id: int, **overrides: Any) -> int:
    values = {"dataset_id": dataset_id, "method": "reference", **overrides}
    return db.execute(insert(Run).values(**values).returning(Run.id)).scalar_one()


def add_prediction(
    db: Connection, run_id: int, frame_id: int, area_id: int, camera_id: int, **overrides: Any
) -> None:
    values = {
        "run_id": run_id,
        "dataset_id": db.scalar(select(Camera.dataset_id).where(Camera.id == camera_id)),
        "frame_id": frame_id,
        "area_id": area_id,
        "camera_id": camera_id,
        "planogram_compliant": True,
        "stock_level": "full",
        "compliance_confidence": 0.9,
        "stock_confidence": 0.8,
        **overrides,
    }
    db.execute(insert(Prediction).values(**values))


def add_predicted_frame(
    db: Connection, run_id: int, area_id: int, camera_id: int, ts: int = 1751052241
) -> int:
    frame_id = add_frame(db, camera_id, ts)
    add_prediction(db, run_id, frame_id, area_id, camera_id)
    return frame_id


def add_task(
    db: Connection, run_id: int, area_id: int, opened_frame_id: int, **overrides: Any
) -> int:
    values = {
        "run_id": run_id,
        "area_id": area_id,
        "kind": "restock",
        "opened_frame_id": opened_frame_id,
        **overrides,
    }
    return db.execute(insert(Task).values(**values).returning(Task.id)).scalar_one()
