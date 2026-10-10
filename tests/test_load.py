from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import Connection, func, insert, select
from sqlalchemy.exc import IntegrityError

from config.db.load import load_dataset
from config.db.schema import Area, Camera, Frame, Prediction, Product, Run
from config.db.schema import Dataset as DatasetLoad
from ingest.dataset import Dataset, read_dataset
from tests.builders import area, write_dataset


def dataset(root: Path, cameras, frames) -> Dataset:
    return read_dataset(*write_dataset(root, cameras, frames))


@pytest.fixture
def morning(tmp_path: Path) -> Dataset:
    return dataset(
        tmp_path / "morning",
        {"6215": [area("6215_1"), area("6215_2", "20923", "KIWI")], "6383": [area("6383_1")]},
        {"6215": [1751052241, 1751055841], "6383": [1751052241]},
    )


@pytest.fixture
def judge(tmp_path: Path) -> Dataset:
    return dataset(
        tmp_path / "judge",
        {"6215": [area("6215_1", "20923", "KIWI", top_left=[1, 2])]},
        {"6215": [1751100000]},
    )


def count(db: Connection, table) -> int:
    return db.scalar(select(func.count()).select_from(table))


def external_ids(db: Connection, table, dataset_id: int) -> list[str]:
    return list(
        db.scalars(
            select(table.external_id)
            .where(table.dataset_id == dataset_id)
            .order_by(table.external_id)
        )
    )


def frame_names(db: Connection, dataset_id: int) -> list[str]:
    return list(
        db.scalars(
            select(Frame.filename)
            .join(Frame.camera)
            .where(Camera.dataset_id == dataset_id)
            .order_by(Frame.filename)
        )
    )


def add_run_with_prediction(db: Connection, dataset_id: int) -> int:
    run_id = db.execute(
        insert(Run).values(dataset_id=dataset_id, method="reference").returning(Run.id)
    ).scalar_one()
    frame_id, camera_id = db.execute(
        select(Frame.id, Frame.camera_id)
        .join(Frame.camera)
        .where(Camera.dataset_id == dataset_id)
        .limit(1)
    ).one()
    area_id = db.scalar(select(Area.id).where(Area.camera_id == camera_id).limit(1))
    db.execute(
        insert(Prediction).values(
            run_id=run_id,
            frame_id=frame_id,
            area_id=area_id,
            dataset_id=dataset_id,
            camera_id=camera_id,
            planogram_compliant=True,
            stock_level="full",
        )
    )
    return run_id


def test_loads_cameras_products_areas_and_frames(db: Connection, morning: Dataset):
    report = load_dataset(db, morning)

    assert external_ids(db, Camera, report.dataset_id) == ["6215", "6383"]
    assert external_ids(db, Product, report.dataset_id) == ["11908", "20923"]
    assert external_ids(db, Area, report.dataset_id) == ["6215_1", "6215_2", "6383_1"]
    assert frame_names(db, report.dataset_id) == [
        "1751052241.jpg",
        "1751052241.jpg",
        "1751055841.jpg",
    ]
    assert (report.cameras, report.products, report.areas, report.frames) == (2, 2, 3, 3)


def test_links_areas_and_frames_to_generated_ids(db: Connection, morning: Dataset):
    report = load_dataset(db, morning)

    kiwi_area = db.execute(
        select(Camera.external_id, Area.number, Product.name, Area.top_left)
        .join(Area.camera)
        .join(Area.product)
        .where(Area.dataset_id == report.dataset_id, Area.external_id == "6215_2")
    ).one()
    assert tuple(kiwi_area) == ("6215", 2, "KIWI", [299, 350])
    frame_cameras = db.scalars(
        select(Camera.external_id).join(Frame.camera).where(Frame.filename == "1751052241.jpg")
    )
    assert sorted(frame_cameras) == ["6215", "6383"]


def test_records_the_dataset_load(db: Connection, morning: Dataset):
    report = load_dataset(db, morning)

    load = db.execute(select(DatasetLoad).where(DatasetLoad.id == report.dataset_id)).one()
    assert load.planogram_path == str(morning.planogram_path)
    assert load.images_dir == str(morning.images_dir)
    assert load.loaded_at == report.loaded_at


def test_second_load_reuses_external_ids_and_keeps_the_first(
    db: Connection, morning: Dataset, judge: Dataset
):
    first = load_dataset(db, morning)
    second = load_dataset(db, judge)

    assert second.dataset_id != first.dataset_id
    assert external_ids(db, Camera, second.dataset_id) == ["6215"]
    assert external_ids(db, Product, second.dataset_id) == ["20923"]
    assert db.execute(
        select(Area.external_id, Area.top_left).where(Area.dataset_id == second.dataset_id)
    ).all() == [("6215_1", [1, 2])]
    assert frame_names(db, second.dataset_id) == ["1751100000.jpg"]
    assert external_ids(db, Area, first.dataset_id) == ["6215_1", "6215_2", "6383_1"]
    assert len(frame_names(db, first.dataset_id)) == 3


def test_later_load_keeps_earlier_runs_and_predictions(
    db: Connection, morning: Dataset, judge: Dataset
):
    first = load_dataset(db, morning)
    run_id = add_run_with_prediction(db, first.dataset_id)

    load_dataset(db, judge)

    assert list(db.scalars(select(Run.id))) == [run_id]
    assert count(db, Prediction) == 1


def test_failed_load_leaves_no_trace(db: Connection, morning: Dataset, judge: Dataset):
    first = load_dataset(db, morning)
    broken = replace(judge, frames=judge.frames * 2)

    with pytest.raises(IntegrityError):
        load_dataset(db, broken)

    assert list(db.scalars(select(DatasetLoad.id))) == [first.dataset_id]
    assert count(db, Camera) == 2
    assert count(db, Area) == 3
    assert count(db, Frame) == 3


def test_stores_unreadable_frame_without_size(db: Connection, tmp_path: Path):
    planogram_path, images_dir = write_dataset(tmp_path, {"6215": [area("6215_1")]}, {})
    (images_dir / "6215").mkdir()
    (images_dir / "6215" / "1751052241.jpg").write_bytes(b"not an image")

    report = load_dataset(db, read_dataset(planogram_path, images_dir))

    assert db.execute(select(Frame.width, Frame.height)).one() == (None, None)
    assert report.unreadable_frames == 1
