from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from alembic import command
from sqlalchemy import Connection, Engine, delete, func, insert, select
from sqlalchemy.exc import IntegrityError

from config.db.schema import Area, Camera, Frame, Prediction, Product, Run
from tests.conftest import alembic_config

CORNERS = {
    "top_left": [299, 350],
    "top_right": [464, 347],
    "bottom_right": [479, 438],
    "bottom_left": [331, 438],
}


def add_camera(db: Connection, camera_id: str) -> str:
    db.execute(insert(Camera).values(id=camera_id))
    return camera_id


def add_product(db: Connection, product_id: str = "11908") -> str:
    db.execute(insert(Product).values(id=product_id, name="BANANA CAT I CNT KG"))
    return product_id


def add_area(db: Connection, camera_id: str, number: int, **overrides: Any) -> str:
    area_id = f"{camera_id}_{number}"
    values = {
        "id": area_id,
        "camera_id": camera_id,
        "number": number,
        "product_id": "11908",
        **CORNERS,
        **overrides,
    }
    db.execute(insert(Area).values(**values))
    return area_id


def add_frame(db: Connection, camera_id: str, ts: int = 1751052241, **overrides: Any) -> int:
    values = {
        "camera_id": camera_id,
        "filename": f"{ts}.jpg",
        "ts": datetime.fromtimestamp(ts, UTC),
        "width": 1280,
        "height": 720,
        **overrides,
    }
    return db.execute(insert(Frame).values(**values).returning(Frame.id)).scalar_one()


def add_run(db: Connection, **overrides: Any) -> int:
    values = {"method": "reference", **overrides}
    return db.execute(insert(Run).values(**values).returning(Run.id)).scalar_one()


def add_prediction(
    db: Connection, run_id: int, frame_id: int, area_id: str, camera_id: str, **overrides: Any
) -> None:
    values = {
        "run_id": run_id,
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


def assert_rejected(
    db: Connection, by: str | set[str], add: Any, *args: Any, **kwargs: Any
) -> None:
    with pytest.raises(IntegrityError) as error, db.begin_nested():
        add(db, *args, **kwargs)
    cause = error.value.orig
    assert isinstance(cause, psycopg.Error)
    assert cause.diag.constraint_name in ({by} if isinstance(by, str) else by)


@pytest.fixture
def camera(db: Connection) -> str:
    add_product(db)
    return add_camera(db, "6215")


def test_migrations_match_models(engine: Engine):
    with engine.connect() as conn:
        command.check(alembic_config(conn))


def test_migrations_downgrade_to_base_and_back(engine: Engine):
    with engine.connect() as conn, conn.begin() as transaction:
        command.downgrade(alembic_config(conn), "base")
        command.upgrade(alembic_config(conn), "head")
        transaction.rollback()


@pytest.mark.parametrize("point", [[299, 350], [-12, 350], [640, -3]])
def test_area_corner_accepts_xy_point_including_off_image(db: Connection, camera: str, point):
    add_area(db, camera, 1, top_left=point)


@pytest.mark.parametrize("point", [[1, 2, 3], [1], [], [1, None], [[1], [2]]])
def test_area_corner_rejects_anything_but_two_integers(db: Connection, camera: str, point):
    assert_rejected(db, "ck_areas_top_left_is_xy", add_area, camera, 1, top_left=point)


def test_area_number_must_be_positive(db: Connection, camera: str):
    assert_rejected(db, "ck_areas_number_positive", add_area, camera, 0)


def test_area_number_is_unique_per_camera(db: Connection, camera: str):
    add_area(db, camera, 1)
    assert_rejected(db, "uq_areas_camera_id_number", add_area, camera, 1, id="6215_one")


def test_frame_filename_is_unique_per_camera(db: Connection, camera: str):
    add_frame(db, camera)
    assert_rejected(db, "uq_frames_camera_id_filename", add_frame, camera)


def test_frame_filename_can_repeat_across_cameras(db: Connection, camera: str):
    add_frame(db, camera)
    add_frame(db, add_camera(db, "6383"))


@pytest.mark.parametrize(("width", "height"), [(1280, None), (None, 720)])
def test_frame_size_is_both_or_neither(db: Connection, camera: str, width, height):
    assert_rejected(
        db, "ck_frames_size_both_or_neither", add_frame, camera, width=width, height=height
    )


def test_frame_without_size_marks_it_unreadable(db: Connection, camera: str):
    add_frame(db, camera, width=None, height=None)


def test_prediction_accepts_frame_and_area_of_same_camera(db: Connection, camera: str):
    add_prediction(db, add_run(db), add_frame(db, camera), add_area(db, camera, 1), camera)


def test_prediction_rejects_area_from_another_camera(db: Connection, camera: str):
    other = add_camera(db, "6383")
    run_id, frame_id, area_id = add_run(db), add_frame(db, camera), add_area(db, other, 1)
    ids = (run_id, frame_id, area_id)
    assert_rejected(db, "fk_predictions_area_id_areas", add_prediction, *ids, camera)
    assert_rejected(db, "fk_predictions_frame_id_frames", add_prediction, *ids, other)


def test_prediction_rejects_camera_that_owns_neither(db: Connection, camera: str):
    other = add_camera(db, "6383")
    run_id, frame_id, area_id = add_run(db), add_frame(db, camera), add_area(db, camera, 1)
    both = {"fk_predictions_area_id_areas", "fk_predictions_frame_id_frames"}
    assert_rejected(db, both, add_prediction, run_id, frame_id, area_id, other)


@pytest.mark.parametrize("field", ["compliance_confidence", "stock_confidence"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_prediction_confidence_is_within_zero_and_one(db: Connection, camera: str, field, value):
    run_id, frame_id, area_id = add_run(db), add_frame(db, camera), add_area(db, camera, 1)
    assert_rejected(
        db,
        f"ck_predictions_{field}_unit",
        add_prediction,
        run_id,
        frame_id,
        area_id,
        camera,
        **{field: value},
    )


@pytest.mark.parametrize(
    "missing",
    [
        {"compliance_confidence"},
        {"stock_confidence"},
        {"compliance_confidence", "stock_confidence"},
    ],
)
def test_prediction_without_confidence_estimate_is_stored(db: Connection, camera: str, missing):
    run_id, frame_id, area_id = add_run(db), add_frame(db, camera), add_area(db, camera, 1)
    add_prediction(db, run_id, frame_id, area_id, camera, **dict.fromkeys(missing))
    stored = db.execute(select(Prediction.compliance_confidence, Prediction.stock_confidence)).one()
    assert {field for field, value in stored._asdict().items() if value is None} == missing


def test_deleting_run_deletes_its_predictions(db: Connection, camera: str):
    run_id = add_run(db)
    add_prediction(db, run_id, add_frame(db, camera), add_area(db, camera, 1), camera)

    db.execute(delete(Run).where(Run.id == run_id))

    assert db.scalar(select(func.count()).select_from(Prediction)) == 0


def test_run_score_cannot_exceed_two_points_per_labelled_area(db: Connection):
    add_run(db, score=20, labelled_areas=10)
    assert_rejected(db, "ck_runs_score_within_max", add_run, score=21, labelled_areas=10)


def test_run_labelled_areas_cannot_be_negative(db: Connection):
    assert_rejected(db, "ck_runs_labelled_areas_non_negative", add_run, labelled_areas=-1)
