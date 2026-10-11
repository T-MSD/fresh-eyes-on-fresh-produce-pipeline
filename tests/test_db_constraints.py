from typing import Any

import psycopg
import pytest
from alembic import command
from sqlalchemy import Connection, Engine, delete, func, select
from sqlalchemy.exc import IntegrityError

from config.db.schema import Area, Dataset, Prediction, Run, Task
from tests.conftest import alembic_config
from tests.db_builders import (
    add_area,
    add_camera,
    add_dataset,
    add_frame,
    add_predicted_frame,
    add_prediction,
    add_product,
    add_run,
    add_task,
)


def assert_rejected(
    db: Connection, by: str | set[str], add: Any, *args: Any, **kwargs: Any
) -> None:
    with pytest.raises(IntegrityError) as error, db.begin_nested():
        add(db, *args, **kwargs)
    cause = error.value.orig
    assert isinstance(cause, psycopg.Error)
    assert cause.diag.constraint_name in ({by} if isinstance(by, str) else by)


def test_migrations_match_models(engine: Engine):
    with engine.connect() as conn:
        command.check(alembic_config(conn))


def test_migrations_downgrade_to_base_and_back(engine: Engine):
    with engine.connect() as conn, conn.begin() as transaction:
        command.downgrade(alembic_config(conn), "base")
        command.upgrade(alembic_config(conn), "head")
        transaction.rollback()


def test_database_generates_primary_keys(db: Connection, dataset: int):
    first, second = add_camera(db, dataset, "6215"), add_camera(db, dataset, "6383")
    assert isinstance(first, int)
    assert second != first


def test_dataset_records_paths_and_load_time(db: Connection):
    dataset_id = add_dataset(db)
    stored = db.execute(select(Dataset).where(Dataset.id == dataset_id)).one()
    assert stored.planogram_path == "/data/planogram.json"
    assert stored.images_dir == "/data/training"
    assert stored.loaded_at is not None


@pytest.mark.parametrize(
    ("add", "constraint"),
    [
        (
            lambda db, dataset, camera: add_camera(db, dataset, "6215"),
            "uq_cameras_dataset_id_external_id",
        ),
        (
            lambda db, dataset, camera: add_product(db, dataset),
            "uq_products_dataset_id_external_id",
        ),
        (
            lambda db, dataset, camera: add_area(db, camera, 2, external_id="6215_1"),
            "uq_areas_dataset_id_external_id",
        ),
    ],
    ids=["camera", "product", "area"],
)
def test_external_id_is_unique_within_a_dataset(
    db: Connection, dataset: int, camera: int, add, constraint
):
    add_area(db, camera, 1)
    assert_rejected(db, constraint, add, dataset, camera)


def test_external_id_can_repeat_across_datasets(db: Connection, camera: int):
    add_area(db, camera, 1)
    later = add_dataset(db)
    add_product(db, later)
    add_area(db, add_camera(db, later, "6215"), 1)
    assert list(db.scalars(select(Area.external_id))) == ["6215_1", "6215_1"]


def test_area_rejects_product_from_another_dataset(db: Connection, camera: int):
    product = add_product(db, add_dataset(db))
    assert_rejected(db, "fk_areas_product_id_products", add_area, camera, 1, product_id=product)


def test_area_rejects_dataset_other_than_its_cameras(db: Connection, camera: int):
    later = add_dataset(db)
    product = add_product(db, later)
    assert_rejected(
        db, "fk_areas_camera_id_cameras", add_area, camera, 1, dataset_id=later, product_id=product
    )


@pytest.mark.parametrize("point", [[299, 350], [-12, 350], [640, -3]])
def test_area_corner_accepts_xy_point_including_off_image(db: Connection, camera: int, point):
    add_area(db, camera, 1, top_left=point)


@pytest.mark.parametrize("point", [[1, 2, 3], [1], [], [1, None], [[1], [2]]])
def test_area_corner_rejects_anything_but_two_integers(db: Connection, camera: int, point):
    assert_rejected(db, "ck_areas_top_left_is_xy", add_area, camera, 1, top_left=point)


def test_area_number_must_be_positive(db: Connection, camera: int):
    assert_rejected(db, "ck_areas_number_positive", add_area, camera, 0)


def test_area_number_is_unique_per_camera(db: Connection, camera: int):
    add_area(db, camera, 1)
    assert_rejected(db, "uq_areas_camera_id_number", add_area, camera, 1, external_id="6215_one")


def test_frame_filename_is_unique_per_camera(db: Connection, camera: int):
    add_frame(db, camera)
    assert_rejected(db, "uq_frames_camera_id_filename", add_frame, camera)


def test_frame_filename_can_repeat_across_cameras(db: Connection, dataset: int, camera: int):
    add_frame(db, camera)
    add_frame(db, add_camera(db, dataset, "6383"))


@pytest.mark.parametrize(("width", "height"), [(1280, None), (None, 720)])
def test_frame_size_is_both_or_neither(db: Connection, camera: int, width, height):
    assert_rejected(
        db, "ck_frames_size_both_or_neither", add_frame, camera, width=width, height=height
    )


def test_frame_without_size_marks_it_unreadable(db: Connection, camera: int):
    add_frame(db, camera, width=None, height=None)


def test_prediction_accepts_frame_and_area_of_same_camera(
    db: Connection, dataset: int, camera: int
):
    add_prediction(db, add_run(db, dataset), add_frame(db, camera), add_area(db, camera, 1), camera)


def test_prediction_gets_generated_id_and_is_unique_per_run_frame_and_area(
    db: Connection, dataset: int, camera: int
):
    ids = (add_run(db, dataset), add_frame(db, camera), add_area(db, camera, 1))
    other_area = add_area(db, camera, 2)
    first = add_prediction(db, *ids, camera)
    second = add_prediction(db, ids[0], ids[1], other_area, camera)
    assert first != second
    assert_rejected(db, "uq_predictions_run_id_frame_id_area_id", add_prediction, *ids, camera)


def test_prediction_rejects_area_from_another_camera(db: Connection, dataset: int, camera: int):
    other = add_camera(db, dataset, "6383")
    run_id, frame_id, area_id = add_run(db, dataset), add_frame(db, camera), add_area(db, other, 1)
    ids = (run_id, frame_id, area_id)
    assert_rejected(db, "fk_predictions_area_id_areas", add_prediction, *ids, camera)
    assert_rejected(db, "fk_predictions_frame_id_frames", add_prediction, *ids, other)


def test_prediction_rejects_camera_that_owns_neither(db: Connection, dataset: int, camera: int):
    other = add_camera(db, dataset, "6383")
    run_id, frame_id, area_id = add_run(db, dataset), add_frame(db, camera), add_area(db, camera, 1)
    both = {"fk_predictions_area_id_areas", "fk_predictions_frame_id_frames"}
    assert_rejected(db, both, add_prediction, run_id, frame_id, area_id, other)


def test_prediction_rejects_run_from_another_dataset(db: Connection, camera: int):
    run_id = add_run(db, add_dataset(db))
    frame_id, area_id = add_frame(db, camera), add_area(db, camera, 1)
    assert_rejected(
        db, "fk_predictions_run_id_runs", add_prediction, run_id, frame_id, area_id, camera
    )


def test_prediction_rejects_dataset_other_than_its_cameras(db: Connection, camera: int):
    later = add_dataset(db)
    run_id, frame_id, area_id = add_run(db, later), add_frame(db, camera), add_area(db, camera, 1)
    assert_rejected(
        db,
        "fk_predictions_camera_id_cameras",
        add_prediction,
        run_id,
        frame_id,
        area_id,
        camera,
        dataset_id=later,
    )


def test_prediction_accepts_detected_product_of_its_dataset(
    db: Connection, dataset: int, camera: int
):
    product = add_product(db, dataset, "20923")
    run_id, frame_id, area_id = add_run(db, dataset), add_frame(db, camera), add_area(db, camera, 1)
    add_prediction(db, run_id, frame_id, area_id, camera, detected_product_id=product)


def test_prediction_rejects_detected_product_from_another_dataset(
    db: Connection, dataset: int, camera: int
):
    product = add_product(db, add_dataset(db))
    run_id, frame_id, area_id = add_run(db, dataset), add_frame(db, camera), add_area(db, camera, 1)
    assert_rejected(
        db,
        "fk_predictions_detected_product_id_products",
        add_prediction,
        run_id,
        frame_id,
        area_id,
        camera,
        detected_product_id=product,
    )


@pytest.mark.parametrize("field", ["compliance_confidence", "stock_confidence"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_prediction_confidence_is_within_zero_and_one(
    db: Connection, dataset: int, camera: int, field, value
):
    run_id, frame_id, area_id = add_run(db, dataset), add_frame(db, camera), add_area(db, camera, 1)
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
def test_prediction_without_confidence_estimate_is_stored(
    db: Connection, dataset: int, camera: int, missing
):
    run_id, frame_id, area_id = add_run(db, dataset), add_frame(db, camera), add_area(db, camera, 1)
    add_prediction(db, run_id, frame_id, area_id, camera, **dict.fromkeys(missing))
    stored = db.execute(select(Prediction.compliance_confidence, Prediction.stock_confidence)).one()
    assert {field for field, value in stored._asdict().items() if value is None} == missing


def test_deleting_run_deletes_its_predictions(db: Connection, dataset: int, camera: int):
    run_id = add_run(db, dataset)
    add_prediction(db, run_id, add_frame(db, camera), add_area(db, camera, 1), camera)

    db.execute(delete(Run).where(Run.id == run_id))

    assert db.scalar(select(func.count()).select_from(Prediction)) == 0


def test_run_rejects_unknown_dataset(db: Connection):
    assert_rejected(db, "fk_runs_dataset_id_datasets", add_run, -1)


def test_run_score_cannot_exceed_two_points_per_labelled_area(db: Connection, dataset: int):
    add_run(db, dataset, score=20, labelled_areas=10)
    assert_rejected(db, "ck_runs_score_within_max", add_run, dataset, score=21, labelled_areas=10)


def test_run_labelled_areas_cannot_be_negative(db: Connection, dataset: int):
    assert_rejected(db, "ck_runs_labelled_areas_non_negative", add_run, dataset, labelled_areas=-1)


def test_task_opens_and_closes_on_frames_its_run_predicted_for_its_area(
    db: Connection, dataset: int, camera: int
):
    run_id, area_id = add_run(db, dataset), add_area(db, camera, 1)
    opened = add_predicted_frame(db, run_id, area_id, camera)
    closed = add_predicted_frame(db, run_id, area_id, camera, 1751055841)
    add_task(db, run_id, area_id, opened, closed_frame_id=closed)


@pytest.mark.parametrize("frame", ["opened_frame_id", "closed_frame_id"])
def test_task_rejects_frame_its_run_did_not_predict_for_its_area(
    db: Connection, dataset: int, camera: int, frame
):
    run_id, area_id = add_run(db, dataset), add_area(db, camera, 1)
    predicted = add_predicted_frame(db, run_id, area_id, camera)
    unpredicted = add_frame(db, camera, 1751055841)
    by_another_run = add_predicted_frame(db, add_run(db, dataset), area_id, camera, 1751059441)
    for_another_area = add_predicted_frame(db, run_id, add_area(db, camera, 2), camera, 1751063041)
    for other in (unpredicted, by_another_run, for_another_area):
        frames = {"opened_frame_id": predicted, frame: other}
        assert_rejected(db, f"fk_tasks_{frame}_predictions", add_task, run_id, area_id, **frames)


def test_task_is_unique_per_run_area_kind_and_opened_frame(
    db: Connection, dataset: int, camera: int
):
    run_id, area_id = add_run(db, dataset), add_area(db, camera, 1)
    opened = add_predicted_frame(db, run_id, area_id, camera)
    closed = add_predicted_frame(db, run_id, area_id, camera, 1751055841)
    add_task(db, run_id, area_id, opened, closed_frame_id=closed)
    add_task(db, run_id, area_id, opened, kind="move_product", closed_frame_id=closed)
    assert_rejected(
        db,
        "uq_tasks_run_id_area_id_kind_opened_frame_id",
        add_task,
        run_id,
        area_id,
        opened,
        closed_frame_id=closed,
    )


def test_area_has_at_most_one_open_task_per_run(db: Connection, dataset: int, camera: int):
    run_id, area_id = add_run(db, dataset), add_area(db, camera, 1)
    first, second, third = (
        add_predicted_frame(db, run_id, area_id, camera, ts)
        for ts in (1751052241, 1751055841, 1751059441)
    )
    add_task(db, run_id, area_id, first, kind="top_up", closed_frame_id=second)
    add_task(db, run_id, area_id, second)
    assert_rejected(
        db, "uq_tasks_run_id_area_id_open", add_task, run_id, area_id, third, kind="move_product"
    )


def test_open_tasks_of_other_areas_or_runs_do_not_conflict(
    db: Connection, dataset: int, camera: int
):
    run_id, area_id, frame_id = add_run(db, dataset), add_area(db, camera, 1), add_frame(db, camera)
    for run, area in [
        (run_id, area_id),
        (run_id, add_area(db, camera, 2)),
        (add_run(db, dataset), area_id),
    ]:
        add_prediction(db, run, frame_id, area, camera)
        add_task(db, run, area, frame_id)


def test_task_cannot_close_on_the_frame_that_opened_it(db: Connection, dataset: int, camera: int):
    run_id, area_id = add_run(db, dataset), add_area(db, camera, 1)
    frame_id = add_predicted_frame(db, run_id, area_id, camera)
    assert_rejected(
        db,
        "ck_tasks_closed_on_another_frame",
        add_task,
        run_id,
        area_id,
        frame_id,
        closed_frame_id=frame_id,
    )


def test_task_starts_open_and_not_done(db: Connection, dataset: int, camera: int):
    run_id, area_id = add_run(db, dataset), add_area(db, camera, 1)
    task_id = add_task(db, run_id, area_id, add_predicted_frame(db, run_id, area_id, camera))
    stored = db.execute(select(Task.closed_frame_id, Task.done_at).where(Task.id == task_id)).one()
    assert tuple(stored) == (None, None)


def test_deleting_run_deletes_its_tasks(db: Connection, dataset: int, camera: int):
    run_id, area_id = add_run(db, dataset), add_area(db, camera, 1)
    add_task(db, run_id, area_id, add_predicted_frame(db, run_id, area_id, camera))

    db.execute(delete(Run).where(Run.id == run_id))

    assert db.scalar(select(func.count()).select_from(Task)) == 0
