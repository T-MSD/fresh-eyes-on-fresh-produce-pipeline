from pathlib import Path

import pytest
from sqlalchemy import Connection, delete, select

from config.db.load import load_dataset
from config.db.save_run import UnusableRun, derive_tasks, save_run
from config.db.schema import Area, Frame, Task
from ingest.dataset import read_dataset
from tests.builders import area, write_dataset

TIMES = [1751052241, 1751055841, 1751059441, 1751063041]


@pytest.fixture
def dataset_id(db: Connection, tmp_path: Path) -> int:
    dataset = read_dataset(
        *write_dataset(tmp_path, {"6215": [area("6215_1"), area("6215_2")]}, {"6215": TIMES})
    )
    return load_dataset(db, dataset).dataset_id


def label(level: str = "full", compliant: bool = True) -> dict:
    return {"planogram_compliant": compliant, "stock_level": level}


def predictions(*labels: dict) -> dict:
    """`labels` are area 6215_1's, one per frame in time order; area 6215_2 is always full."""
    return {
        "6215": {
            f"{ts}.jpg": {"6215_1": predicted, "6215_2": label()}
            for ts, predicted in zip(TIMES, labels, strict=True)
        }
    }


def stored_tasks(db: Connection, run_id: int) -> list[tuple]:
    opened, closed = Frame.__table__.alias(), Frame.__table__.alias()
    return [
        tuple(row)
        for row in db.execute(
            select(Area.external_id, Task.kind, opened.c.filename, closed.c.filename)
            .join(Area, Area.id == Task.area_id)
            .join(opened, opened.c.id == Task.opened_frame_id)
            .outerjoin(closed, closed.c.id == Task.closed_frame_id)
            .where(Task.run_id == run_id)
            .order_by(opened.c.ts)
        )
    ]


def test_opens_on_the_problem_frame_and_closes_on_the_first_frame_without_it(db, dataset_id):
    run_id = save_run(
        db,
        dataset_id,
        predictions(label(), label("empty"), label("empty"), label()),
        method="run_test",
    )

    assert stored_tasks(db, run_id) == [("6215_1", "restock", "1751055841.jpg", "1751063041.jpg")]


def test_a_problem_still_there_on_the_last_frame_stays_open(db, dataset_id):
    run_id = save_run(
        db, dataset_id, predictions(label(), label(), label(), label("half")), method="run_test"
    )

    assert stored_tasks(db, run_id) == [("6215_1", "top_up", "1751063041.jpg", None)]


def test_a_different_problem_closes_the_task_and_opens_its_own_on_that_frame(db, dataset_id):
    run_id = save_run(
        db,
        dataset_id,
        predictions(label("half"), label("empty"), label("full", False), label("full", False)),
        method="run_test",
    )

    assert stored_tasks(db, run_id) == [
        ("6215_1", "top_up", "1751052241.jpg", "1751055841.jpg"),
        ("6215_1", "restock", "1751055841.jpg", "1751059441.jpg"),
        ("6215_1", "move_product", "1751059441.jpg", None),
    ]


def test_empty_wins_over_a_wrong_product_and_a_wrong_product_over_half(db, dataset_id):
    run_id = save_run(
        db,
        dataset_id,
        predictions(label("empty", False), label("half", False), label(), label()),
        method="run_test",
    )

    assert stored_tasks(db, run_id) == [
        ("6215_1", "restock", "1751052241.jpg", "1751055841.jpg"),
        ("6215_1", "move_product", "1751055841.jpg", "1751059441.jpg"),
    ]


def test_fallback_predictions_neither_open_nor_close_a_task(db, dataset_id):
    run_id = save_run(
        db,
        dataset_id,
        predictions(label("half"), label("empty"), label("empty"), label()),
        method="run_test",
        fallback={("6215", "1751052241.jpg", "6215_1"), ("6215", "1751063041.jpg", "6215_1")},
    )

    assert stored_tasks(db, run_id) == [("6215_1", "restock", "1751055841.jpg", None)]


def test_backfills_a_run_stored_without_tasks(db, dataset_id):
    run_id = save_run(
        db, dataset_id, predictions(label(), label("half"), label(), label()), method="run_test"
    )
    db.execute(delete(Task).where(Task.run_id == run_id))

    assert derive_tasks(db, run_id) == 1
    assert stored_tasks(db, run_id) == [("6215_1", "top_up", "1751055841.jpg", "1751059441.jpg")]


def test_refuses_a_run_that_already_has_tasks(db, dataset_id):
    run_id = save_run(
        db, dataset_id, predictions(label("empty"), label(), label(), label()), method="run_test"
    )

    with pytest.raises(UnusableRun, match=f"run {run_id} already has tasks"):
        derive_tasks(db, run_id)

    assert len(stored_tasks(db, run_id)) == 1


def test_refuses_an_unknown_run(db, dataset_id):
    with pytest.raises(UnusableRun, match="run 999999 does not exist"):
        derive_tasks(db, 999999)
