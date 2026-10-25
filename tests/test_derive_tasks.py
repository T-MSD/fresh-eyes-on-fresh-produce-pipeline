from pathlib import Path

import pytest
from sqlalchemy import Connection, delete, select

from config.db.load import load_dataset
from config.db.queries import SelectedRun, get_predictions
from config.db.save_run import UnusableRun, derive_tasks, save_run
from config.db.schema import Area, Frame, Task
from ingest.dataset import read_dataset
from tests.builders import area, label, write_dataset

TIMES = [1751052241, 1751055841, 1751059441, 1751063041]
KIND_OF_STATUS = {"empty": "restock", "wrong_product": "move_product", "half": "top_up", "ok": None}


@pytest.fixture
def dataset_id(db: Connection, tmp_path: Path) -> int:
    dataset = read_dataset(
        *write_dataset(tmp_path, {"6215": [area("6215_1"), area("6215_2")]}, {"6215": TIMES})
    )
    return load_dataset(db, dataset).dataset_id


def predictions(*labels: dict, second: tuple[dict, ...] | None = None) -> dict:
    """`labels` are area 6215_1's, one per frame in time order; `second` is area 6215_2's,
    full on every frame when omitted."""
    return {
        "6215": {
            f"{ts}.jpg": {"6215_1": first, "6215_2": other}
            for ts, first, other in zip(
                TIMES, labels, second or [label()] * len(TIMES), strict=True
            )
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
            .order_by(opened.c.ts, Area.external_id)
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


def test_one_frame_without_the_problem_keeps_its_task_open(db, dataset_id):
    run_id = save_run(
        db,
        dataset_id,
        predictions(label("empty"), label(), label("empty"), label()),
        method="run_test",
    )

    assert stored_tasks(db, run_id) == [("6215_1", "restock", "1751052241.jpg", "1751063041.jpg")]


def test_one_frame_with_a_problem_between_frames_without_it_opens_no_task(db, dataset_id):
    run_id = save_run(
        db,
        dataset_id,
        predictions(label(), label("full", False), label(), label()),
        method="run_test",
    )

    assert stored_tasks(db, run_id) == []


def test_alternating_frames_keep_the_first_frames_state(db, dataset_id):
    run_id = save_run(
        db,
        dataset_id,
        predictions(label("half"), label(), label("half"), label()),
        method="run_test",
    )

    assert stored_tasks(db, run_id) == [("6215_1", "top_up", "1751052241.jpg", "1751063041.jpg")]


def test_one_frame_change_is_judged_against_the_nearest_frames_that_are_not_fallbacks(
    db, dataset_id
):
    run_id = save_run(
        db,
        dataset_id,
        predictions(label(), label("half"), label(), label("full")),
        method="run_test",
        fallback={("6215", "1751059441.jpg", "6215_1")},
    )

    assert stored_tasks(db, run_id) == []


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


def test_each_area_keeps_its_own_tasks_while_their_problems_overlap(db, dataset_id):
    run_id = save_run(
        db,
        dataset_id,
        predictions(
            label("empty"),
            label("empty"),
            label(),
            label(),
            second=(label("half"), label("empty"), label("empty"), label()),
        ),
        method="run_test",
    )

    assert stored_tasks(db, run_id) == [
        ("6215_1", "restock", "1751052241.jpg", "1751059441.jpg"),
        ("6215_2", "top_up", "1751052241.jpg", "1751055841.jpg"),
        ("6215_2", "restock", "1751055841.jpg", "1751063041.jpg"),
    ]


def test_task_kinds_match_the_store_page_status(db, tmp_path):
    combos = [
        (level, compliant) for level in ("empty", "half", "full") for compliant in (True, False)
    ]
    areas = [area(f"6215_{n}") for n in range(1, len(combos) + 1)]
    dataset = read_dataset(*write_dataset(tmp_path, {"6215": areas}, {"6215": TIMES[:1]}))
    dataset_id = load_dataset(db, dataset).dataset_id
    run_id = save_run(
        db,
        dataset_id,
        {
            "6215": {
                f"{TIMES[0]}.jpg": {
                    a["area_id"]: label(level, compliant)
                    for a, (level, compliant) in zip(areas, combos, strict=True)
                }
            }
        },
        method="run_test",
    )

    kinds = dict(
        db.execute(
            select(Area.external_id, Task.kind)
            .join(Area, Area.id == Task.area_id)
            .where(Task.run_id == run_id)
        ).all()
    )
    statuses = {
        a.area_id: a.status
        for camera in get_predictions(db, SelectedRun(run_id, dataset_id)).cameras
        for a in camera.areas
    }
    assert statuses.keys() == {a["area_id"] for a in areas}
    assert {area_id: kinds.get(area_id) for area_id in statuses} == {
        area_id: KIND_OF_STATUS[status] for area_id, status in statuses.items()
    }


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
        db,
        dataset_id,
        predictions(label(), label("half"), label("half"), label()),
        method="run_test",
    )
    db.execute(delete(Task).where(Task.run_id == run_id))

    assert derive_tasks(db, run_id) == 1
    assert stored_tasks(db, run_id) == [("6215_1", "top_up", "1751055841.jpg", "1751063041.jpg")]


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
