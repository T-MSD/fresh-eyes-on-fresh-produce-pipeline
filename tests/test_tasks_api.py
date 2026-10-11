from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text

from config.db.connection import get_connection
from src.main import app
from tests.db_builders import (
    add_area,
    add_frame,
    add_predicted_frame,
    add_prediction,
    add_run,
    add_task,
)

MORNING, NOON, EVENING = 1751052241, 1751055841, 1751059441


@pytest.fixture
def client(db: Connection) -> Iterator[TestClient]:
    app.dependency_overrides[get_connection] = lambda: db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat().replace("+00:00", "Z")


def test_tasks_without_runs(client: TestClient):
    response = client.get("/tasks")
    assert response.status_code == 404
    assert response.json() == {"detail": "No completed runs available"}


def test_lists_tasks_of_the_latest_run_not_marked_done(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 2)
    opened, closed, later = (
        add_predicted_frame(db, run_id, area_id, camera, ts) for ts in (MORNING, NOON, EVENING)
    )
    add_task(db, run_id, area_id, opened, kind="top_up", closed_frame_id=closed)
    add_task(db, run_id, area_id, later, done_at=datetime.now(UTC))

    response = client.get("/tasks")

    assert response.status_code == 200
    assert response.json() == {
        "tasks": [
            {
                "camera": "6215",
                "area_id": "6215_2",
                "product_id": "11908",
                "product_name": "BANANA CAT I CNT KG",
                "kind": "top_up",
                "opened": {"filename": f"{MORNING}.jpg", "ts": iso(MORNING)},
                "closed": {"filename": f"{NOON}.jpg", "ts": iso(NOON)},
            }
        ],
    }


def test_open_task_has_no_closing_frame(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    add_task(db, run_id, area_id, add_predicted_frame(db, run_id, area_id, camera))

    [task] = client.get("/tasks").json()["tasks"]

    assert task["closed"] is None


@pytest.mark.parametrize("timezone", ["UTC", "Europe/Lisbon", "America/New_York"])
def test_frame_timestamps_are_utc_whatever_the_session_timezone(
    db: Connection, client: TestClient, dataset: int, camera: int, timezone: str
):
    db.execute(text("SELECT set_config('TimeZone', :timezone, true)"), {"timezone": timezone})
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    opened, closed = (
        add_predicted_frame(db, run_id, area_id, camera, ts) for ts in (MORNING, NOON)
    )
    add_task(db, run_id, area_id, opened, closed_frame_id=closed)

    [task] = client.get("/tasks").json()["tasks"]

    assert (task["opened"]["ts"], task["closed"]["ts"]) == (iso(MORNING), iso(NOON))


@pytest.mark.parametrize("status", ["running", "done", "failed"])
def test_only_latest_completed_run_tasks(
    db: Connection, client: TestClient, dataset: int, camera: int, status
):
    area_id, frame_id = add_area(db, camera, 1), add_frame(db, camera)
    latest = add_run(db, dataset, started_at=datetime(2026, 1, 2, tzinfo=UTC), status=status)
    older = add_run(db, dataset, started_at=datetime(2026, 1, 1, tzinfo=UTC), status="done")
    for run_id, kind in ((older, "restock"), (latest, "move_product")):
        add_prediction(db, run_id, frame_id, area_id, camera)
        add_task(db, run_id, area_id, frame_id, kind=kind)

    tasks = client.get("/tasks").json()["tasks"]

    assert [task["kind"] for task in tasks] == ["move_product" if status == "done" else "restock"]


def test_latest_run_without_pending_tasks_never_uses_older_runs(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    area_id, frame_id = add_area(db, camera, 1), add_frame(db, camera)
    started_at = datetime(2026, 1, 2, tzinfo=UTC)
    older = add_run(db, dataset, started_at=started_at - timedelta(days=1), status="done")
    add_prediction(db, older, frame_id, area_id, camera)
    add_task(db, older, area_id, frame_id)
    latest = add_run(db, dataset, started_at=started_at, status="done")
    add_prediction(db, latest, frame_id, area_id, camera)
    add_task(db, latest, area_id, frame_id, done_at=datetime.now(UTC))

    response = client.get("/tasks")

    assert response.status_code == 200
    assert response.json() == {"tasks": []}


def test_sorts_by_priority_then_opening_time(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id = add_run(db, dataset, status="done")
    frames = {ts: add_frame(db, camera, ts) for ts in (MORNING, NOON)}
    for number, kind, ts in [
        (1, "top_up", MORNING),
        (2, "restock", NOON),
        (3, "move_product", MORNING),
        (4, "restock", MORNING),
    ]:
        area_id = add_area(db, camera, number)
        add_prediction(db, run_id, frames[ts], area_id, camera)
        add_task(db, run_id, area_id, frames[ts], kind=kind)

    tasks = client.get("/tasks").json()["tasks"]

    assert [(task["area_id"], task["kind"]) for task in tasks] == [
        ("6215_4", "restock"),
        ("6215_2", "restock"),
        ("6215_3", "move_product"),
        ("6215_1", "top_up"),
    ]


def test_tasks_tied_on_every_other_key_keep_creation_order(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    names = ["b.jpg", "a.jpg", "c.jpg"]
    frames = [add_frame(db, camera, MORNING, filename=name) for name in names]
    for frame_id in frames:
        add_prediction(db, run_id, frame_id, area_id, camera)
    closing = add_predicted_frame(db, run_id, area_id, camera, NOON)
    for frame_id in frames:
        add_task(db, run_id, area_id, frame_id, closed_frame_id=closing)

    tasks = client.get("/tasks").json()["tasks"]

    assert [task["opened"]["filename"] for task in tasks] == names
