from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, select, text

from config.db.connection import get_connection
from config.db.queries import get_latest_run, mark_task_done
from config.db.schema import Base, Task
from src.api.dependencies import Transaction
from src.main import create_app
from tests.assertions import iso
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

MORNING, NOON, EVENING = 1751052241, 1751055841, 1751059441


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


def done_body(area_id: str = "6215_1", kind: str = "restock", ts: int = MORNING) -> dict:
    return {"area_id": area_id, "kind": kind, "opened_filename": f"{ts}.jpg"}


def done_at(db: Connection, task_id: int) -> datetime | None:
    return db.scalar(select(Task.done_at).where(Task.id == task_id))


@pytest.mark.parametrize("status", [None, "running", "failed"])
def test_mark_done_without_completed_run(db: Connection, client: TestClient, dataset: int, status):
    if status is not None:
        add_run(db, dataset, status=status)

    response = client.post("/tasks/done", json=done_body())

    assert response.status_code == 404
    assert response.json() == {"detail": "No completed runs available"}


def test_marks_the_task_done(db: Connection, client: TestClient, dataset: int, camera: int):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    task_id = add_task(db, run_id, area_id, add_predicted_frame(db, run_id, area_id, camera))

    response = client.post("/tasks/done", json=done_body())

    assert response.status_code == 204
    assert response.content == b""
    assert done_at(db, task_id) is not None
    assert client.get("/tasks").json() == {"tasks": []}


def test_marking_a_done_task_again_keeps_its_first_done_time(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    first = datetime(2026, 1, 1, tzinfo=UTC)
    task_id = add_task(
        db, run_id, area_id, add_predicted_frame(db, run_id, area_id, camera), done_at=first
    )

    response = client.post("/tasks/done", json=done_body())

    assert response.status_code == 204
    assert done_at(db, task_id) == first


def test_marks_only_the_identified_task(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id = add_run(db, dataset, status="done")
    first, second = add_area(db, camera, 1), add_area(db, camera, 2)
    morning, noon = add_frame(db, camera, MORNING), add_frame(db, camera, NOON)
    for area_id in (first, second):
        for frame_id in (morning, noon):
            add_prediction(db, run_id, frame_id, area_id, camera)
    target = add_task(db, run_id, first, morning, closed_frame_id=noon)
    others = [
        add_task(db, run_id, first, morning, kind="top_up", closed_frame_id=noon),
        add_task(db, run_id, first, noon),
        add_task(db, run_id, second, morning),
    ]

    response = client.post("/tasks/done", json=done_body())

    assert response.status_code == 204
    assert done_at(db, target) is not None
    assert [done_at(db, task_id) for task_id in others] == [None, None, None]


@pytest.mark.parametrize(
    "body",
    [
        done_body(area_id="6215_9"),
        done_body(area_id="9999_1"),
        done_body(kind="top_up"),
        done_body(ts=NOON),
    ],
)
def test_unknown_task_is_not_found(
    db: Connection, client: TestClient, dataset: int, camera: int, body: dict
):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    add_frame(db, camera, NOON)
    task_id = add_task(db, run_id, area_id, add_predicted_frame(db, run_id, area_id, camera))

    response = client.post("/tasks/done", json=body)

    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found in the latest completed run"}
    assert done_at(db, task_id) is None


def test_task_of_an_older_run_is_not_found(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    area_id, frame_id = add_area(db, camera, 1), add_frame(db, camera)
    started_at = datetime(2026, 1, 2, tzinfo=UTC)
    older = add_run(db, dataset, started_at=started_at - timedelta(days=1), status="done")
    add_prediction(db, older, frame_id, area_id, camera)
    task_id = add_task(db, older, area_id, frame_id)
    add_run(db, dataset, started_at=started_at, status="done")

    response = client.post("/tasks/done", json=done_body())

    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found in the latest completed run"}
    assert done_at(db, task_id) is None


@pytest.mark.parametrize(
    "body",
    [
        done_body(kind="sweep"),
        {"area_id": "6215_1", "kind": "restock"},
        {"kind": "restock", "opened_filename": f"{MORNING}.jpg"},
    ],
)
def test_invalid_mark_done_body_is_rejected(client: TestClient, body: dict):
    assert client.post("/tasks/done", json=body).status_code == 422


def test_frontend_origin_can_mark_tasks_done(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173")
    with TestClient(create_app()) as client:
        response = client.options(
            "/tasks/done",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

    assert response.status_code == 200
    assert "POST" in response.headers["access-control-allow-methods"]


@pytest.fixture
def committed(engine: Engine) -> Iterator[Engine]:
    try:
        yield engine
    finally:
        tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
        with engine.begin() as conn:
            conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


def add_committed_task(engine: Engine) -> int:
    with engine.begin() as db:
        dataset = add_dataset(db)
        add_product(db, dataset)
        camera = add_camera(db, dataset, "6215")
        run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
        return add_task(db, run_id, area_id, add_predicted_frame(db, run_id, area_id, camera))


def committed_done_at(engine: Engine, task_id: int) -> datetime | None:
    with engine.connect() as db:
        return done_at(db, task_id)


def committed_client(engine: Engine, app: FastAPI) -> TestClient:
    def connect() -> Iterator[Connection]:
        with engine.connect() as connection:
            yield connection

    app.dependency_overrides[get_connection] = connect
    return TestClient(app)


def test_marking_done_is_committed(committed: Engine):
    task_id = add_committed_task(committed)

    with committed_client(committed, create_app()) as client:
        response = client.post("/tasks/done", json=done_body())

    assert response.status_code == 204
    assert committed_done_at(committed, task_id) is not None


def test_a_failing_write_is_rolled_back(committed: Engine):
    task_id = add_committed_task(committed)
    app = create_app()

    @app.post("/fail-after-marking-done")
    def fail_after_marking_done(connection: Transaction) -> None:
        run = get_latest_run(connection)
        if run is not None and mark_task_done(
            connection, run, "6215_1", "restock", "1751052241.jpg"
        ):
            raise HTTPException(status_code=409)

    with committed_client(committed, app) as client:
        response = client.post("/fail-after-marking-done")

    assert response.status_code == 409
    assert committed_done_at(committed, task_id) is None
