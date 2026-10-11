from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text

from tests.db_builders import (
    add_area,
    add_camera,
    add_dataset,
    add_frame,
    add_prediction,
    add_product,
    add_run,
)


def test_hour_range_without_runs(client: TestClient):
    response = client.get("/hour-range")
    assert response.status_code == 404
    assert response.json() == {"detail": "No completed runs available"}


def test_empty_completed_run_never_uses_older_predictions(db: Connection, client: TestClient):
    dataset = add_dataset(db)
    add_product(db, dataset)
    camera = add_camera(db, dataset, "6215")
    area = add_area(db, camera, 1)
    old = add_run(db, dataset, started_at=datetime(2026, 1, 1, tzinfo=UTC), status="done")
    add_prediction(db, old, add_frame(db, camera), area, camera)
    latest = add_run(db, dataset, started_at=datetime(2026, 1, 2, tzinfo=UTC), status="done")

    response = client.get("/hour-range")

    assert response.status_code == 200
    assert response.json() == {"run_id": latest, "start": None, "end": None}


@pytest.mark.parametrize("status", ["running", "failed"])
@pytest.mark.parametrize("has_predictions", [False, True])
def test_incomplete_run_does_not_hide_completed_predictions(
    db: Connection,
    client: TestClient,
    dataset: int,
    camera: int,
    status: str,
    has_predictions: bool,
):
    area = add_area(db, camera, 1)
    old = add_run(db, dataset, started_at=datetime(2026, 1, 1, tzinfo=UTC), status="done")
    add_prediction(db, old, add_frame(db, camera), area, camera)
    latest = add_run(db, dataset, started_at=datetime(2026, 1, 2, tzinfo=UTC), status=status)
    if has_predictions:
        add_prediction(db, latest, add_frame(db, camera, 1751059441), area, camera)

    response = client.get("/hour-range")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == old
    for field in ("start", "end"):
        assert datetime.fromisoformat(payload[field]) == datetime.fromtimestamp(1751052241, UTC)


@pytest.mark.parametrize("status", ["running", "failed"])
@pytest.mark.parametrize("endpoint", ["/hour-range", "/tasks"])
def test_without_completed_runs_returns_404(
    db: Connection, client: TestClient, dataset: int, status: str, endpoint: str
):
    add_run(db, dataset, status=status)

    response = client.get(endpoint)

    assert response.status_code == 404
    assert response.json() == {"detail": "No completed runs available"}


def test_hour_range_only_includes_latest_run_frames(db: Connection, client: TestClient):
    dataset = add_dataset(db)
    add_product(db, dataset)
    latest = add_run(db, dataset, started_at=datetime(2026, 1, 2, tzinfo=UTC), status="done")
    older = add_run(db, dataset, started_at=datetime(2026, 1, 1, tzinfo=UTC), status="done")
    for camera_id, ts in [("6215", 1751052241), ("6383", 1751059441)]:
        camera = add_camera(db, dataset, camera_id)
        area = add_area(db, camera, 1)
        second_area = add_area(db, camera, 2)
        frame = add_frame(db, camera, ts)
        add_prediction(db, latest, frame, area, camera)
        add_prediction(db, latest, frame, second_area, camera)
        add_prediction(db, older, add_frame(db, camera, ts - 86400), area, camera)
        add_frame(db, camera, ts + 86400)

    response = client.get("/hour-range")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == latest
    assert datetime.fromisoformat(payload["start"]) == datetime.fromtimestamp(1751052241, UTC)
    assert datetime.fromisoformat(payload["end"]) == datetime.fromtimestamp(1751059441, UTC)


def test_single_frame_and_tied_run_start_times(db: Connection, client: TestClient):
    dataset = add_dataset(db)
    add_product(db, dataset)
    camera = add_camera(db, dataset, "6215")
    area = add_area(db, camera, 1)
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    older = add_run(db, dataset, started_at=started_at, status="done")
    latest = add_run(db, dataset, started_at=started_at, status="done")
    add_prediction(db, older, add_frame(db, camera, 1751052241), area, camera)
    add_prediction(db, latest, add_frame(db, camera, 1751059441), area, camera)

    response = client.get("/hour-range")

    assert response.status_code == 200
    timestamp = datetime.fromtimestamp(1751059441, UTC)
    payload = response.json()
    assert payload["run_id"] == latest
    assert datetime.fromisoformat(payload["start"]) == timestamp
    assert datetime.fromisoformat(payload["end"]) == timestamp


def test_hour_range_reselects_latest_run_on_each_request(db: Connection, client: TestClient):
    dataset = add_dataset(db)
    older = add_run(db, dataset, started_at=datetime(2026, 1, 1, tzinfo=UTC), status="done")
    assert client.get("/hour-range").json()["run_id"] == older
    latest = add_run(db, dataset, started_at=datetime(2026, 1, 2, tzinfo=UTC), status="done")
    assert client.get("/hour-range").json()["run_id"] == latest


def test_new_dataset_load_does_not_replace_latest_run(db: Connection, client: TestClient):
    dataset = add_dataset(db)
    add_product(db, dataset)
    camera = add_camera(db, dataset, "6215")
    area = add_area(db, camera, 1)
    run = add_run(db, dataset, status="done")
    add_prediction(db, run, add_frame(db, camera), area, camera)
    newer_dataset = add_dataset(db)
    add_product(db, newer_dataset)
    newer_camera = add_camera(db, newer_dataset, "6215")
    add_frame(db, newer_camera, 1751059441)

    response = client.get("/hour-range")

    assert response.status_code == 200
    timestamp = datetime.fromtimestamp(1751052241, UTC)
    payload = response.json()
    assert payload["run_id"] == run
    assert datetime.fromisoformat(payload["start"]) == timestamp
    assert datetime.fromisoformat(payload["end"]) == timestamp


@pytest.mark.parametrize("timezone", ["UTC", "Europe/Lisbon", "America/New_York"])
def test_hour_range_normalizes_database_session_timezone(
    db: Connection, client: TestClient, timezone: str
):
    db.execute(text("SELECT set_config('TimeZone', :timezone, true)"), {"timezone": timezone})
    dataset = add_dataset(db)
    add_product(db, dataset)
    camera = add_camera(db, dataset, "6215")
    area = add_area(db, camera, 1)
    run = add_run(db, dataset, status="done")
    start, end = 1751052241, 1751059441
    for ts in (start, end):
        add_prediction(db, run, add_frame(db, camera, ts), area, camera)

    response = client.get("/hour-range")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run
    for field, ts in [("start", start), ("end", end)]:
        assert datetime.fromisoformat(payload[field]) == datetime.fromtimestamp(ts, UTC)
        assert payload[field].endswith("Z")
