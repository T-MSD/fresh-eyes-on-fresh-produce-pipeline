from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text

from config.db.connection import get_connection
from src.api.hour_range import HourRangeResponse
from src.main import app
from tests.db_builders import (
    add_area,
    add_camera,
    add_dataset,
    add_frame,
    add_prediction,
    add_product,
    add_run,
)


@pytest.fixture
def client(db: Connection) -> Iterator[TestClient]:
    app.dependency_overrides[get_connection] = lambda: db
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def test_hour_range_without_runs(client: TestClient):
    response = client.get("/hour-range")
    assert response.status_code == 404
    assert response.json() == {"detail": "No runs available"}


@pytest.mark.parametrize("status", ["running", "done", "failed"])
def test_empty_latest_run_never_uses_older_predictions(db: Connection, client: TestClient, status):
    dataset = add_dataset(db)
    add_product(db, dataset)
    camera = add_camera(db, dataset, "6215")
    area = add_area(db, camera, 1)
    old = add_run(db, dataset, started_at=datetime(2026, 1, 1, tzinfo=UTC), status="done")
    add_prediction(db, old, add_frame(db, camera), area, camera)
    latest = add_run(db, dataset, started_at=datetime(2026, 1, 2, tzinfo=UTC), status=status)

    response = client.get("/hour-range")

    assert response.status_code == 200
    assert response.json() == {"run_id": latest, "start": None, "end": None}


def test_hour_range_only_includes_latest_run_frames(db: Connection, client: TestClient):
    dataset = add_dataset(db)
    add_product(db, dataset)
    latest = add_run(db, dataset, started_at=datetime(2026, 1, 2, tzinfo=UTC))
    older = add_run(db, dataset, started_at=datetime(2026, 1, 1, tzinfo=UTC))
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
    older = add_run(db, dataset, started_at=started_at)
    latest = add_run(db, dataset, started_at=started_at)
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
    older = add_run(db, dataset, started_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert client.get("/hour-range").json()["run_id"] == older
    latest = add_run(db, dataset, started_at=datetime(2026, 1, 2, tzinfo=UTC))
    assert client.get("/hour-range").json()["run_id"] == latest


def test_new_dataset_load_does_not_replace_latest_run(db: Connection, client: TestClient):
    dataset = add_dataset(db)
    add_product(db, dataset)
    camera = add_camera(db, dataset, "6215")
    area = add_area(db, camera, 1)
    run = add_run(db, dataset)
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


@pytest.mark.parametrize("start", [None, "2025-06-27T20:24:01+01:00"])
@pytest.mark.parametrize("end", [None, "2025-06-27T17:24:01-04:00"])
def test_hour_range_response_normalizes_offsets_to_utc(start: str | None, end: str | None):
    response = HourRangeResponse.model_validate({"run_id": 1, "start": start, "end": end})
    payload = response.model_dump(mode="json")
    for field, original in [("start", start), ("end", end)]:
        if original is None:
            assert payload[field] is None
        else:
            assert datetime.fromisoformat(payload[field]) == datetime.fromisoformat(original)
            assert payload[field].endswith("Z")


@pytest.mark.parametrize("timezone", ["UTC", "Europe/Lisbon", "America/New_York"])
def test_hour_range_normalizes_database_session_timezone(
    db: Connection, client: TestClient, timezone: str
):
    db.execute(text("SELECT set_config('TimeZone', :timezone, true)"), {"timezone": timezone})
    dataset = add_dataset(db)
    add_product(db, dataset)
    camera = add_camera(db, dataset, "6215")
    area = add_area(db, camera, 1)
    run = add_run(db, dataset)
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
