from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection

from config.db.schema import RunStatus
from src.main import create_app
from tests.test_db_constraints import (
    add_area,
    add_camera,
    add_frame,
    add_prediction,
    add_product,
    add_run,
)


@pytest.fixture
def client(db: Connection):
    with TestClient(create_app(db)) as client:
        yield client


@pytest.fixture
def prediction_ids(db: Connection):
    camera_id = add_camera(db, "6215")
    add_product(db)
    area_id = add_area(db, camera_id, 1)
    frame_id = add_frame(db, camera_id)
    return camera_id, frame_id, area_id


def test_no_run_returns_404(client: TestClient):
    response = client.get("/predictions/1/6215_1/frame")
    assert response.status_code == 404
    assert response.json()["detail"] == "No run available"


def test_unknown_prediction_returns_404(db: Connection, client: TestClient):
    add_run(db)
    response = client.get("/predictions/1/6215_1/frame")
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found in the most recent run"


@pytest.mark.parametrize("status", ["running", "done", "failed"])
def test_returns_frame_from_newest_run(
    db: Connection, client: TestClient, prediction_ids, status: RunStatus
):
    camera_id, frame_id, area_id = prediction_ids
    started_at = datetime(2026, 10, 10, tzinfo=UTC)
    newest = add_run(db, started_at=started_at, status=status)
    add_prediction(db, newest, frame_id, area_id, camera_id)
    add_run(db, started_at=started_at - timedelta(hours=1), status="done")

    response = client.get(f"/predictions/{frame_id}/{area_id}/frame")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": newest,
        "id": frame_id,
        "camera_id": camera_id,
        "filename": "1751052241.jpg",
        "ts": "2025-06-27T19:24:01Z",
        "width": 1280,
        "height": 720,
    }


def test_does_not_fall_back_to_an_older_run(db: Connection, client: TestClient, prediction_ids):
    camera_id, frame_id, area_id = prediction_ids
    old = add_run(db, started_at=datetime(2026, 10, 9, tzinfo=UTC))
    add_prediction(db, old, frame_id, area_id, camera_id)
    add_run(db, started_at=datetime(2026, 10, 10, tzinfo=UTC))
    response = client.get(f"/predictions/{frame_id}/{area_id}/frame")
    assert response.status_code == 404


def test_tied_start_times_use_highest_run_id(db: Connection, client: TestClient, prediction_ids):
    camera_id, frame_id, area_id = prediction_ids
    started_at = datetime(2026, 10, 10, tzinfo=UTC)
    old = add_run(db, started_at=started_at)
    add_prediction(db, old, frame_id, area_id, camera_id)
    add_run(db, started_at=started_at)
    assert client.get(f"/predictions/{frame_id}/{area_id}/frame").status_code == 404


def test_new_run_is_selected_on_each_request(db: Connection, client: TestClient, prediction_ids):
    camera_id, frame_id, area_id = prediction_ids
    old = add_run(db, started_at=datetime(2026, 10, 9, tzinfo=UTC))
    add_prediction(db, old, frame_id, area_id, camera_id)
    path = f"/predictions/{frame_id}/{area_id}/frame"
    assert client.get(path).status_code == 200
    add_run(db, started_at=datetime(2026, 10, 10, tzinfo=UTC))
    assert client.get(path).status_code == 404


def test_frame_without_dimensions_returns_nulls(db: Connection, client: TestClient):
    camera_id = add_camera(db, "6215")
    add_product(db)
    area_id = add_area(db, camera_id, 1)
    frame_id = add_frame(db, camera_id, width=None, height=None)
    add_prediction(db, add_run(db), frame_id, area_id, camera_id)
    response = client.get(f"/predictions/{frame_id}/{area_id}/frame")
    assert response.status_code == 200
    assert response.json()["width"] is None
    assert response.json()["height"] is None


@pytest.mark.parametrize("frame_id", ["0", "-1", "invalid"])
def test_invalid_frame_id_returns_422(db: Connection, client: TestClient, frame_id: str):
    add_run(db)
    assert client.get(f"/predictions/{frame_id}/6215_1/frame").status_code == 422
