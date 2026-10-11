from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection

from config.db.schema import RunStatus
from src.main import create_app
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
def client(db: Connection):
    with TestClient(create_app(db)) as client:
        yield client


@pytest.fixture
def prediction_ids(db: Connection, dataset: int):
    camera_id = add_camera(db, dataset, "6215")
    add_product(db, dataset)
    area_id = add_area(db, camera_id, 1)
    frame_id = add_frame(db, camera_id)
    return camera_id, frame_id, area_id


def test_no_run_returns_404(client: TestClient):
    response = client.get("/predictions/1/6215_1/frame")
    assert response.status_code == 404
    assert response.json()["detail"] == "No run available"


def test_unknown_prediction_returns_404(db: Connection, client: TestClient, dataset: int):
    add_run(db, dataset)
    response = client.get("/predictions/1/6215_1/frame")
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found in the most recent run"


@pytest.mark.parametrize("status", ["running", "done", "failed"])
def test_returns_frame_from_newest_run(
    db: Connection, client: TestClient, dataset: int, prediction_ids, status: RunStatus
):
    camera_id, frame_id, area_id = prediction_ids
    started_at = datetime(2026, 10, 10, tzinfo=UTC)
    newest = add_run(db, dataset, started_at=started_at, status=status)
    add_prediction(db, newest, frame_id, area_id, camera_id)
    add_run(db, dataset, started_at=started_at - timedelta(hours=1), status="done")

    response = client.get(f"/predictions/{frame_id}/6215_1/frame")

    assert response.status_code == 200
    assert response.json() == {
        "run_id": newest,
        "id": frame_id,
        "camera_id": "6215",
        "filename": "1751052241.jpg",
        "ts": "2025-06-27T19:24:01Z",
        "width": 1280,
        "height": 720,
    }


def test_does_not_fall_back_to_an_older_run(
    db: Connection, client: TestClient, dataset: int, prediction_ids
):
    camera_id, frame_id, area_id = prediction_ids
    old = add_run(db, dataset, started_at=datetime(2026, 10, 9, tzinfo=UTC))
    add_prediction(db, old, frame_id, area_id, camera_id)
    add_run(db, dataset, started_at=datetime(2026, 10, 10, tzinfo=UTC))
    response = client.get(f"/predictions/{frame_id}/6215_1/frame")
    assert response.status_code == 404


def test_tied_start_times_use_highest_run_id(
    db: Connection, client: TestClient, dataset: int, prediction_ids
):
    camera_id, frame_id, area_id = prediction_ids
    started_at = datetime(2026, 10, 10, tzinfo=UTC)
    old = add_run(db, dataset, started_at=started_at)
    add_prediction(db, old, frame_id, area_id, camera_id)
    add_run(db, dataset, started_at=started_at)
    assert client.get(f"/predictions/{frame_id}/6215_1/frame").status_code == 404


def test_new_run_is_selected_on_each_request(
    db: Connection, client: TestClient, dataset: int, prediction_ids
):
    camera_id, frame_id, area_id = prediction_ids
    old = add_run(db, dataset, started_at=datetime(2026, 10, 9, tzinfo=UTC))
    add_prediction(db, old, frame_id, area_id, camera_id)
    path = f"/predictions/{frame_id}/6215_1/frame"
    assert client.get(path).status_code == 200
    add_run(db, dataset, started_at=datetime(2026, 10, 10, tzinfo=UTC))
    assert client.get(path).status_code == 404


def test_frame_without_dimensions_returns_nulls(db: Connection, client: TestClient, dataset: int):
    camera_id = add_camera(db, dataset, "6215")
    add_product(db, dataset)
    area_id = add_area(db, camera_id, 1)
    frame_id = add_frame(db, camera_id, width=None, height=None)
    add_prediction(db, add_run(db, dataset), frame_id, area_id, camera_id)
    response = client.get(f"/predictions/{frame_id}/6215_1/frame")
    assert response.status_code == 200
    assert response.json()["width"] is None
    assert response.json()["height"] is None


@pytest.mark.parametrize("frame_id", ["0", "-1", "invalid"])
def test_invalid_frame_id_returns_422(
    db: Connection, client: TestClient, dataset: int, frame_id: str
):
    add_run(db, dataset)
    assert client.get(f"/predictions/{frame_id}/6215_1/frame").status_code == 422


def test_same_external_area_id_in_another_dataset_is_not_used(
    db: Connection, client: TestClient, dataset: int, prediction_ids
):
    camera_id, frame_id, area_id = prediction_ids
    old = add_run(db, dataset, started_at=datetime(2026, 10, 9, tzinfo=UTC))
    add_prediction(db, old, frame_id, area_id, camera_id)
    later_dataset = add_dataset(db)
    add_product(db, later_dataset)
    later_camera = add_camera(db, later_dataset, "6215")
    later_area = add_area(db, later_camera, 1)
    later_frame = add_frame(db, later_camera)
    newest = add_run(db, later_dataset, started_at=datetime(2026, 10, 10, tzinfo=UTC))
    add_prediction(db, newest, later_frame, later_area, later_camera)

    assert client.get(f"/predictions/{frame_id}/6215_1/frame").status_code == 404
    response = client.get(f"/predictions/{later_frame}/6215_1/frame")
    assert response.status_code == 200
    assert response.json()["run_id"] == newest
    assert response.json()["camera_id"] == "6215"
