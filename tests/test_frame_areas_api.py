from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection

from config.db.connection import get_connection
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

URL = "/cameras/6215/frames/1751052241.jpg/areas"


@pytest.fixture
def client(db: Connection) -> Iterator[TestClient]:
    app.dependency_overrides[get_connection] = lambda: db
    try:
        with TestClient(app) as value:
            yield value
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def areas(db: Connection, camera: int) -> dict[int, int]:
    second = add_area(db, camera, 2)
    first = add_area(db, camera, 1, top_left=[-12, 350])
    return {1: first, 2: second}


@pytest.fixture
def frame(db: Connection, camera: int, areas: dict[int, int]) -> int:
    return add_frame(db, camera)


def test_unknown_frame(db: Connection, client: TestClient, dataset: int, camera: int):
    add_run(db, dataset, status="done")
    response = client.get(URL)
    assert response.status_code == 404
    assert response.json() == {"detail": "Frame not found"}


@pytest.mark.parametrize("status", [None, "running", "failed"])
def test_no_completed_runs(db: Connection, client: TestClient, dataset: int, status: str | None):
    if status is not None:
        add_run(db, dataset, status=status)
    response = client.get(URL)
    assert response.status_code == 404
    assert response.json() == {"detail": "No completed runs available"}


@pytest.mark.parametrize("status", ["running", "failed", "done"])
def test_latest_completed_run_is_global_and_never_fills_from_history(
    db: Connection,
    client: TestClient,
    dataset: int,
    camera: int,
    frame: int,
    areas: dict[int, int],
    status: str,
):
    now = datetime.now(UTC)
    old = add_run(db, dataset, started_at=now - timedelta(hours=1), status="done")
    latest = add_run(db, dataset, started_at=now, status="done")
    newest = add_run(db, dataset, started_at=now + timedelta(seconds=1), status=status)
    for area in areas.values():
        add_prediction(db, old, frame, area, camera, stock_level="full")
    add_prediction(
        db,
        latest,
        frame,
        areas[2],
        camera,
        stock_level="half",
        compliance_confidence=None,
        stock_confidence=None,
        is_fallback=True,
    )
    other_camera = add_camera(db, dataset, "6383")
    other_area = add_area(db, other_camera, 1)
    other_frame = add_frame(db, other_camera)
    add_prediction(db, latest, other_frame, other_area, other_camera)
    if status != "done":
        add_prediction(db, newest, frame, areas[1], camera, stock_level="empty")

    response = client.get(URL)

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == (newest if status == "done" else latest)
    assert set(body) == {"run_id", "camera_id", "filename", "ts", "areas"}
    assert body["camera_id"] == "6215"
    assert body["filename"] == "1751052241.jpg"
    assert body["ts"] == datetime.fromtimestamp(1751052241, UTC).isoformat().replace("+00:00", "Z")
    assert [area["area_id"] for area in body["areas"]] == ["6215_1", "6215_2"]
    assert body["areas"][0]["top_left"] == [-12, 350]
    assert body["areas"][0]["prediction"] is None
    if status == "done":
        assert body["areas"][1]["prediction"] is None
        return
    area = body["areas"][1]
    assert area["product_id"] == "11908"
    assert area["product_name"] == "BANANA CAT I CNT KG"
    assert area["prediction"] == {
        "planogram_compliant": True,
        "stock_level": "half",
        "stock_value": 0.5,
        "detected_product_id": None,
        "compliance_confidence": None,
        "stock_confidence": None,
        "is_fallback": True,
    }


def test_latest_empty_run_does_not_return_old_predictions(
    db: Connection,
    client: TestClient,
    dataset: int,
    camera: int,
    frame: int,
    areas: dict[int, int],
):
    now = datetime.now(UTC)
    old = add_run(db, dataset, started_at=now, status="done")
    add_prediction(db, old, frame, areas[1], camera)
    latest = add_run(db, dataset, started_at=now, status="done")
    body = client.get(URL).json()
    assert body["run_id"] == latest
    assert all(area["prediction"] is None for area in body["areas"])


def test_started_at_takes_priority_over_run_id(
    db: Connection,
    client: TestClient,
    dataset: int,
    frame: int,
):
    now = datetime.now(UTC)
    latest = add_run(db, dataset, started_at=now, status="done")
    add_run(db, dataset, started_at=now - timedelta(hours=1), status="done")
    assert client.get(URL).json()["run_id"] == latest


def test_run_is_resolved_again_for_each_request(
    db: Connection,
    client: TestClient,
    dataset: int,
    frame: int,
):
    now = datetime.now(UTC)
    old = add_run(db, dataset, started_at=now, status="done")
    assert client.get(URL).json()["run_id"] == old
    latest = add_run(db, dataset, started_at=now + timedelta(seconds=1), status="done")
    assert client.get(URL).json()["run_id"] == latest


def test_frame_without_areas_or_image_size(
    db: Connection,
    client: TestClient,
    dataset: int,
    camera: int,
):
    add_frame(db, camera, width=None, height=None)
    latest = add_run(db, dataset, status="done")
    body = client.get(URL).json()
    assert body["run_id"] == latest
    assert body["areas"] == []


def test_unknown_camera(db: Connection, client: TestClient, dataset: int):
    add_run(db, dataset, status="done")
    response = client.get(URL)
    assert response.status_code == 404
    assert response.json() == {"detail": "Camera not found"}


@pytest.mark.parametrize("level,value", [("empty", 0), ("half", 0.5), ("full", 1)])
def test_detected_product_uses_external_id_and_stock_value_is_derived(
    db: Connection,
    client: TestClient,
    dataset: int,
    camera: int,
    frame: int,
    areas: dict[int, int],
    level: str,
    value: float,
):
    detected = add_product(db, dataset, "99999")
    run = add_run(db, dataset, status="done")
    add_prediction(
        db, run, frame, areas[1], camera, detected_product_id=detected, stock_level=level
    )

    response = client.get(URL)

    assert response.status_code == 200
    assert response.json()["areas"][0]["prediction"]["detected_product_id"] == "99999"
    assert response.json()["areas"][0]["prediction"]["stock_value"] == value


@pytest.mark.parametrize("missing", ["camera", "frame", None])
def test_latest_run_dataset_scopes_frame_lookup(
    db: Connection,
    client: TestClient,
    dataset: int,
    camera: int,
    frame: int,
    areas: dict[int, int],
    missing: str | None,
):
    old = add_run(db, dataset, status="done")
    add_prediction(db, old, frame, areas[1], camera)
    other_dataset = add_dataset(db)
    latest = add_run(db, other_dataset, status="done")
    if missing != "camera":
        add_product(db, other_dataset)
        other_camera = add_camera(db, other_dataset, "6215")
        other_area = add_area(db, other_camera, 3)
        if missing != "frame":
            other_frame = add_frame(db, other_camera)
            add_prediction(db, latest, other_frame, other_area, other_camera, stock_level="empty")

    response = client.get(URL)
    if missing is not None:
        assert response.status_code == 404
        assert response.json() == {"detail": f"{missing.capitalize()} not found"}
        return

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == latest
    assert [area["area_id"] for area in body["areas"]] == ["6215_3"]
    assert body["areas"][0]["prediction"]["stock_level"] == "empty"


def test_predictions_are_scoped_to_requested_frame(
    db: Connection,
    client: TestClient,
    dataset: int,
    camera: int,
    frame: int,
    areas: dict[int, int],
):
    other_frame = add_frame(db, camera, ts=1751052242)
    run = add_run(db, dataset, status="done")
    add_prediction(db, run, other_frame, areas[1], camera)
    response = client.get(URL)
    assert response.status_code == 200
    assert all(area["prediction"] is None for area in response.json()["areas"])
