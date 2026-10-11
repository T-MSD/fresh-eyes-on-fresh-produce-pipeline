from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, update

from config.db.connection import get_connection
from config.db.schema import Dataset
from src.main import create_app
from tests.builders import CORNERS, write_frame
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
    app = create_app()
    app.dependency_overrides[get_connection] = lambda: db
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    "endpoint", ["/cameras/6215/frame", "/images/6215/1751052241.jpg", "/areas/6215_1/stock"]
)
@pytest.mark.parametrize("status", [None, "running", "failed"])
def test_without_completed_run(db, client, dataset, endpoint, status):
    if status is not None:
        add_run(db, dataset, status=status)
    response = client.get(endpoint)
    assert response.status_code == 404
    assert response.json() == {"detail": "No completed runs available"}


def test_all_areas_and_nullable_predictions(db, client, dataset, camera):
    area = add_area(db, camera, 1, top_left=[-20, 350])
    add_area(db, camera, 2)
    other_camera = add_camera(db, dataset, "6383")
    add_area(db, other_camera, 1)
    frame = add_frame(db, camera, width=None, height=None)
    run = add_run(db, dataset, status="done")
    detected = add_product(db, dataset, "other-product")
    add_prediction(
        db,
        run,
        frame,
        area,
        camera,
        stock_level="half",
        detected_product_id=detected,
        compliance_confidence=None,
        stock_confidence=None,
    )
    response = client.get("/cameras/6215/frame")
    assert response.status_code == 200
    assert response.json() == {
        "camera": "6215",
        "camera_name": None,
        "filename": "1751052241.jpg",
        "ts": "2025-06-27T19:24:01Z",
        "width": None,
        "height": None,
        "areas": [
            {
                "area_id": "6215_1",
                "number": 1,
                "product_id": "11908",
                "product_name": "BANANA CAT I CNT KG",
                "corners": {**CORNERS, "top_left": [-20, 350]},
                "prediction": {
                    "planogram_compliant": True,
                    "stock_level": "half",
                    "stock_value": 0.5,
                    "detected_product_id": "other-product",
                    "detected_product_name": "BANANA CAT I CNT KG",
                    "compliance_confidence": None,
                    "stock_confidence": None,
                },
            },
            {
                "area_id": "6215_2",
                "number": 2,
                "product_id": "11908",
                "product_name": "BANANA CAT I CNT KG",
                "corners": CORNERS,
                "prediction": None,
            },
        ],
    }


@pytest.mark.parametrize("status", ["running", "failed", "done"])
@pytest.mark.parametrize("tied", [False, True])
def test_latest_completed_run_no_prediction_fallback(db, client, dataset, camera, status, tied):
    area = add_area(db, camera, 1)
    frame = add_frame(db, camera)
    started = datetime(2026, 10, 10, tzinfo=UTC)
    old = add_run(db, dataset, started_at=started, status="done")
    add_prediction(db, old, frame, area, camera)
    add_run(
        db,
        dataset,
        started_at=started if tied else started + timedelta(hours=1),
        status=status,
    )
    response = client.get("/cameras/6215/frame")
    assert response.status_code == 200
    prediction = response.json()["areas"][0]["prediction"]
    assert (prediction is None) == (status == "done")


def test_dataset_scoping_and_run_reselection(db, client, dataset, camera, tmp_path):
    add_area(db, camera, 1)
    add_frame(db, camera)
    started = datetime(2026, 10, 10, tzinfo=UTC)
    add_run(db, dataset, started_at=started, status="done")
    path = "/cameras/6215/frame"
    assert client.get(path).json()["width"] == 1280
    newer = add_dataset(db, images_dir=str(tmp_path))
    add_product(db, newer)
    new_camera = add_camera(db, newer, "6215")
    add_area(db, new_camera, 2)
    add_frame(db, new_camera, width=4, height=3)
    assert client.get(path).json()["width"] == 1280
    add_run(db, newer, started_at=started + timedelta(hours=1), status="done")
    payload = client.get(path).json()
    assert payload["width"] == 4
    assert [area["area_id"] for area in payload["areas"]] == ["6215_2"]


@pytest.mark.parametrize("endpoint", ["cameras", "images"])
@pytest.mark.parametrize("camera_name,filename", [("unknown", "1751052241.jpg"), ("6215", "0.jpg")])
def test_unknown_frame(db, client, dataset, camera, endpoint, camera_name, filename):
    add_frame(db, camera)
    add_run(db, dataset, status="done")
    path = (
        f"/images/{camera_name}/{filename}"
        if endpoint == "images"
        else f"/cameras/{camera_name}/frame?at=1970-01-01T00:00:00Z"
        if filename == "0.jpg"
        else f"/cameras/{camera_name}/frame"
    )
    assert client.get(path).status_code == 404


def test_camera_without_areas(db, client, dataset, camera):
    add_frame(db, camera)
    add_run(db, dataset, status="done")
    assert client.get("/cameras/6215/frame").json()["areas"] == []


def test_image_uses_selected_dataset_stored_frame(db, client, dataset, camera, tmp_path):
    db.execute(update(Dataset).where(Dataset.id == dataset).values(images_dir=str(tmp_path)))
    image = write_frame(tmp_path, "6215", 1751052241)
    add_frame(db, camera)
    add_run(db, dataset, status="done")
    response = client.get("/images/6215/1751052241.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == image.read_bytes()


def test_missing_image_returns_404(db, client, dataset, camera, tmp_path):
    db.execute(update(Dataset).where(Dataset.id == dataset).values(images_dir=str(tmp_path)))
    add_frame(db, camera)
    add_run(db, dataset, status="done")
    assert client.get("/images/6215/1751052241.jpg").status_code == 404
    assert client.get("/cameras/6215/frame").status_code == 200


def test_image_symlink_cannot_escape_dataset(db, client, dataset, camera, tmp_path):
    images = tmp_path / "images"
    db.execute(update(Dataset).where(Dataset.id == dataset).values(images_dir=str(images)))
    outside = write_frame(tmp_path / "outside", "6215", 1751052241)
    (images / "6215").mkdir(parents=True)
    (images / "6215" / outside.name).symlink_to(outside)
    add_frame(db, camera)
    add_run(db, dataset, status="done")
    assert client.get("/images/6215/1751052241.jpg").status_code == 404


@pytest.mark.parametrize("filename", ["..%2Fsecret.jpg", "%2Fetc%2Fpasswd"])
def test_path_traversal_is_rejected(db, client, dataset, filename):
    add_run(db, dataset, status="done")
    assert client.get(f"/images/6215/{filename}").status_code == 404


def test_routes_use_camera_and_filename():
    paths = create_app().openapi()["paths"]
    assert "/cameras/{camera}/frame" in paths
    assert "/areas/{area_id}/stock" in paths
    assert "/images/{camera}/{filename}" in paths


def test_frontend_origin_can_read_camera_api(db, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173")
    app = create_app()
    app.dependency_overrides[get_connection] = lambda: db
    with TestClient(app) as client:
        response = client.get("/cameras/6215/frame", headers={"Origin": "http://localhost:5173"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


@pytest.mark.parametrize(
    "at,expected",
    [
        (None, "1751055841.jpg"),
        ("2025-06-27T20:24:01Z", "1751055841.jpg"),
        ("2025-06-27T19:30:00Z", "1751052241.jpg"),
        ("2025-06-27T19:24:00Z", None),
    ],
)
def test_frame_at_or_before_time(db, client, dataset, camera, at, expected):
    add_frame(db, camera)
    add_frame(db, camera, 1751055841)
    add_run(db, dataset, status="done")
    response = client.get("/cameras/6215/frame", params={} if at is None else {"at": at})
    assert response.status_code == (404 if expected is None else 200)
    if expected is not None:
        assert response.json()["filename"] == expected


@pytest.mark.parametrize("at", ["invalid", "2025-06-27T20:00:00", "1751052241.jpg"])
def test_invalid_time(db, client, dataset, camera, at):
    add_frame(db, camera)
    add_run(db, dataset, status="done")
    assert client.get("/cameras/6215/frame", params={"at": at}).status_code == 422


def test_stock_history_is_sorted_and_scoped(db, client, dataset, camera):
    area = add_area(db, camera, 1)
    add_area(db, camera, 2)
    started = datetime(2026, 10, 10, tzinfo=UTC)
    old = add_run(db, dataset, started_at=started, status="done")
    latest = add_run(db, dataset, started_at=started + timedelta(hours=1), status="done")
    for ts, level in [(1751055841, "full"), (1751052241, "empty")]:
        frame = add_frame(db, camera, ts)
        add_prediction(db, old, frame, area, camera, stock_level="half")
        add_prediction(db, latest, frame, area, camera, stock_level=level, stock_confidence=None)
    assert client.get("/areas/6215_1/stock").json() == [
        {
            "ts": "2025-06-27T19:24:01Z",
            "stock_level": "empty",
            "stock_value": 0.0,
            "stock_confidence": None,
        },
        {
            "ts": "2025-06-27T20:24:01Z",
            "stock_level": "full",
            "stock_value": 1.0,
            "stock_confidence": None,
        },
    ]
    assert client.get("/areas/6215_2/stock").json() == []
    assert client.get("/areas/unknown/stock").status_code == 404


def test_history_and_image_switch_to_selected_runs_dataset(db, client, dataset, camera, tmp_path):
    old_area = add_area(db, camera, 1)
    old_frame = add_frame(db, camera)
    started = datetime(2026, 10, 10, tzinfo=UTC)
    old = add_run(db, dataset, started_at=started, status="done")
    add_prediction(db, old, old_frame, old_area, camera, stock_level="half")
    images = tmp_path / "new"
    newer = add_dataset(db, images_dir=str(images))
    add_product(db, newer)
    new_camera = add_camera(db, newer, "6215")
    new_area = add_area(db, new_camera, 1)
    new_frame = add_frame(db, new_camera)
    image = write_frame(images, "6215", 1751052241)
    latest = add_run(db, newer, started_at=started + timedelta(hours=1), status="done")
    add_prediction(db, latest, new_frame, new_area, new_camera, stock_level="empty")
    response = client.get("/areas/6215_1/stock")
    assert response.status_code == 200
    assert [item["stock_level"] for item in response.json()] == ["empty"]
    assert client.get("/images/6215/1751052241.jpg").content == image.read_bytes()
    add_run(db, newer, started_at=started + timedelta(hours=2), status="done")
    assert client.get("/areas/6215_1/stock").json() == []
