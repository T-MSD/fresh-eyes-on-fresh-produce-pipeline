from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection
from sqlalchemy.orm import Session

from src.api.dependencies import get_session
from src.main import app
from tests.test_db_constraints import (
    add_area,
    add_camera,
    add_frame,
    add_prediction,
    add_product,
    add_run,
)


@pytest.fixture
def client(db: Connection) -> Iterator[TestClient]:
    def session() -> Iterator[Session]:
        with Session(bind=db) as value:
            yield value

    app.dependency_overrides[get_session] = session
    with TestClient(app) as value:
        yield value
    app.dependency_overrides.clear()


@pytest.fixture
def frame(db: Connection) -> int:
    add_product(db)
    add_camera(db, "6215")
    add_area(db, "6215", 2)
    add_area(db, "6215", 1, top_left=[-12, 350])
    return add_frame(db, "6215")


def test_unknown_frame(client: TestClient):
    assert client.get("/frames/999/areas").status_code == 404


def test_no_runs(client: TestClient, frame: int):
    assert client.get(f"/frames/{frame}/areas").status_code == 503


@pytest.mark.parametrize("status", ["running", "failed", "done"])
def test_latest_run_is_global_and_never_fills_from_history(
    db: Connection, client: TestClient, frame: int, status: str
):
    now = datetime.now(UTC)
    old = add_run(db, started_at=now - timedelta(hours=1), status="done")
    latest = add_run(db, started_at=now, status=status)
    add_prediction(db, old, frame, "6215_1", "6215", stock_level="full")
    add_prediction(db, old, frame, "6215_2", "6215", stock_level="full")
    add_prediction(
        db,
        latest,
        frame,
        "6215_2",
        "6215",
        stock_level="half",
        compliance_confidence=None,
        stock_confidence=None,
        is_fallback=True,
    )
    add_camera(db, "6383")
    add_area(db, "6383", 1)
    other_frame = add_frame(db, "6383")
    add_prediction(db, latest, other_frame, "6383_1", "6383")

    response = client.get(f"/frames/{frame}/areas")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == latest
    assert body["frame_id"] == frame
    assert body["camera_id"] == "6215"
    assert body["filename"] == "1751052241.jpg"
    assert [area["area_id"] for area in body["areas"]] == ["6215_1", "6215_2"]
    assert body["areas"][0]["top_left"] == [-12, 350]
    assert body["areas"][0]["prediction"] is None
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
    db: Connection, client: TestClient, frame: int
):
    now = datetime.now(UTC)
    old = add_run(db, started_at=now)
    add_prediction(db, old, frame, "6215_1", "6215")
    latest = add_run(db, started_at=now)
    body = client.get(f"/frames/{frame}/areas").json()
    assert body["run_id"] == latest
    assert all(area["prediction"] is None for area in body["areas"])


def test_started_at_takes_priority_over_run_id(db: Connection, client: TestClient, frame: int):
    now = datetime.now(UTC)
    latest = add_run(db, started_at=now)
    add_run(db, started_at=now - timedelta(hours=1))
    assert client.get(f"/frames/{frame}/areas").json()["run_id"] == latest


def test_run_is_resolved_again_for_each_request(db: Connection, client: TestClient, frame: int):
    now = datetime.now(UTC)
    old = add_run(db, started_at=now)
    assert client.get(f"/frames/{frame}/areas").json()["run_id"] == old
    latest = add_run(db, started_at=now + timedelta(seconds=1))
    assert client.get(f"/frames/{frame}/areas").json()["run_id"] == latest


def test_frame_without_areas_or_image_size(db: Connection, client: TestClient):
    add_camera(db, "6215")
    frame = add_frame(db, "6215", width=None, height=None)
    latest = add_run(db)
    body = client.get(f"/frames/{frame}/areas").json()
    assert body["run_id"] == latest
    assert body["areas"] == []


@pytest.mark.parametrize("frame_id", ["abc", "0", "-1"])
def test_invalid_frame_id(client: TestClient, frame_id: str):
    assert client.get(f"/frames/{frame_id}/areas").status_code == 422
