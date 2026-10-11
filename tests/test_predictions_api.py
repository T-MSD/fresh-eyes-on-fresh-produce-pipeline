from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection

from tests.assertions import iso
from tests.db_builders import (
    add_area,
    add_camera,
    add_frame,
    add_prediction,
    add_product,
    add_run,
)

MORNING, NOON = 1751052241, 1751055841


@pytest.mark.parametrize("statuses", [[], ["running"], ["running", "failed"]])
def test_predictions_without_completed_runs(
    db: Connection, client: TestClient, dataset: int, statuses: list[str]
):
    for status in statuses:
        add_run(db, dataset, status=status)

    response = client.get("/predictions")

    assert response.status_code == 404
    assert response.json() == {"detail": "No completed runs available"}


def test_returns_the_latest_frame_of_each_camera(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id = add_run(db, dataset, status="done")
    detected = add_product(db, dataset, "11909")
    full, wrong, missing = (add_area(db, camera, number) for number in (1, 2, 3))
    morning, noon = add_frame(db, camera, MORNING), add_frame(db, camera, NOON)
    for area_id in (full, wrong, missing):
        add_prediction(db, run_id, morning, area_id, camera, stock_level="empty")
    add_prediction(db, run_id, noon, full, camera)
    add_prediction(
        db,
        run_id,
        noon,
        wrong,
        camera,
        planogram_compliant=False,
        stock_level="half",
        detected_product_id=detected,
        compliance_confidence=None,
        stock_confidence=None,
        is_fallback=True,
    )

    response = client.get("/predictions")

    assert response.status_code == 200
    product = {"product_id": "11908", "product_name": "BANANA CAT I CNT KG"}
    assert response.json() == {
        "cameras": [
            {
                "camera": "6215",
                "frame": {"filename": f"{NOON}.jpg", "ts": iso(NOON)},
                "areas": [
                    {
                        "area_id": "6215_1",
                        **product,
                        "status": "ok",
                        "prediction": {
                            "planogram_compliant": True,
                            "stock_level": "full",
                            "stock_value": 1.0,
                            "detected_product_id": None,
                            "detected_product_name": None,
                            "compliance_confidence": 0.9,
                            "stock_confidence": 0.8,
                        },
                    },
                    {
                        "area_id": "6215_2",
                        **product,
                        "status": "wrong_product",
                        "prediction": {
                            "planogram_compliant": False,
                            "stock_level": "half",
                            "stock_value": 0.5,
                            "detected_product_id": "11909",
                            "detected_product_name": "BANANA CAT I CNT KG",
                            "compliance_confidence": None,
                            "stock_confidence": None,
                        },
                    },
                    {"area_id": "6215_3", **product, "status": "unknown", "prediction": None},
                ],
            }
        ]
    }


@pytest.mark.parametrize(("at", "filename"), [(NOON - 1, MORNING), (NOON, NOON)])
def test_at_picks_the_latest_frame_at_or_before_it(
    db: Connection, client: TestClient, dataset: int, camera: int, at: int, filename: int
):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    for ts in (MORNING, NOON):
        add_prediction(db, run_id, add_frame(db, camera, ts), area_id, camera)

    [snapshot] = client.get("/predictions", params={"at": iso(at)}).json()["cameras"]

    assert snapshot["frame"]["filename"] == f"{filename}.jpg"


def test_camera_without_a_frame_by_at_is_left_out(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id, other = add_run(db, dataset, status="done"), add_camera(db, dataset, "6383")
    for camera_id, ts in ((camera, MORNING), (other, NOON)):
        add_prediction(
            db, run_id, add_frame(db, camera_id, ts), add_area(db, camera_id, 1), camera_id
        )

    cameras = client.get("/predictions", params={"at": iso(MORNING)}).json()["cameras"]

    assert [snapshot["camera"] for snapshot in cameras] == ["6215"]


def test_at_without_a_timezone_is_rejected(db: Connection, client: TestClient, dataset: int):
    add_run(db, dataset, status="done")

    response = client.get("/predictions", params={"at": "2025-06-27T19:24:01"})

    assert response.status_code == 422


def test_newest_frame_without_predictions_shows_unknown_areas(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    add_prediction(db, run_id, add_frame(db, camera, MORNING), area_id, camera)
    add_frame(db, camera, NOON)

    [snapshot] = client.get("/predictions").json()["cameras"]

    assert snapshot["frame"]["filename"] == f"{NOON}.jpg"
    assert snapshot["areas"] == [
        {
            "area_id": "6215_1",
            "product_id": "11908",
            "product_name": "BANANA CAT I CNT KG",
            "status": "unknown",
            "prediction": None,
        }
    ]


def test_empty_completed_run_never_uses_older_predictions(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    area_id, frame_id = add_area(db, camera, 1), add_frame(db, camera)
    older = add_run(db, dataset, started_at=datetime(2026, 1, 1, tzinfo=UTC), status="done")
    add_prediction(db, older, frame_id, area_id, camera, stock_level="empty")
    add_run(db, dataset, started_at=datetime(2026, 1, 2, tzinfo=UTC), status="done")

    [snapshot] = client.get("/predictions").json()["cameras"]

    assert [(area["status"], area["prediction"]) for area in snapshot["areas"]] == [
        ("unknown", None)
    ]


@pytest.mark.parametrize("at", [None, iso(MORNING), iso(NOON)])
def test_picks_the_same_frame_as_the_frame_view(
    db: Connection, client: TestClient, dataset: int, camera: int, at: str | None
):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    add_prediction(db, run_id, add_frame(db, camera, MORNING), area_id, camera)
    for name in ("x.jpg", "y.jpg"):
        add_frame(db, camera, NOON, filename=name)
    params = {} if at is None else {"at": at}

    [snapshot] = client.get("/predictions", params=params).json()["cameras"]
    frame_view = client.get("/cameras/6215/frame", params=params).json()

    assert snapshot["frame"]["filename"] == frame_view["filename"]


def test_frames_tied_on_time_use_the_highest_filename(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id, area_id = add_run(db, dataset, status="done"), add_area(db, camera, 1)
    for name in ("b.jpg", "a.jpg"):
        add_prediction(db, run_id, add_frame(db, camera, NOON, filename=name), area_id, camera)

    [snapshot] = client.get("/predictions").json()["cameras"]

    assert snapshot["frame"]["filename"] == "b.jpg"


@pytest.mark.parametrize(
    ("planogram_compliant", "stock_level", "status"),
    [
        (True, "full", "ok"),
        (True, "half", "half"),
        (True, "empty", "empty"),
        (False, "full", "wrong_product"),
        (False, "half", "wrong_product"),
        (False, "empty", "empty"),
    ],
)
def test_area_status_is_empty_then_wrong_product_then_half(
    db: Connection,
    client: TestClient,
    dataset: int,
    camera: int,
    planogram_compliant: bool,
    stock_level: str,
    status: str,
):
    run_id = add_run(db, dataset, status="done")
    add_prediction(
        db,
        run_id,
        add_frame(db, camera),
        add_area(db, camera, 1),
        camera,
        planogram_compliant=planogram_compliant,
        stock_level=stock_level,
    )

    [snapshot] = client.get("/predictions").json()["cameras"]

    assert [area["status"] for area in snapshot["areas"]] == [status]


def test_sorts_areas_by_number(db: Connection, client: TestClient, dataset: int, camera: int):
    run_id, frame_id = add_run(db, dataset, status="done"), add_frame(db, camera)
    for number in (10, 2):
        add_prediction(db, run_id, frame_id, add_area(db, camera, number), camera)

    [snapshot] = client.get("/predictions").json()["cameras"]

    assert [area["area_id"] for area in snapshot["areas"]] == ["6215_2", "6215_10"]


def test_sorts_cameras_most_urgent_first(db: Connection, client: TestClient, dataset: int):
    add_product(db, dataset)
    run_id = add_run(db, dataset, status="done")
    cameras = {
        "6371": [(True, "full")],
        "6215": [(True, "full")],
        "6386": [(True, "half")],
        "6594": [(False, "full"), (True, "half"), (True, "half"), (True, "half")],
        "6373": [(False, "full"), (False, "half")],
        "6443": [(True, "empty"), (True, "full")],
        "6100": [None],
    }
    for external_id, areas in cameras.items():
        camera_id = add_camera(db, dataset, external_id)
        frame_id = add_frame(db, camera_id)
        for number, prediction in enumerate(areas, start=1):
            area_id = add_area(db, camera_id, number)
            if prediction is not None:
                planogram_compliant, stock_level = prediction
                add_prediction(
                    db,
                    run_id,
                    frame_id,
                    area_id,
                    camera_id,
                    planogram_compliant=planogram_compliant,
                    stock_level=stock_level,
                )

    body = client.get("/predictions").json()

    assert [snapshot["camera"] for snapshot in body["cameras"]] == [
        "6443",
        "6373",
        "6594",
        "6386",
        "6100",
        "6215",
        "6371",
    ]


@pytest.mark.parametrize("newer_status", ["running", "failed"])
def test_uses_latest_completed_run_ignoring_newer_unfinished_runs(
    db: Connection, client: TestClient, dataset: int, camera: int, newer_status: str
):
    area_id, frame_id = add_area(db, camera, 1), add_frame(db, camera)
    runs = [
        (datetime(2026, 1, 1, tzinfo=UTC), "done", "empty"),
        (datetime(2026, 1, 2, tzinfo=UTC), "done", "half"),
        (datetime(2026, 1, 3, tzinfo=UTC), newer_status, "full"),
    ]
    for started_at, status, stock_level in runs:
        run_id = add_run(db, dataset, started_at=started_at, status=status)
        add_prediction(db, run_id, frame_id, area_id, camera, stock_level=stock_level)

    [snapshot] = client.get("/predictions").json()["cameras"]

    assert [area["prediction"]["stock_level"] for area in snapshot["areas"]] == ["half"]


def test_completed_runs_started_together_use_the_highest_id(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    area_id, frame_id = add_area(db, camera, 1), add_frame(db, camera)
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    for stock_level in ("empty", "full"):
        run_id = add_run(db, dataset, started_at=started_at, status="done")
        add_prediction(db, run_id, frame_id, area_id, camera, stock_level=stock_level)

    [snapshot] = client.get("/predictions").json()["cameras"]

    assert [area["prediction"]["stock_level"] for area in snapshot["areas"]] == ["full"]


def test_filters_by_camera(db: Connection, client: TestClient, dataset: int, camera: int):
    run_id, other = add_run(db, dataset, status="done"), add_camera(db, dataset, "6383")
    for camera_id in (camera, other):
        add_prediction(db, run_id, add_frame(db, camera_id), add_area(db, camera_id, 1), camera_id)

    cameras = client.get("/predictions", params={"camera": "6383"}).json()["cameras"]

    assert [snapshot["camera"] for snapshot in cameras] == ["6383"]


def test_unknown_camera_has_no_predictions(
    db: Connection, client: TestClient, dataset: int, camera: int
):
    run_id = add_run(db, dataset, status="done")
    add_prediction(db, run_id, add_frame(db, camera), add_area(db, camera, 1), camera)

    response = client.get("/predictions", params={"camera": "9999"})

    assert response.status_code == 200
    assert response.json() == {"cameras": []}
