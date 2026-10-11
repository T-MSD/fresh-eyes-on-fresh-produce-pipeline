from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

from config.db.schema import Camera
from tests.builders import CORNERS
from tests.db_builders import add_area, add_camera, add_dataset, add_frame, add_product, add_run


@pytest.mark.parametrize("status", [None, "running", "failed"])
def test_without_completed_run(db, client, dataset, status):
    if status is not None:
        add_run(db, dataset, status=status)
    response = client.get("/cameras")
    assert response.status_code == 404
    assert response.json() == {"detail": "No completed runs available"}


def test_every_camera_with_its_planogram(db, client, dataset, camera):
    db.execute(update(Camera).where(Camera.id == camera).values(name="Banana island"))
    add_area(db, camera, 2)
    add_area(db, camera, 1, top_left=[-20, 350])
    add_camera(db, dataset, "6383")
    add_frame(db, camera)
    add_run(db, dataset, status="done")
    response = client.get("/cameras")
    assert response.status_code == 200
    assert response.json() == {
        "cameras": [
            {
                "camera": "6215",
                "camera_name": "Banana island",
                "areas": [
                    {
                        "area_id": "6215_1",
                        "number": 1,
                        "product_id": "11908",
                        "product_name": "BANANA CAT I CNT KG",
                        "corners": {**CORNERS, "top_left": [-20, 350]},
                    },
                    {
                        "area_id": "6215_2",
                        "number": 2,
                        "product_id": "11908",
                        "product_name": "BANANA CAT I CNT KG",
                        "corners": CORNERS,
                    },
                ],
            },
            {"camera": "6383", "camera_name": None, "areas": []},
        ]
    }


def test_cameras_follow_the_latest_completed_run_dataset(db, client, dataset, camera):
    add_area(db, camera, 1)
    started = datetime(2026, 10, 10, tzinfo=UTC)
    add_run(db, dataset, started_at=started, status="done")
    newer = add_dataset(db)
    add_product(db, newer)
    add_area(db, add_camera(db, newer, "6376"), 3)
    assert [item["camera"] for item in client.get("/cameras").json()["cameras"]] == ["6215"]
    add_run(db, newer, started_at=started + timedelta(hours=1), status="done")
    cameras = client.get("/cameras").json()["cameras"]
    assert [item["camera"] for item in cameras] == ["6376"]
    assert [area["area_id"] for area in cameras[0]["areas"]] == ["6376_3"]
