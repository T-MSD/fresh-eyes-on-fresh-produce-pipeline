from datetime import UTC, datetime

import pytest
from sqlalchemy import Connection, select, text

from config.db.schema import Dataset, Frame, Run, Task
from tests.db_builders import add_area, add_predicted_frame, add_run, add_task

MOMENT = datetime(2025, 6, 27, 19, 24, 1, tzinfo=UTC)


@pytest.mark.parametrize("timezone", ["Europe/Lisbon", "America/New_York", "Asia/Tokyo"])
def test_every_timestamp_reads_back_in_utc(
    db: Connection, dataset: int, camera: int, timezone: str
):
    db.execute(text("SELECT set_config('TimeZone', :timezone, true)"), {"timezone": timezone})
    run_id = add_run(db, dataset, started_at=MOMENT, finished_at=MOMENT)
    area_id = add_area(db, camera, 1)
    frame_id = add_predicted_frame(db, run_id, area_id, camera, int(MOMENT.timestamp()))
    add_task(db, run_id, area_id, frame_id, done_at=MOMENT)

    values = [
        db.scalar(select(column))
        for column in (Dataset.loaded_at, Frame.ts, Run.started_at, Run.finished_at, Task.done_at)
    ]

    assert all(value.tzinfo is UTC for value in values)
    assert values[1:] == [MOMENT] * 4
