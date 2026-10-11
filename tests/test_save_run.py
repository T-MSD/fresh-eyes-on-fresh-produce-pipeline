import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import Connection, select

from config.db.load import load_dataset
from config.db.save_run import UnusableRun, fallback_keys, latest_dataset_id, save_run
from config.db.schema import Prediction, Run
from ingest.dataset import read_dataset
from tests.builders import area, label, write_dataset


@pytest.fixture
def dataset_id(db: Connection, tmp_path: Path) -> int:
    dataset = read_dataset(
        *write_dataset(
            tmp_path,
            {"6215": [area("6215_1"), area("6215_2", "20923", "KIWI")], "6383": [area("6383_1")]},
            {"6215": [1751052241, 1751055841], "6383": [1751052241]},
        )
    )
    return load_dataset(db, dataset).dataset_id


def complete() -> dict:
    return {
        "6215": {
            "1751052241.jpg": {"6215_1": label(), "6215_2": label("half")},
            "1751055841.jpg": {"6215_1": label("empty", False), "6215_2": label()},
        },
        "6383": {"1751052241.jpg": {"6383_1": label()}},
    }


def test_stores_a_prediction_for_every_area_of_every_frame(db, dataset_id):
    run_id = save_run(db, dataset_id, complete(), method="run_test", params={"topk": 5})

    run = db.execute(select(Run).where(Run.id == run_id)).one()
    stored = db.execute(select(Prediction).where(Prediction.run_id == run_id)).all()
    assert (run.dataset_id, run.method, run.status, run.params) == (
        dataset_id,
        "run_test",
        "done",
        {"topk": 5},
    )
    assert run.finished_at is not None
    assert len(stored) == 5
    assert sorted((p.stock_level, p.planogram_compliant) for p in stored)[0] == ("empty", False)
    assert all(p.compliance_confidence is None and p.stock_confidence is None for p in stored)


def test_marks_only_the_fallback_areas(db, dataset_id):
    run_id = save_run(
        db,
        dataset_id,
        complete(),
        method="run_test",
        fallback={("6383", "1751052241.jpg", "6383_1")},
    )

    flagged = db.scalars(select(Prediction.is_fallback).where(Prediction.run_id == run_id)).all()
    assert sorted(flagged) == [False] * 4 + [True]


def test_refuses_a_run_that_is_missing_an_area_and_stores_nothing(db, dataset_id):
    predictions = complete()
    del predictions["6215"]["1751055841.jpg"]["6215_2"]

    with pytest.raises(UnusableRun) as error:
        save_run(db, dataset_id, predictions, method="run_test")

    assert "missing 6215/1751055841.jpg/6215_2" in error.value.problems
    assert db.scalars(select(Run.id)).all() == []


def test_refuses_a_frame_or_area_the_dataset_does_not_have(db, dataset_id):
    predictions = complete()
    predictions["6215"]["999.jpg"] = {"6215_1": label()}
    predictions["6383"]["1751052241.jpg"]["6383_9"] = label()

    with pytest.raises(UnusableRun) as error:
        save_run(db, dataset_id, predictions, method="run_test")

    assert "unknown 6215/999.jpg/6215_1" in error.value.problems
    assert "unknown 6383/1751052241.jpg/6383_9" in error.value.problems


def test_refuses_a_stock_level_outside_the_three_levels(db, dataset_id):
    predictions = complete()
    predictions["6383"]["1751052241.jpg"]["6383_1"] = label("plenty")

    with pytest.raises(ValidationError):
        save_run(db, dataset_id, predictions, method="run_test")


def test_refuses_an_unknown_dataset(db, dataset_id):
    with pytest.raises(UnusableRun, match="does not exist"):
        save_run(db, dataset_id + 1000, complete(), method="run_test")


def test_the_newest_load_is_the_default_dataset(db, dataset_id):
    assert latest_dataset_id(db) == dataset_id


def test_reads_the_fallback_areas_of_a_crop_areas_file(tmp_path):
    path = tmp_path / "fallbacks.json"
    path.write_text(
        json.dumps({"6383": {"1751052241.jpg": {"6383_1": {**label("empty"), "stock_value": 0.0}}}})
    )

    assert fallback_keys(path) == {("6383", "1751052241.jpg", "6383_1")}
