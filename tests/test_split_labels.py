import json

import pytest
import typer

from src.split_labels import held_out_frames, main


def keys(camera: str, timestamps: list[int], areas: int = 2) -> list[str]:
    return [f"{camera}/{ts}.jpg/{camera}_{n}" for ts in timestamps for n in range(1, areas + 1)]


def test_latest_fifth_of_each_camera_is_test():
    labelled = keys("6215", [500, 100, 300, 200, 400, 900, 700, 600, 800, 1000])
    labelled += keys("6371", [10, 20, 30, 40, 50])
    assert held_out_frames(labelled) == {
        "6215/900.jpg",
        "6215/1000.jpg",
        "6371/50.jpg",
    }


def test_timestamps_order_by_number_not_text():
    labelled = keys("6215", [99, 100, 98, 97, 96])
    assert held_out_frames(labelled) == {"6215/100.jpg"}


def test_camera_with_one_frame_stays_in_training():
    labelled = keys("6215", [100]) + keys("6371", [1, 2, 3, 4, 5])
    assert held_out_frames(labelled) == {"6371/5.jpg"}


def test_cli_writes_sorted_test_frames(tmp_path):
    labels = {
        k: {"planogram_compliant": True, "stock_level": "full"}
        for k in keys("6215", [1, 2, 3, 4, 5])
    }
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    main(labels=tmp_path / "labels.json", out=tmp_path / "split.json")
    assert json.loads((tmp_path / "split.json").read_text()) == {"test_frames": ["6215/5.jpg"]}


def test_cli_keeps_an_existing_split_frozen(tmp_path):
    labels = {
        k: {"planogram_compliant": True, "stock_level": "full"}
        for k in keys("6215", [1, 2, 3, 4, 5])
    }
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "split.json").write_text(json.dumps({"test_frames": ["6215/1.jpg"]}))
    with pytest.raises(typer.BadParameter, match="frozen"):
        main(labels=tmp_path / "labels.json", out=tmp_path / "split.json")
    assert json.loads((tmp_path / "split.json").read_text()) == {"test_frames": ["6215/1.jpg"]}


def test_cli_rewrites_the_split_only_with_force(tmp_path):
    labels = {
        k: {"planogram_compliant": True, "stock_level": "full"}
        for k in keys("6215", [1, 2, 3, 4, 5])
    }
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "split.json").write_text(json.dumps({"test_frames": ["6215/1.jpg"]}))
    main(labels=tmp_path / "labels.json", out=tmp_path / "split.json", force=True)
    assert json.loads((tmp_path / "split.json").read_text()) == {"test_frames": ["6215/5.jpg"]}
