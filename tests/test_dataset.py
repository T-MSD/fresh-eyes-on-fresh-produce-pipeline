from pathlib import Path

import pytest

from ingest.dataset import read_dataset
from ingest.errors import InvalidInput
from tests.builders import area, write_dataset


def test_reads_planogram_and_frames_with_absolute_paths(tmp_path: Path, monkeypatch):
    write_dataset(tmp_path, {"6215": [area("6215_1")]}, {"6215": [1751052241, 1751055841]})
    monkeypatch.chdir(tmp_path)

    dataset = read_dataset(Path("planogram.json"), Path("images"))

    assert dataset.planogram_path == tmp_path.resolve() / "planogram.json"
    assert dataset.images_dir == tmp_path.resolve() / "images"
    assert dataset.planogram.cameras == ("6215",)
    assert [f.filename for f in dataset.frames] == ["1751052241.jpg", "1751055841.jpg"]


def test_lists_planogram_cameras_without_frames(tmp_path: Path):
    paths = write_dataset(
        tmp_path, {"6215": [area("6215_1")], "6383": [area("6383_1")]}, {"6215": [1751052241]}
    )
    assert read_dataset(*paths).cameras_without_frames == ("6383",)


def test_stops_on_camera_folder_missing_from_planogram(tmp_path: Path):
    paths = write_dataset(
        tmp_path, {"6215": [area("6215_1")]}, {"6215": [1751052241], "9999": [1751052241]}
    )
    with pytest.raises(InvalidInput) as error:
        read_dataset(*paths)
    assert "9999" in "\n".join(error.value.problems)


def test_stops_when_no_frames_are_found(tmp_path: Path):
    paths = write_dataset(tmp_path, {"6215": [area("6215_1")]}, {})
    with pytest.raises(InvalidInput) as error:
        read_dataset(*paths)
    assert "no frames" in "\n".join(error.value.problems)


def test_stops_on_invalid_planogram_before_scanning(tmp_path: Path):
    planogram_path, images_dir = write_dataset(tmp_path, {"6215": [area("6215_0")]}, {})
    with pytest.raises(InvalidInput) as error:
        read_dataset(planogram_path, images_dir)
    assert "6215_0" in "\n".join(error.value.problems)
