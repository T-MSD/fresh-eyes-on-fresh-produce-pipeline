from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingest.frames import FrameFile, scan_frames
from tests.builders import declare_size, encode, orient, write_frame, write_frame_bytes


def test_finds_frames_per_camera_with_timestamp_and_size(tmp_path: Path):
    write_frame(tmp_path, "6215", 1751052241, width=8, height=6)
    write_frame(tmp_path, "6383", 1751001268)

    scan = scan_frames(tmp_path)

    assert scan.images_dir == tmp_path
    assert scan.frames == (
        FrameFile(
            camera="6215",
            filename="1751052241.jpg",
            ts=datetime.fromtimestamp(1751052241, UTC),
            size=(8, 6),
        ),
        FrameFile(
            camera="6383",
            filename="1751001268.jpg",
            ts=datetime.fromtimestamp(1751001268, UTC),
            size=(4, 3),
        ),
    )
    assert scan.skipped == ()


VALID = encode(width=8, height=6)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not an image", id="not-an-image"),
        pytest.param(b"\xff\xd8", id="start-marker-only"),
        pytest.param(VALID[: VALID.index(b"\xff\xc0") + 6], id="header-cut-in-size"),
        pytest.param(VALID[: len(VALID) // 2], id="data-cut-off"),
        pytest.param(declare_size(VALID, 0, 0), id="declares-zero-size"),
        pytest.param(declare_size(VALID, 30000, 30000), id="declares-oversized"),
        pytest.param(declare_size(VALID, 65535, 48), id="declares-huge-width"),
        pytest.param(encode(extension=".png"), id="png-named-jpg"),
    ],
)
def test_keeps_unreadable_image_without_size(tmp_path: Path, data: bytes):
    write_frame_bytes(tmp_path, "6215", 1751052241, data)

    (frame,) = scan_frames(tmp_path).frames

    assert frame.filename == "1751052241.jpg"
    assert frame.size is None


def test_reads_size_of_jpeg_with_extra_segments_before_the_frame_header(tmp_path: Path):
    comment = b"\xff\xfe\x00\x08camera"
    write_frame_bytes(tmp_path, "6215", 1751052241, VALID[:2] + comment + VALID[2:])

    (frame,) = scan_frames(tmp_path).frames

    assert frame.size == (8, 6)


@pytest.mark.parametrize(
    ("orientation", "size"),
    [(1, (8, 6)), (3, (8, 6)), (5, (6, 8)), (6, (6, 8)), (7, (6, 8)), (8, (6, 8))],
)
def test_stores_size_after_exif_rotation(tmp_path: Path, orientation: int, size: tuple[int, int]):
    write_frame_bytes(tmp_path, "6215", 1751052241, orient(VALID, orientation))

    (frame,) = scan_frames(tmp_path).frames

    assert frame.size == size


def test_accepts_jpg_and_jpeg_in_any_case_and_keeps_the_filename(tmp_path: Path):
    for name in ("1751001268.JPG", "1751005412.jpeg", "1751009590.Jpg"):
        write_frame(tmp_path, "6215", 0).rename(tmp_path / "6215" / name)

    scan = scan_frames(tmp_path)

    assert [f.filename for f in scan.frames] == [
        "1751001268.JPG",
        "1751005412.jpeg",
        "1751009590.Jpg",
    ]
    assert scan.skipped == ()


def test_skips_files_that_are_not_timestamped_jpgs(tmp_path: Path):
    write_frame(tmp_path, "6215", 1751052241)
    camera_dir = tmp_path / "6215"
    for name in (".DS_Store", "notes.txt", "frame.jpg", "1751052241.png", "1751052241.jpg.bak"):
        (camera_dir / name).write_bytes(b"")
    (camera_dir / "nested").mkdir()
    (tmp_path / "readme.md").write_bytes(b"")
    (tmp_path / ".hidden").mkdir()

    scan = scan_frames(tmp_path)

    assert [f.filename for f in scan.frames] == ["1751052241.jpg"]
    assert {p.name for p in scan.skipped} == {
        ".DS_Store",
        "notes.txt",
        "frame.jpg",
        "1751052241.png",
        "1751052241.jpg.bak",
        "nested",
        "readme.md",
        ".hidden",
    }


def test_skips_timestamp_out_of_range(tmp_path: Path):
    path = tmp_path / "6215" / f"{10**20}.jpg"
    path.parent.mkdir()
    path.write_bytes(b"")

    scan = scan_frames(tmp_path)

    assert scan.frames == ()
    assert scan.skipped == (path,)


def test_empty_folder_has_no_frames(tmp_path: Path):
    assert scan_frames(tmp_path).frames == ()
