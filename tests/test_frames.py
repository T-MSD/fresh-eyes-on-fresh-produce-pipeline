from datetime import UTC, datetime
from pathlib import Path

from ingest.frames import FrameFile, scan_frames
from tests.builders import write_frame


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


def test_keeps_unreadable_image_without_size(tmp_path: Path):
    path = tmp_path / "6215" / "1751052241.jpg"
    path.parent.mkdir()
    path.write_bytes(b"not an image")

    (frame,) = scan_frames(tmp_path).frames

    assert frame.filename == "1751052241.jpg"
    assert frame.size is None


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
