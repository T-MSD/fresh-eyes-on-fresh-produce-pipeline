import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2

FRAME_NAME = re.compile(r"[0-9]+\.jpe?g", re.IGNORECASE)


@dataclass(frozen=True)
class FrameFile:
    camera: str
    filename: str
    ts: datetime
    size: tuple[int, int] | None


@dataclass(frozen=True)
class FrameScan:
    images_dir: Path
    frames: tuple[FrameFile, ...]
    skipped: tuple[Path, ...]


def scan_frames(images_dir: Path) -> FrameScan:
    """Find `<images_dir>/<camera>/<unix_ts>.jpg`; anything else lands in `skipped`.

    An image that cannot be decoded is kept with `size=None`, so its areas still get a fallback.
    """
    frames: list[FrameFile] = []
    skipped: list[Path] = []
    for camera_dir in sorted(images_dir.iterdir()):
        if camera_dir.name.startswith(".") or not camera_dir.is_dir():
            skipped.append(camera_dir)
            continue
        for path in sorted(camera_dir.iterdir()):
            ts = frame_ts(path)
            if ts is None:
                skipped.append(path)
                continue
            frames.append(FrameFile(camera_dir.name, path.name, ts, image_size(path)))
    return FrameScan(images_dir, tuple(frames), tuple(skipped))


def frame_ts(path: Path) -> datetime | None:
    if not path.is_file() or not FRAME_NAME.fullmatch(path.name):
        return None
    try:
        return datetime.fromtimestamp(int(path.stem), UTC)
    except (OverflowError, OSError, ValueError):
        return None


def image_size(path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height
