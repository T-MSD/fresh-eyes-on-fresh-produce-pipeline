import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

FRAME_NAME = re.compile(r"[0-9]+\.jpe?g", re.IGNORECASE)
MAX_FRAME_PIXELS = 4096 * 4096
FRAME_HEADERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
STANDALONE_MARKERS = frozenset({0x01, *range(0xD0, 0xD8)})


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
    """(width, height) of a JPEG that decodes to the size its header declares, else None.

    The header is checked before decoding, so a header claiming a huge size is never decoded.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    declared = jpeg_size(data)
    if declared is None or not 0 < declared[0] * declared[1] <= MAX_FRAME_PIXELS:
        return None
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    return (width, height) if (width, height) == declared else None


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from the first frame header of a JPEG, or None if there is none."""
    if not data.startswith(b"\xff\xd8"):
        return None
    position = 2
    while position + 4 <= len(data):
        if data[position] != 0xFF:
            return None
        marker = data[position + 1]
        if marker == 0xFF:
            position += 1
            continue
        if marker in STANDALONE_MARKERS:
            position += 2
            continue
        length = int.from_bytes(data[position + 2 : position + 4], "big")
        if length < 2:
            return None
        if marker in FRAME_HEADERS:
            if length < 8 or position + 9 > len(data):
                return None
            height = int.from_bytes(data[position + 5 : position + 7], "big")
            width = int.from_bytes(data[position + 7 : position + 9], "big")
            return width, height
        if marker in (0xD9, 0xDA):
            return None
        position += 2 + length
    return None
