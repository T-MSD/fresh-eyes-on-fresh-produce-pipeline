import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CORNERS = {
    "top_left": [299, 350],
    "top_right": [464, 347],
    "bottom_right": [479, 438],
    "bottom_left": [331, 438],
}


def area(
    area_id: str,
    product_id: str = "11908",
    product_name: str = "BANANA CAT I CNT KG",
    product_key: str = "product_id",
    **corners: list[int],
) -> dict[str, Any]:
    return {
        "area_id": area_id,
        product_key: product_id,
        "product_name": product_name,
        "image_coordinates": {**CORNERS, **corners},
    }


def planogram_json(
    cameras: dict[str, list[dict[str, Any]]], store: str = "Frutas e Legumes"
) -> bytes:
    stores = {
        store: {"cameras": {camera: {"products": areas} for camera, areas in cameras.items()}}
    }
    return json.dumps(stores).encode()


def encode(width: int = 4, height: int = 3, extension: str = ".jpg") -> bytes:
    ok, encoded = cv2.imencode(extension, np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def declare_size(jpeg: bytes, width: int, height: int) -> bytes:
    """Rewrite the size in a baseline JPEG header, leaving the image data as it was."""
    sof = jpeg.index(b"\xff\xc0")
    return jpeg[: sof + 5] + height.to_bytes(2, "big") + width.to_bytes(2, "big") + jpeg[sof + 9 :]


def orient(jpeg: bytes, orientation: int) -> bytes:
    """Add an EXIF segment holding only the orientation tag."""
    entry = b"\x01\x12\x00\x03\x00\x00\x00\x01" + orientation.to_bytes(2, "big") + b"\x00\x00"
    exif = b"Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08\x00\x01" + entry + b"\x00\x00\x00\x00"
    return jpeg[:2] + b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif + jpeg[2:]


def write_frame(images_dir: Path, camera: str, ts: int, width: int = 4, height: int = 3) -> Path:
    return write_frame_bytes(images_dir, camera, ts, encode(width, height))


def write_frame_bytes(images_dir: Path, camera: str, ts: int, data: bytes) -> Path:
    path = images_dir / camera / f"{ts}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_dataset(
    root: Path, cameras: dict[str, list[dict[str, Any]]], frames: dict[str, list[int]]
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    planogram_path = root / "planogram.json"
    planogram_path.write_bytes(planogram_json(cameras))
    images_dir = root / "images"
    images_dir.mkdir(exist_ok=True)
    for camera, timestamps in frames.items():
        for ts in timestamps:
            write_frame(images_dir, camera, ts)
    return planogram_path, images_dir
