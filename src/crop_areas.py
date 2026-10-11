"""Cut every planogram area out of every camera image (perspective-corrected).

Writes out/crops/<area_id>/<unix_ts>.jpg and out/crops/index.json, which the review tool reads.

    uv run --env-file .env python -m src.crop_areas --data-dir <dataset-folder>
"""

import json
from pathlib import Path

import cv2
import numpy as np
import typer

from ingest.planogram import parse_planogram
from src.prediction_io import Label

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT.parent / "new_dataset_hackathon_junctionx_10_oct_2026_v2"
MAX_SIDE = 512
MIN_SIDE = 32
CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")

app = typer.Typer(add_completion=False)


def warp_area(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Warp a (possibly partly off-image) quadrilateral to an upright rectangle."""
    tl, tr, br, bl = quad
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    scale = min(1.0, MAX_SIDE / max(w, h, 1))
    w, h = max(int(w * scale), MIN_SIDE), max(int(h * scale), MIN_SIDE)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


@app.command()
def main(
    data_dir: Path = typer.Option(DEFAULT_DATA, help="Folder holding planogram.json and images"),
    out_dir: Path = typer.Option(ROOT / "out" / "crops", help="Where crops and index.json go"),
) -> None:
    planogram_file = next(
        (
            d / "planogram.json"
            for d in (data_dir, data_dir.parent)
            if (d / "planogram.json").exists()
        ),
        None,
    )
    if planogram_file is None:
        raise typer.BadParameter(f"planogram.json not found in {data_dir} or its parent")
    planogram = parse_planogram(planogram_file.read_bytes())
    names = {product.external_id: product.name for product in planogram.products}
    candidates = [data_dir, *sorted(p for p in data_dir.iterdir() if p.is_dir())]
    images_root = next(
        (c for c in candidates if any((c / cam).is_dir() for cam in planogram.cameras)), None
    )
    if images_root is None:
        raise typer.BadParameter(
            f"no camera folders ({', '.join(planogram.cameras)}) found in {data_dir}"
        )

    items = []
    fallbacks: dict = {}
    for camera in sorted(planogram.cameras):
        areas = [area for area in planogram.areas if area.camera == camera]
        for img_path in sorted((images_root / camera).glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                typer.echo(f"fallback for unreadable {img_path}", err=True)
                fallbacks.setdefault(camera, {})[img_path.name] = {
                    area.external_id: Label(
                        planogram_compliant=True, stock_level="empty"
                    ).model_dump()
                    for area in areas
                }
                continue
            for area in areas:
                quad = np.array([getattr(area.corners, c) for c in CORNERS], dtype=np.float32)
                rel = Path(area.external_id) / img_path.name
                (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                written = cv2.imwrite(
                    str(out_dir / rel), warp_area(img, quad), [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                if not written:
                    raise OSError(f"Could not write crop {out_dir / rel}")
                items.append(
                    {
                        "key": f"{camera}/{img_path.name}/{area.external_id}",
                        "camera": camera,
                        "image": img_path.name,
                        "image_path": str(img_path),
                        "area_id": area.external_id,
                        "product_id": area.product,
                        "product_name": names[area.product],
                        "polygon": quad.tolist(),
                        "image_size": [img.shape[1], img.shape[0]],
                        "crop": rel.as_posix(),
                    }
                )
        typer.echo(f"camera {camera} done")

    if not items and not fallbacks:
        typer.echo(f"0 crops made - no .jpg files found under {images_root}", err=True)
        raise typer.Exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(items))
    (out_dir / "fallbacks.json").write_text(json.dumps(fallbacks))
    typer.echo(f"{len(items)} crops -> {out_dir}")


if __name__ == "__main__":
    app()
