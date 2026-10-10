"""Cut every planogram area out of every camera image (perspective-corrected).

Writes out/crops/<area_id>/<unix_ts>.jpg and out/crops/index.json, which the review tool reads.

    python src/crop_areas.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import typer

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT.parent / "new_dataset_hackathon_junctionx_10_oct_2026_v2"
SECTION = "Frutas e Legumes"
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
    planogram = json.loads(planogram_file.read_text())[SECTION]["cameras"]
    # Images live in <root>/<camera_id>/*.jpg, where root is data_dir or one of its subfolders.
    candidates = [data_dir, *sorted(p for p in data_dir.iterdir() if p.is_dir())]
    images_root = next(
        (c for c in candidates if any((c / cam).is_dir() for cam in planogram)), None
    )
    if images_root is None:
        raise typer.BadParameter(f"no camera folders ({', '.join(planogram)}) found in {data_dir}")

    items = []
    for camera, cam in sorted(planogram.items()):
        for img_path in sorted((images_root / camera).glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is None:
                typer.echo(f"skip unreadable {img_path}", err=True)
                continue
            for area in cam["products"]:
                quad = np.array([area["image_coordinates"][c] for c in CORNERS], dtype=np.float32)
                rel = Path(area["area_id"]) / img_path.name
                (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(out_dir / rel), warp_area(img, quad), [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                items.append(
                    {
                        "key": f"{camera}/{img_path.name}/{area['area_id']}",
                        "camera": camera,
                        "image": img_path.name,
                        "image_path": str(img_path),
                        "area_id": area["area_id"],
                        "product_id": area["product_id"],
                        "product_name": area["product_name"],
                        "polygon": quad.tolist(),
                        "image_size": [img.shape[1], img.shape[0]],
                        "crop": rel.as_posix(),
                    }
                )
        typer.echo(f"camera {camera} done")

    if not items:
        typer.echo(f"0 crops made - no .jpg files found under {images_root}", err=True)
        raise typer.Exit(1)
    (out_dir / "index.json").write_text(json.dumps(items))
    typer.echo(f"{len(items)} crops -> {out_dir}")


if __name__ == "__main__":
    app()
