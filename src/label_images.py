"""Run Google Cloud Vision label detection on the first N cropped planogram areas.

Writes out/vision_labels.json ({crop path <area_id>/<ts>.jpg: [{description, score}, ...]}). Resumable: images
already in the file are skipped, and results are saved every batch.

    python src/label_images.py --limit 1000
"""

import json
import time
from pathlib import Path

import typer
from google.api_core.exceptions import ResourceExhausted
from google.cloud import vision

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "out" / "crops"
OUT = ROOT / "out" / "vision_labels.json"
BATCH = 16  # Vision allows up to 16 images per batch request

app = typer.Typer(add_completion=False)


@app.command()
def main(
    data: Path = DEFAULT_DATA,
    limit: int = 1000,
    min_score: float = 0.0,
) -> None:
    images = sorted(data.rglob("*.jpg"))[:limit]
    results: dict[str, list[dict]] = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [p for p in images if str(p.relative_to(data)) not in results]
    typer.echo(f"{len(images)} images selected, {len(todo)} still to label")

    client = vision.ImageAnnotatorClient()
    feature = vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION)

    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        requests = [
            vision.AnnotateImageRequest(
                image=vision.Image(content=p.read_bytes()), features=[feature]
            )
            for p in chunk
        ]
        for attempt in range(6):
            try:
                response = client.batch_annotate_images(requests=requests)
                break
            except ResourceExhausted:
                wait = 15 * 2**attempt
                typer.echo(f"  rate limited, waiting {wait}s", err=True)
                time.sleep(wait)
        else:
            raise typer.Exit(1)
        for path, res in zip(chunk, response.responses, strict=False):
            if res.error.message:
                typer.echo(f"  {path}: {res.error.message}", err=True)
                continue
            results[str(path.relative_to(data))] = [
                {"description": label.description, "score": round(label.score, 4)}
                for label in res.label_annotations
                if label.score >= min_score
            ]
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(results, indent=1))
        typer.echo(f"  {len(results)}/{len(images)}")


if __name__ == "__main__":
    app()
