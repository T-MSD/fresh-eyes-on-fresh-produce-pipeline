"""Queue the unlabelled crops most likely to be half-stocked for hand labelling (out/label_queue.json).

Half is the level the model confuses most and has fewest labels for. This ranks the unlabelled crops
by the fine-tuned network's probability of "half" (out/finetune/stock_probs.npy), spread out as in
pick_labels.py: one per area, few per camera (1 for cameras already well labelled), no near-twins,
and none of the areas that run off the image edge. The queue is marked blind, so the app hides the
model's suggestion and the labels are your own judgement.

    uv run --env-file .env python -m experiments.pick_half --n 40
"""

import json
from pathlib import Path

import numpy as np
import typer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

app = typer.Typer(add_completion=False)


@app.command()
def main(n: int = 40, max_per_camera: int = 4, max_similarity: float = 0.95) -> None:
    items = json.loads((OUT / "crops" / "index.json").read_text())
    emb = np.load(OUT / "clip_embeddings.npz")["embeddings"]
    p_half = np.load(OUT / "finetune" / "stock_probs.npy")[:, 1]
    manual = json.loads((OUT / "labels.json").read_text())

    per_camera: dict[str, int] = {}
    for i in items:
        if i["key"] in manual:
            per_camera[i["camera"]] = per_camera.get(i["camera"], 0) + 1

    def off_image(item: dict) -> bool:
        w, h = item["image_size"]
        return any(x < 0 or y < 0 or x > w or y > h for x, y in item["polygon"])

    # Skip crops that look almost identical to one already labelled: they teach nothing new.
    labelled = [k for k, i in enumerate(items) if i["key"] in manual]
    near_labelled = (emb @ emb[labelled].T).max(axis=1) >= max_similarity
    candidates = [
        k
        for k, i in enumerate(items)
        if i["key"] not in manual and not off_image(i) and not near_labelled[k]
    ]
    chosen: list[int] = []
    areas: set[str] = set()
    cams: dict[str, int] = {}
    for k in sorted(candidates, key=lambda k: -p_half[k]):
        cam, area = items[k]["camera"], items[k]["area_id"]
        cap = 1 if per_camera.get(cam, 0) > 50 else max_per_camera
        if area in areas or cams.get(cam, 0) >= cap:
            continue
        if any(float(emb[k] @ emb[m]) >= max_similarity for m in chosen):
            continue
        chosen.append(k)
        areas.add(area)
        cams[cam] = cams.get(cam, 0) + 1
        if len(chosen) == n:
            break

    keys = [items[k]["key"] for k in chosen]
    (OUT / "label_queue.json").write_text(json.dumps({"informative": [], "blind": keys}, indent=1))
    probs = np.load(OUT / "finetune" / "stock_probs.npy")
    guess = np.bincount(probs[chosen].argmax(1), minlength=3).tolist()
    typer.echo(f"{len(keys)} crops -> out/label_queue.json (all blind)")
    typer.echo(
        f"p(half) {p_half[chosen].min():.2f}-{p_half[chosen].max():.2f} (median over unlabelled {np.median(p_half[candidates]):.2f})"
    )
    typer.echo(
        f"network's own guess for them, empty/half/full: {guess}; cameras: {dict(sorted(cams.items()))}"
    )


if __name__ == "__main__":
    app()
