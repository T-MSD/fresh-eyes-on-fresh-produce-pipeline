"""Choose the next crops to label by hand (out/label_queue.json), which review_tool.py then shows.

  informative  crops where the trained stock model has the most to lose, i.e. the highest expected
               regret under the assignment's scoring, which mostly means doubtful "half" shelves.
               Spread out: one per area, few per camera (fewer for cameras already well labelled),
               and no two that look nearly identical.
  random       a few unremarkable crops chosen blind, to measure accuracy on a typical shelf.

Areas that run off the edge of the image are skipped, because labelling them needs a rule
that has not been agreed.

    uv run --env-file .env python -m src.pick_labels --informative 40 --random 10
"""

import json
from pathlib import Path

import numpy as np
import typer

from src.train_classifier import POINTS, features, fit_proba

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
LEVELS = ("empty", "half", "full")

app = typer.Typer(add_completion=False)


@app.command()
def main(
    informative: int = 40,
    random: int = 10,
    max_per_camera: int = 4,
    max_similarity: float = 0.95,
    seed: int = 0,
) -> None:
    items = json.loads((OUT / "crops" / "index.json").read_text())
    emb = np.load(OUT / "clip_embeddings.npz")["embeddings"]
    zs = np.load(OUT / "clip_level_probs.npy")
    manual = json.loads((OUT / "labels.json").read_text())

    x = features(emb, zs, False)
    rows = [n for n, i in enumerate(items) if i["key"] in manual]
    y = np.array([LEVELS.index(manual[items[n]["key"]]["stock_level"]) for n in rows])
    p = fit_proba(x[rows], y, x, 3, 0.03)
    regret = 1 - (p @ POINTS.T).max(axis=1)  # points lost, in expectation, by the best guess

    per_camera: dict[str, int] = {}
    for n in rows:
        per_camera[items[n]["camera"]] = per_camera.get(items[n]["camera"], 0) + 1

    def off_image(item: dict) -> bool:
        w, h = item["image_size"]
        return any(px < 0 or py < 0 or px > w or py > h for px, py in item["polygon"])

    candidates = [n for n, i in enumerate(items) if i["key"] not in manual and not off_image(i)]
    rng = np.random.default_rng(seed)

    chosen: list[int] = []
    areas: set[str] = set()
    cams: dict[str, int] = {}

    def allowed(n: int) -> bool:
        cam, area = items[n]["camera"], items[n]["area_id"]
        cap = 1 if per_camera.get(cam, 0) > 50 else max_per_camera
        if area in areas or cams.get(cam, 0) >= cap:
            return False
        return all(float(emb[n] @ emb[m]) < max_similarity for m in chosen)

    for n in sorted(candidates, key=lambda n: -regret[n]):
        if len(chosen) == informative:
            break
        if allowed(n):
            chosen.append(n)
            areas.add(items[n]["area_id"])
            cams[items[n]["camera"]] = cams.get(items[n]["camera"], 0) + 1
    informative_keys = [items[n]["key"] for n in chosen]

    # Random ones: different cameras, skipping the camera that is already well covered.
    taken = set(chosen)
    random_n: list[int] = []
    seen_cams: set[str] = set()
    for n in rng.permutation(candidates):
        i = items[n]
        if n in taken or i["area_id"] in areas or per_camera.get(i["camera"], 0) > 50:
            continue
        if i["camera"] in seen_cams:
            continue
        random_n.append(int(n))
        seen_cams.add(i["camera"])
        areas.add(i["area_id"])
        if len(random_n) == random:
            break
    random_keys = [items[n]["key"] for n in random_n]

    (OUT / "label_queue.json").write_text(
        json.dumps({"informative": informative_keys, "random": random_keys}, indent=1)
    )
    typer.echo(
        f"{len(informative_keys)} informative + {len(random_keys)} random -> out/label_queue.json"
    )
    pred = p.argmax(axis=1)
    typer.echo(
        "informative, model's best guess: "
        + ", ".join(f"{lv} {int((pred[chosen] == k).sum())}" for k, lv in enumerate(LEVELS))
    )
    typer.echo(
        f"informative, mean regret {regret[chosen].mean():.2f} vs {regret[candidates].mean():.2f} overall"
    )
    cams_all = sorted({items[n]["camera"] for n in chosen + random_n})
    typer.echo(f"cameras covered: {len(cams_all)}")


if __name__ == "__main__":
    app()
