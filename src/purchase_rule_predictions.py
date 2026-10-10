"""Stock levels under the purchase rule -> out/purchase_rule_predictions.json (submission format).

Per product (out/product_groups.json):
  fill  : ridge on the CLIP embedding, trained on the hand-labelled fill percentages (out/fill_labels.json),
          then empty 0-15 / half 16-64 / full 65-100.
  count : OWL-ViT count of loose items (out/owl_counts.json). Count <= the group's "empty" number -> empty.
          Only if the current submission does not already say "full": a full tray of large items shows few
          units, and OWL misses fruit, so a low count must never overrule "full" (it did on 181 crops).
          Otherwise the current submission's level, except that its "empty" becomes "half" (the count says
          a purchase would not empty the tray). Products OWL has no noun for keep the current level.
planogram_compliant is copied unchanged from out/final_predictions.before_hybrid.json. Nothing existing is overwritten.

    python src/purchase_rule_predictions.py
"""

import json
from pathlib import Path

import numpy as np
import typer
from sklearn.linear_model import Ridge

from train_classifier import standardise

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
VALUE = {"empty": 0, "half": 0.5, "full": 1}
LEVELS = ("empty", "half", "full")

app = typer.Typer(add_completion=False)


def level_of_pct(p: float) -> str:
    return "empty" if p <= 15 else "half" if p <= 64 else "full"


@app.command()
def main(alpha: float = 300.0) -> None:
    items = json.loads((OUT / "crops" / "index.json").read_text())
    groups = json.loads((OUT / "product_groups.json").read_text())
    base = json.loads((OUT / "final_predictions.before_hybrid.json").read_text())
    counts = json.loads((OUT / "owl_counts.json").read_text())
    fill = {
        k: v["fill"]
        for k, v in json.loads((OUT / "fill_labels.json").read_text()).items()
        if not v.get("wrong")
    }
    z = np.load(OUT / "clip_embeddings.npz", allow_pickle=True)
    row = {k: n for n, k in enumerate(z["keys"])}
    x = standardise(z["embeddings"])
    train = [k for k in fill if k in row]
    reg = Ridge(alpha=alpha).fit(
        x[[row[k] for k in train]], np.array([fill[k] for k in train], dtype=float)
    )
    typer.echo(f"fill model trained on {len(train)} percentage labels")

    result: dict = {}
    stats = {"fill": [0, 0], "count": [0, 0], "count_no_owl": 0}  # [crops, changed vs current]
    for it in items:
        cam, img, area = it["camera"], it["image"], it["area_id"]
        cur = base[cam][img][area]
        g = groups[it["product_id"]]
        level = cur["stock_level"]
        if g["method"] == "fill":
            pct = float(np.clip(reg.predict(x[[row[it["key"]]]])[0], 0, 100))
            level = level_of_pct(pct)
            stats["fill"][0] += 1
            stats["fill"][1] += level != cur["stock_level"]
        elif it["key"] in counts:
            if counts[it["key"]] <= g["empty_max_units"] and level != "full":
                level = "empty"
            elif level == "empty":
                level = "half"
            stats["count"][0] += 1
            stats["count"][1] += level != cur["stock_level"]
        else:
            stats["count_no_owl"] += 1
        result.setdefault(cam, {}).setdefault(img, {})[area] = {
            "planogram_compliant": cur["planogram_compliant"],
            "stock_level": level,
            "stock_value": VALUE[level],
        }
    (OUT / "purchase_rule_predictions.json").write_text(json.dumps(result, indent=1))
    typer.echo(f"fill crops {stats['fill'][0]}, changed vs current {stats['fill'][1]}")
    typer.echo(
        f"count crops {stats['count'][0]}, changed vs current {stats['count'][1]}; no OWL noun, kept current: {stats['count_no_owl']}"
    )
    for name, src in (("current", base), ("purchase rule", result)):
        c = {lv: 0 for lv in LEVELS}
        for cam in src.values():
            for im in cam.values():
                for a in im.values():
                    c[a["stock_level"]] += 1
        typer.echo(f"{name:14s} {c}")


if __name__ == "__main__":
    app()
