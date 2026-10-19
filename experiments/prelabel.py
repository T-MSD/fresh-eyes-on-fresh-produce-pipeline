"""Turn Vision labels into predictions.

Writes two files:
  out/predictions.json  every area, in the assignment's annotations.json format
                        {camera: {image: {area_id: {planogram_compliant, stock_level, stock_value}}}}
  out/prelabels.json    only the "empty" suggestions, shown pre-filled in the review tool

Vision says what is in a crop, not how full the shelf is, so this is a baseline: "empty" when
no produce-like label is found, otherwise "half" (the cheapest hedge under the scoring), and
always compliant (about 93% of validation areas are).

    uv run --env-file .env python -m experiments.prelabel
"""

import json
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
STOCK_VALUES = {"empty": 0, "half": 0.5, "full": 1}
PRODUCE = {"food", "produce", "fruit", "vegetable", "natural foods", "ingredient", "food group"}

app = typer.Typer(add_completion=False)


@app.command()
def main(min_score: float = 0.6) -> None:
    vision = json.loads((OUT / "vision_labels.json").read_text())
    items = json.loads((OUT / "crops" / "index.json").read_text())
    pre: dict = {}
    predictions: dict = {}
    for item in items:
        labels = vision.get(item["crop"])
        if labels is None:
            continue
        has_produce = any(
            lab["description"].lower() in PRODUCE and lab["score"] >= min_score for lab in labels
        )
        level = "half" if has_produce else "empty"
        predictions.setdefault(item["camera"], {}).setdefault(item["image"], {})[
            item["area_id"]
        ] = {
            "planogram_compliant": True,
            "stock_level": level,
            "stock_value": STOCK_VALUES[level],
        }
        if not has_produce:
            pre[item["key"]] = {
                "planogram_compliant": True,
                "stock_level": "empty",
                "unsure": True,
            }
    (OUT / "prelabels.json").write_text(json.dumps(pre, indent=1))
    (OUT / "predictions.json").write_text(json.dumps(predictions, indent=1))
    n = sum(len(a) for c in predictions.values() for a in c.values())
    typer.echo(
        f"{n} areas -> out/predictions.json, {len(pre)} 'empty' suggestions -> out/prelabels.json"
    )


if __name__ == "__main__":
    app()
