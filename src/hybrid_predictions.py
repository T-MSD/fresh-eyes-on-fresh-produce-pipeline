"""Current submission + only the purchase-rule changes the user judged right (out/final_predictions.hybrid.json).

From the 100 judged disagreements (src/compare_strategies.py): the purchase-rule file was right on
fill-product half->full (18 of 22, all judged), fill empty->full (1 of 1) and counted empty->half
(4 of 5, all judged). It was wrong on fill empty->half, fill full->half and counted half->empty, so
those keep the current level. Planogram compliance is copied unchanged.

    python src/hybrid_predictions.py
"""

import copy
import json
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
VALUE = {"empty": 0, "half": 0.5, "full": 1}
TAKE = {
    ("fill", "half", "full"),
    ("fill", "empty", "full"),
    ("count", "empty", "half"),
}  # (method, current, new)

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    items = json.loads((OUT / "crops" / "index.json").read_text())
    groups = json.loads((OUT / "product_groups.json").read_text())
    cur = json.loads((OUT / "final_predictions.before_hybrid.json").read_text())
    new = json.loads((OUT / "purchase_rule_predictions.json").read_text())
    out = copy.deepcopy(cur)
    taken = 0
    for it in items:
        c, i, a = it["camera"], it["image"], it["area_id"]
        before, after = cur[c][i][a]["stock_level"], new[c][i][a]["stock_level"]
        if (groups[it["product_id"]]["method"], before, after) in TAKE:
            out[c][i][a]["stock_level"], out[c][i][a]["stock_value"] = after, VALUE[after]
            taken += 1
    (OUT / "final_predictions.hybrid.json").write_text(json.dumps(out, indent=1))
    typer.echo(f"{taken} areas changed vs current -> out/final_predictions.hybrid.json")


if __name__ == "__main__":
    app()
