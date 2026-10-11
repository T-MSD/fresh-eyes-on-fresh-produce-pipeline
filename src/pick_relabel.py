"""List the already-labelled crops that could change under the new cutoffs (out/relabel_queue.json).

Old rule: empty <25%, half 25-75%, full >75%.  New rule: empty 0-34%, half 35-79%, full 80-100%.
Only two bands move: 25-34% (half -> empty) and 76-79% (full -> half). Fill percentages were never
recorded, so the crops in those bands have to be found by looking again. The queue holds every
"half" label (the first band is among them), every "unsure" label, and the "full" labels that
CLIP finds least full (a weak hint for the second band). No model is trained.

    uv run --env-file .env python -m src.pick_relabel [--full 40]
"""

import json
from pathlib import Path

import numpy as np
import typer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"

app = typer.Typer(add_completion=False)


@app.command()
def main(full: int = 40) -> None:
    items = json.loads((OUT / "crops" / "index.json").read_text())
    probs = np.load(OUT / "clip_level_probs.npy")
    labels = json.loads((OUT / "labels.json").read_text())
    n_of = {i["key"]: n for n, i in enumerate(items)}

    half = [k for k, v in labels.items() if v["stock_level"] == "half"]
    unsure = [k for k, v in labels.items() if v.get("unsure") and v["stock_level"] != "half"]
    fulls = sorted(
        (k for k, v in labels.items() if v["stock_level"] == "full" and k not in unsure),
        key=lambda k: probs[n_of[k], 2],
    )[:full]
    queue = half + unsure + fulls
    (OUT / "relabel_queue.json").write_text(json.dumps(queue, indent=1))
    typer.echo(
        f"{len(queue)} crops: {len(half)} half, {len(unsure)} other unsure, {len(fulls)} least-full 'full'"
    )


if __name__ == "__main__":
    app()
