"""Does the fill model separate empty/half/full better under the old or the new cutoffs?

Fits on the hand-labelled fill percentages (out/fill_labels.json) and the saved CLIP embeddings,
holding out whole cameras. For each cutoff rule it compares (a) ridge regression on the percentage,
thresholded afterwards, and (b) a classifier trained directly on that rule's levels. Scores with the
assignment's points (POINTS[predicted, true]). Splits are repeated with different camera groupings;
the spread shows how noisy 150 labels are. It says which rule is more learnable, not which one the
hidden annotators used.

    python src/fill_probe.py
"""

import json
from pathlib import Path

import numpy as np
import typer
from sklearn.linear_model import LogisticRegression, Ridge

from src.clip_predict import POINTS
from src.train_classifier import standardise

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
RULES = {
    "old (<25 / 25-75 / >75)": lambda p: np.where(p < 25, 0, np.where(p <= 75, 1, 2)),
    "new (0-34 / 35-79 / 80-100)": lambda p: np.where(p <= 34, 0, np.where(p <= 79, 1, 2)),
    "newer (0-15 / 16-64 / 65-100)": lambda p: np.where(p <= 15, 0, np.where(p <= 64, 1, 2)),
}

# Assumed units a full tray holds, per group. NOT measured: used only to turn "empty = N units or fewer"
# into a fill percentage. The test is repeated with this scaled x0.5 and x2.
CAPACITY = {"G_bag": 10, "E_small_pack": 12, "F_bunch": 8, "C_medium_fruit": 40, "D_medium_veg": 30}

app = typer.Typer(add_completion=False)


def empty_pct(group: str, empty_max: int, scale: float) -> float:
    """Fill percentage at or below which a tray counts as empty: the global 15, or N units, if higher."""
    return max(15.0, 100.0 * empty_max / (CAPACITY[group] * scale))


def score(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    pts = POINTS[pred, true].mean()
    exact = (pred == true).mean()
    half = true == 1
    return pts, exact, (pred[half] == 1).mean() if half.any() else np.nan


@app.command()
def main(repeats: int = 30, alpha: float = 300.0, c: float = 0.05) -> None:
    groups = json.loads((OUT / "product_groups.json").read_text())
    z = np.load(OUT / "clip_embeddings.npz", allow_pickle=True)
    row = {k: n for n, k in enumerate(z["keys"])}
    items = {i["key"]: i for i in json.loads((OUT / "crops" / "index.json").read_text())}
    fill = {
        k: v
        for k, v in json.loads((OUT / "fill_labels.json").read_text()).items()
        if not v.get("wrong")
    }
    keys = sorted(fill)
    x = standardise(z["embeddings"][[row[k] for k in keys]])
    pct = np.array([fill[k]["fill"] for k in keys], dtype=float)
    cams = np.array([items[k]["camera"] for k in keys])
    uniq = np.unique(cams)
    typer.echo(
        f"{len(keys)} labels, {len(uniq)} cameras, held out by camera, {repeats} random splits"
    )
    np.full(len(keys), 15.0)
    t_unit = {
        sc: np.array(
            [
                empty_pct(
                    groups[items[k]["product_id"]]["group"],
                    groups[items[k]["product_id"]]["empty_max_units"],
                    sc,
                )
                for k in keys
            ]
        )
        for sc in (0.5, 1.0, 2.0)
    }
    rules = {n: (r, None) for n, r in RULES.items()}
    for sc, t in t_unit.items():
        rules[f"0-15 + group empty exception (capacity x{sc})"] = (
            lambda p, t=t: np.where(p <= t, 0, np.where(p <= 64, 1, 2)),
            t,
        )
    rng = np.random.default_rng(0)
    # The same camera splits for every rule, so differences are not split luck.
    splits = [rng.choice(uniq, size=max(2, len(uniq) // 4), replace=False) for _ in range(repeats)]
    for name, (rule, t) in rules.items():
        y = rule(pct)
        if t is not None:
            typer.echo(f"\n(empty threshold by crop: {sorted({round(v) for v in t})}%)", nl=False)
        typer.echo(
            f"\n{name}: class counts empty/half/full = {np.bincount(y, minlength=3).tolist()}"
        )
        res = {"regress": [], "classify": []}
        for test_cams in splits:
            te = np.isin(cams, test_cams)
            if te.sum() < 10 or len(set(y[~te])) < 3:
                continue
            reg = Ridge(alpha=alpha).fit(x[~te], pct[~te])
            res["regress"].append(
                score(
                    y[te],
                    rule(reg.predict(x[te]))
                    if t is None
                    else np.where(
                        reg.predict(x[te]) <= t[te], 0, np.where(reg.predict(x[te]) <= 64, 1, 2)
                    ),
                )
            )
            clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced").fit(
                x[~te], y[~te]
            )
            res["classify"].append(score(y[te], clf.predict(x[te])))
        for method, r in res.items():
            a = np.array(r)
            typer.echo(
                f"  {method:9s} points {a[:, 0].mean():+.2f} ±{a[:, 0].std():.2f}   exact {a[:, 1].mean():.0%} ±{a[:, 1].std():.0%}"
                f"   halves found {np.nanmean(a[:, 2]):.0%} ±{np.nanstd(a[:, 2]):.0%}"
            )
        base = max(np.bincount(y)) / len(y)
        typer.echo(f"  always-predict-majority exact: {base:.0%}")


if __name__ == "__main__":
    app()
