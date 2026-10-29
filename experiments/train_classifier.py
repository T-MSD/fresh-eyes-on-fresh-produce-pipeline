"""Learn from the manual labels: logistic regression on the saved CLIP image embeddings.

Compares it with the zero-shot baseline using the assignment's scoring, on cameras the model
was not trained on (so near-identical frames of the same shelf cannot leak into the test).
Only if it scores higher does it overwrite out/predictions.json and out/prelabels.json.

    uv run --extra ml --env-file .env python -m experiments.train_classifier
    uv run --extra ml --env-file .env python -m experiments.train_classifier --force
"""

import json
from pathlib import Path

import numpy as np
import typer
from sklearn.model_selection import GroupKFold

from src.classifier import features, fit_proba
from src.clip_predict import LEVELS, POINTS, STOCK_VALUES
from src.prediction_io import merge_fallbacks

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
MIN_LABELS = 100
MIN_MINORITY = 10  # fewer examples of a class than this -> keep the zero-shot answer

app = typer.Typer(add_completion=False)


def pick_levels(p_stock: np.ndarray) -> np.ndarray:
    """Level with the best expected points: a doubtful shelf leans low (-2 below, -3 above)."""
    return np.argmax(p_stock @ POINTS.T, axis=1)


def score(level_idx, compliant, true_level_idx, true_compliant) -> float:
    """Average points per area: identification +1/-2, stock from POINTS."""
    ident = np.where(np.asarray(compliant) == np.asarray(true_compliant), 1, -2)
    stock = POINTS[level_idx, true_level_idx]
    return float((ident + stock).mean())


@app.command()
def main(force: bool = False) -> None:
    items = json.loads((OUT / "crops" / "index.json").read_text())
    cache = np.load(OUT / "clip_embeddings.npz")
    assert list(cache["keys"]) == [i["key"] for i in items], "rerun python -m src.clip_predict"
    emb_all = cache["embeddings"]
    zs_probs = np.load(OUT / "clip_level_probs.npy")
    zero = json.loads((OUT / "clip_zeroshot.json").read_text())
    manual = json.loads((OUT / "labels.json").read_text())

    rows = [n for n, i in enumerate(items) if i["key"] in manual]
    typer.echo(f"{len(rows)} manual labels")
    if len(rows) < MIN_LABELS:
        typer.echo(f"Need at least {MIN_LABELS} labels before training is worth trusting.")
        raise typer.Exit(1)

    keys = [items[n]["key"] for n in rows]
    groups = np.array([items[n]["camera"] for n in rows])
    y_level = np.array([LEVELS.index(manual[k]["stock_level"]) for k in keys])
    y_ok = np.array([int(manual[k]["planogram_compliant"]) for k in keys])
    z_level = np.array([LEVELS.index(zero[k]["stock_level"]) for k in keys])
    z_ok = np.array([int(zero[k]["planogram_compliant"]) for k in keys])

    n_cams = len(set(groups))
    typer.echo(
        f"cameras: {n_cams}; stock {np.bincount(y_level, minlength=3).tolist()} "
        f"(empty/half/full); wrong product {int((y_ok == 0).sum())}"
    )
    if n_cams < 3:
        typer.echo("Labels cover fewer than 3 cameras - label more cameras first.")
        raise typer.Exit(1)

    folds = list(GroupKFold(n_splits=min(5, n_cams)).split(y_level, y_level, groups))
    use_ok_model = min(int((y_ok == 0).sum()), int((y_ok == 1).sum())) >= MIN_MINORITY
    if not use_ok_model:
        typer.echo("Too few wrong-product labels to learn compliance; keeping zero-shot for it.")

    typer.echo("\nheld-out cameras, points per area (stock max 1.0, identification max 1.0):")
    typer.echo(
        f"  always half / always compliant: stock {POINTS[1, y_level].mean():+.2f}, "
        f"identification {np.where(y_ok == 1, 1, -2).mean():+.2f}"
    )

    # Stock level and compliance are judged separately: a gain on one must not hide a loss on
    # the other. Each keeps the zero-shot answer unless a trained model beats it by itself.
    stock_zs = float(POINTS[z_level, y_level].mean())
    ident_zs = float(np.where(z_ok == y_ok, 1, -2).mean())
    best_stock = (-9.0, None, None)  # (points, C, with_zeroshot_features)
    best_ok = (ident_zs, None, None)  # None = keep zero-shot
    for with_zs in (False, True):
        x = features(emb_all, zs_probs, with_zs)[rows]
        name = "emb+zeroshot" if with_zs else "emb only    "
        for c in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
            p_level = np.zeros((len(rows), 3))
            p_ok = np.zeros((len(rows), 2))
            for tr, te in folds:
                p_level[te] = fit_proba(x[tr], y_level[tr], x[te], 3, c)
                if use_ok_model:
                    p_ok[te] = fit_proba(x[tr], y_ok[tr], x[te], 2, c)
            stock = float(POINTS[pick_levels(p_level), y_level].mean())
            ident = (
                float(np.where(p_ok.argmax(axis=1) == y_ok, 1, -2).mean()) if use_ok_model else None
            )
            typer.echo(
                f"  trained {name} C={c:<7} stock {stock:+.2f}"
                + (f"  identification {ident:+.2f}" if use_ok_model else "")
            )
            if stock > best_stock[0]:
                best_stock = (stock, c, with_zs)
            if use_ok_model and ident > best_ok[0]:
                best_ok = (ident, c, with_zs)

    typer.echo(f"\nstock level     zero-shot {stock_zs:+.2f}  best trained {best_stock[0]:+.2f}")
    typer.echo(
        f"identification zero-shot {ident_zs:+.2f}  best trained "
        + (f"{best_ok[0]:+.2f}" if best_ok[1] is not None else "none beats zero-shot")
    )
    typer.echo(
        "(settings are picked on the same folds, so the trained numbers are slightly optimistic)"
    )
    use_trained_stock = best_stock[0] > stock_zs
    use_trained_ok = best_ok[1] is not None
    if not (use_trained_stock or use_trained_ok) and not force:
        typer.echo("Zero-shot is still as good or better; predictions left unchanged.")
        return

    if use_trained_stock:
        c, with_zs = best_stock[1:]
        x_all = features(emb_all, zs_probs, with_zs)
        p_level = fit_proba(x_all[rows], y_level, x_all, 3, c)
        level_idx = pick_levels(p_level)
    else:
        p_level = np.where(
            np.eye(3)[[LEVELS.index(zero[i["key"]]["stock_level"]) for i in items]] > 0, 1.0, 0.0
        )
        level_idx = p_level.argmax(axis=1)
    if use_trained_ok:
        c, with_zs = best_ok[1:]
        x_all = features(emb_all, zs_probs, with_zs)
        p_ok = fit_proba(x_all[rows], y_ok, x_all, 2, c)[:, 1]
    typer.echo(f"using trained stock: {use_trained_stock}; trained compliance: {use_trained_ok}")

    predictions: dict = {}
    pre: dict = {}
    for n, item in enumerate(items):
        level = LEVELS[level_idx[n]]
        ok = bool(p_ok[n] >= 0.5) if use_trained_ok else zero[item["key"]]["planogram_compliant"]
        predictions.setdefault(item["camera"], {}).setdefault(item["image"], {})[
            item["area_id"]
        ] = {"planogram_compliant": ok, "stock_level": level, "stock_value": STOCK_VALUES[level]}
        pre[item["key"]] = {
            "planogram_compliant": ok,
            "stock_level": level,
            "unsure": bool(p_level[n].max() < 0.6),
        }
    merge_fallbacks(predictions, OUT / "crops")
    (OUT / "predictions.json").write_text(json.dumps(predictions, indent=1))
    (OUT / "prelabels.json").write_text(json.dumps(pre, indent=1))
    typer.echo(f"wrote out/predictions.json and out/prelabels.json for {len(items)} areas")


if __name__ == "__main__":
    app()
