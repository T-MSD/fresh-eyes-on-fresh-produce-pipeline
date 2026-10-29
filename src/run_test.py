"""Predictions for NEW camera images, reproducing the submitted file without retraining anything.

  1. crop every planogram area of the new data           (src/crop_areas.py, into out/test/crops)
  2. CLIP: embeddings, zero-shot stock probabilities, compliance flag (same prompts as clip_predict.py)
  3. fine-tuned network (out/finetune/model.pt, no training)  -> stock probabilities
  4. CLIP + logistic regression fitted on the 404 hand labels (from the bundle) -> stock probabilities
  5. stock = 0.75 network + 0.25 classifier, level picked by expected points (leans low when unsure)
  6. one extra rule: for percentage-method products, a blend "empty/half" becomes "full" if the fill
     model (ridge on CLIP, from the bundle) says >= 65%. Judged on 22+1 crops: 19 right, 4 wrong.
     Skipped for products missing from out/product_groups.json.
planogram_compliant is the zero-shot CLIP flag. A product no class rule knows is scored by the
classifier alone and counted compliant. Nothing under out/ except out/test/ is written.

Reads only out/finetune/model.pt, out/finetune/bundle.joblib (both from finetune --final) and
out/product_groups.json. --selftest also reads the training crops it rebuilds the submission from.

    uv run --extra ml --env-file .env python -m src.run_test --data-dir <dataset-folder>
    uv run --extra ml --env-file .env python -m src.run_test --selftest
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from config.db.connection import make_engine
from config.db.save_run import UnusableRun, fallback_keys, latest_dataset_id, save_run
from src.blend import FILL_FULL_PCT, W_NET
from src.clip_predict import LEVELS, POINTS, STOCK_VALUES, product_class, softmax
from src.prediction_io import NetworkCrop, merge_fallbacks

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
TEST = OUT / "test"
BACKBONE, SIZE = "resnet50.a1_in1k", 224
BUNDLE = OUT / "finetune" / "bundle.joblib"
UNKNOWN_CLASS = "fresh produce"

app = typer.Typer(add_completion=False)


def class_of(product_name: str) -> str:
    """The product's class, or a generic one for a product no class rule knows."""
    try:
        return product_class(product_name)
    except ValueError:
        return UNKNOWN_CLASS


def standardise_with(a: np.ndarray, bundle: dict) -> np.ndarray:
    """Column-standardise `a` with the TRAINING set's mean/std stored in the bundle."""
    return (a - bundle["emb_mean"]) / (bundle["emb_std"] + 1e-6)


def clip_zeroshot(items, crops_dir: Path, classes: list[str], topk: int, cached: np.ndarray | None):
    """Image embeddings, level probabilities and the compliance flag, as src/clip_predict.py computes them."""
    import open_clip
    import torch
    from PIL import Image

    name, pretrained = "ViT-B-16-quickgelu", "openai"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.to(device).eval()
    tok = open_clip.get_tokenizer(name)
    scale = float(model.logit_scale.exp().item())

    def text(texts):
        with torch.no_grad():
            return (
                torch.nn.functional.normalize(model.encode_text(tok(texts).to(device)), dim=-1)
                .cpu()
                .numpy()
            )

    if cached is not None:
        img = cached
    else:
        chunks = []
        for i in range(0, len(items), 32):
            batch = torch.stack(
                [
                    preprocess(Image.open(crops_dir / it["crop"]).convert("RGB"))
                    for it in items[i : i + 32]
                ]
            ).to(device)
            with torch.no_grad():
                chunks.append(
                    torch.nn.functional.normalize(model.encode_image(batch), dim=-1).cpu().numpy()
                )
            typer.echo(f"  CLIP {min(i + 32, len(items))}/{len(items)}")
        img = np.concatenate(chunks)
    class_txt = text([f"a photo of {c} on a supermarket shelf" for c in classes])
    class_prob = softmax(scale * img @ class_txt.T)
    idx = {c: n for n, c in enumerate(classes)}
    level_txt = {
        c: text(
            [
                f"an empty supermarket shelf with no {c}",
                f"a supermarket shelf with only a few {c}",
                f"a supermarket shelf full of piled {c}",
            ]
        )
        for c in [*classes, UNKNOWN_CLASS]
    }
    zs, ok = np.zeros((len(items), 3)), []
    for n, it in enumerate(items):
        c = class_of(it["product_name"])
        zs[n] = softmax(scale * img[n] @ level_txt[c].T)
        level = LEVELS[int(np.argmax(zs[n] @ POINTS.T))]
        ok.append(
            bool(
                level == "empty"
                or c == UNKNOWN_CLASS
                or idx[c] in np.argsort(class_prob[n])[::-1][:topk]
            )
        )
    return img, zs, np.array(ok)


def blend_network(
    items: list[NetworkCrop], crops_dir: Path, train_classes: list[str], p_lr: np.ndarray
) -> np.ndarray:
    known = [n for n, item in enumerate(items) if class_of(item.product_name) in train_classes]
    if not known:
        return p_lr.copy()

    import torch
    from PIL import Image

    from src.finetune import Net, predict

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Net(BACKBONE, len(train_classes), pretrained=False)
    model.load_state_dict(
        torch.load(OUT / "finetune" / "model.pt", map_location="cpu", weights_only=True)
    )
    model.to(device)
    imgs = torch.from_numpy(
        np.stack(
            [
                np.asarray(
                    Image.open(crops_dir / items[n].crop).convert("RGB").resize((SIZE, SIZE))
                )
                for n in known
            ]
        ).transpose(0, 3, 1, 2)
    )
    cls = torch.tensor([train_classes.index(class_of(items[n].product_name)) for n in known])
    p_known, _ = predict(model, imgs, cls, device)
    probabilities = p_lr.copy()
    probabilities[known] = W_NET * p_known + (1 - W_NET) * p_lr[known]
    return probabilities


def write_predictions(result: dict, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=1))


def store_run(
    result: dict, crops_dir: Path, dataset_id: int | None, data_dir: Path | None, topk: int
) -> None:
    fallbacks_file = crops_dir / "fallbacks.json"
    fallback = fallback_keys(fallbacks_file) if fallbacks_file.exists() else set()
    engine = make_engine()
    try:
        with engine.begin() as conn:
            target = dataset_id if dataset_id is not None else latest_dataset_id(conn)
            run_id = save_run(
                conn,
                target,
                result,
                method="run_test",
                params={"data_dir": str(data_dir), "topk": topk, "w_net": W_NET},
                fallback=fallback,
            )
    except UnusableRun as error:
        for problem in error.problems:
            typer.echo(f"error      {problem}", err=True)
        typer.echo("predictions are in the output file; nothing was stored", err=True)
        raise typer.Exit(1) from error
    finally:
        engine.dispose()
    typer.echo(f"stored run {run_id} on dataset {target} ({len(fallback)} fallback areas)")


@app.command()
def main(
    data_dir: Path | None = typer.Option(
        None, help="New data: folder with planogram.json and camera folders"
    ),
    selftest: bool = typer.Option(
        False, help="Rebuild the current submission from its own crops and compare"
    ),
    out_file: Path = TEST / "predictions.json",
    crops_dir: Path = TEST / "crops",
    topk: int = 5,
    db: Annotated[
        bool, typer.Option(help="Also store the run and its predictions in the database")
    ] = False,
    dataset_id: Annotated[
        int | None,
        typer.Option(help="Dataset load the run belongs to; default is the newest (cli.py load)"),
    ] = None,
) -> None:
    if not selftest and data_dir is None:
        raise typer.BadParameter("give --data-dir, or --selftest")
    if db and selftest:
        raise typer.BadParameter("--db stores a run on new data; the self-test is not one")
    if selftest:
        items, crops_dir = json.loads((OUT / "crops" / "index.json").read_text()), OUT / "crops"
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.crop_areas",
                "--data-dir",
                str(data_dir),
                "--out-dir",
                str(crops_dir),
            ],
            check=True,
            cwd=ROOT,
        )
        items = json.loads((crops_dir / "index.json").read_text())
    typer.echo(f"{len(items)} areas to predict")
    if not items:
        result: dict = {}
        merge_fallbacks(result, crops_dir)
        write_predictions(result, out_file)
        typer.echo(f"wrote fallbacks to {out_file}")
        if db:
            store_run(result, crops_dir, dataset_id, data_dir, topk)
        return

    import joblib

    from src.classifier import proba_of

    bundle = joblib.load(BUNDLE)

    train_classes = bundle["train_classes"]
    new_classes = sorted({class_of(i["product_name"]) for i in items} - {UNKNOWN_CLASS})
    classes = sorted(set(train_classes) | set(new_classes))
    unseen = sorted(set(new_classes) - set(train_classes))
    if unseen:
        typer.echo(
            f"WARNING: product classes not seen in training: {unseen}. The network cannot score them; the classifier alone is used there."
        )
    no_rule = sorted(
        {i["product_name"] for i in items if class_of(i["product_name"]) == UNKNOWN_CLASS}
    )
    if no_rule:
        typer.echo(
            f"WARNING: no class rule for {no_rule}. They are scored as '{UNKNOWN_CLASS}' by the classifier alone and counted compliant."
        )

    # CLIP (cached embeddings only in the self-test, where the crops are the training crops).
    cached = np.load(OUT / "clip_embeddings.npz")["embeddings"] if selftest else None
    emb, _, compliant = clip_zeroshot(items, crops_dir, classes, topk, cached)

    x_new = standardise_with(emb, bundle)
    p_lr = proba_of(bundle["classifier"], x_new, len(LEVELS))

    p = blend_network(
        [NetworkCrop.model_validate(item) for item in items], crops_dir, train_classes, p_lr
    )
    level = (p @ POINTS.T).argmax(1)

    # The one extra rule, for percentage-method products.
    groups = json.loads((OUT / "product_groups.json").read_text())
    pct = np.clip(bundle["fill_model"].predict(x_new), 0, 100)
    raised = 0
    for n, it in enumerate(items):
        g = groups.get(it["product_id"])
        if g and g["method"] == "fill" and level[n] < 2 and pct[n] >= FILL_FULL_PCT:
            level[n] = 2
            raised += 1
    typer.echo(f"fill rule raised {raised} areas to full")

    result = {}
    for n, it in enumerate(items):
        lv = LEVELS[level[n]]
        result.setdefault(it["camera"], {}).setdefault(it["image"], {})[it["area_id"]] = {
            "planogram_compliant": bool(compliant[n]),
            "stock_level": lv,
            "stock_value": STOCK_VALUES[lv],
        }
    merge_fallbacks(result, crops_dir)
    write_predictions(result, out_file)
    if db:
        store_run(result, crops_dir, dataset_id, data_dir, topk)
    counts = {lv: int((level == k).sum()) for k, lv in enumerate(LEVELS)}
    typer.echo(f"wrote {out_file}: {counts}, non-compliant {int((~compliant).sum())}")

    if selftest:
        ref = json.loads((OUT / "final_predictions.json").read_text())
        diff_stock = diff_ok = 0
        for it in items:
            a, b = (
                result[it["camera"]][it["image"]][it["area_id"]],
                ref[it["camera"]][it["image"]][it["area_id"]],
            )
            diff_stock += a["stock_level"] != b["stock_level"]
            diff_ok += a["planogram_compliant"] != b["planogram_compliant"]
        typer.echo(
            f"SELF-TEST vs out/final_predictions.json: stock differs on {diff_stock}, compliance on {diff_ok} of {len(items)} areas "
            f"(expected: about 5 stock areas, which the OWL count rule changed and this script does not)"
        )


if __name__ == "__main__":
    app()
