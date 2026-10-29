"""Fine-tune a vision network on the confident labels (manual + CLIP) and test it on manual labels.

Inputs per crop: the image and the product the planogram expects there (a learned embedding of its
generic class), because stock and "right product" both depend on it. Outputs: stock level (3
classes) and planogram_compliant.

  evaluate  (default) trains on the training frames of labels/split.json (python -m src.split_labels)
            and scores the manual labels of its test frames, next to zero-shot CLIP and the CLIP
            classifier. Nothing from a test frame is trained on: not its manual labels, CLIP labels
            or fill labels. Writes out/finetune/metrics.json.
  --final   trains on every label, ignoring the split, and writes out/finetune/predictions.json,
            model.pt, the inference bundle bundle.joblib and metadata.json. It does not touch
            out/predictions.json.
  --bundle-only  rebuilds bundle.joblib from the current crops index and every label, without
            training. It stops unless the classes of that index match the ones in
            out/finetune/classes.json, written next to model.pt.

metrics.json and metadata.json record what produced them: params, git commit and the sha256 of the
label, fill-label and split files.

    uv run --extra ml --env-file .env python -m src.finetune
    uv run --extra ml --env-file .env python -m src.finetune --final
"""

import json
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
from PIL import Image
from sklearn.linear_model import Ridge

from src.blend import SHIPPED, W_NET
from src.classifier import features, fit_classifier, fit_proba
from src.clip_predict import LEVELS, POINTS, STOCK_VALUES, product_class
from src.prediction_io import merge_fallbacks
from src.provenance import provenance, repo_path
from src.split_labels import SPLIT, frame_of, held_out_window, load_test_frames

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
WORK = OUT / "finetune"
BUNDLE = WORK / "bundle.joblib"
CLASSES = WORK / "classes.json"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

app = typer.Typer(add_completion=False)


def load_crops(
    items: list[dict], size: int, crops_dir: Path = OUT / "crops", work_dir: Path = WORK
) -> torch.Tensor:
    """All crops resized (squashed) to size x size, uint8 NCHW, cached on disk."""
    cache = work_dir / f"crops{size}.npy"
    if cache.exists():
        return torch.from_numpy(np.load(cache))
    work_dir.mkdir(parents=True, exist_ok=True)
    arr = np.stack(
        [
            np.asarray(Image.open(crops_dir / i["crop"]).convert("RGB").resize((size, size)))
            for i in items
        ]
    ).transpose(0, 3, 1, 2)
    np.save(cache, arr)
    return torch.from_numpy(arr)


class Net(nn.Module):
    def __init__(self, backbone: str, n_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.product = nn.Embedding(n_classes, 32)
        self.stock = nn.Sequential(nn.Dropout(0.3), nn.Linear(d + 32, 3))
        self.ok = nn.Sequential(nn.Dropout(0.3), nn.Linear(d + 32, 1))

    def forward(self, x: torch.Tensor, cls: torch.Tensor):
        h = torch.cat([self.backbone(x), self.product(cls)], dim=1)
        return self.stock(h), self.ok(h).squeeze(1)


def prepare(x: torch.Tensor, train: bool) -> torch.Tensor:
    x = x.float() / 255
    if train:  # horizontal flip, brightness/contrast jitter, mild random crop
        flip = torch.rand(len(x)) < 0.5
        x[flip] = x[flip].flip(-1)
        x = (x - 0.5) * (1 + 0.4 * (torch.rand(len(x), 1, 1, 1) - 0.5)) + 0.5
        x = x + 0.2 * (torch.rand(len(x), 1, 1, 1) - 0.5)
        s = int(x.shape[-1] * (0.85 + 0.15 * float(torch.rand(1))))
        i, j = (int(torch.randint(0, x.shape[-1] - s + 1, (1,))) for _ in range(2))
        x = F.interpolate(x[..., i : i + s, j : j + s], size=x.shape[-2:], mode="bilinear")
    return (x.clamp(0, 1) - MEAN) / STD


def predict(model, imgs, cls, device, batch=64):
    model.eval()
    stock, ok = [], []
    with torch.no_grad():
        for i in range(0, len(imgs), batch):
            s, o = model(
                prepare(imgs[i : i + batch], False).to(device), cls[i : i + batch].to(device)
            )
            stock.append(s.softmax(-1).cpu())
            ok.append(o.sigmoid().cpu())
    return torch.cat(stock).numpy(), torch.cat(ok).numpy()


def build_bundle(items: list[dict], manual: dict, emb: np.ndarray, fill: dict) -> dict:
    """Everything run_test needs from the training data, so it never reads the training crops.

    train_classes is the class order of the network head: model.pt is only valid with the
    bundle written next to it.
    """
    mean, std = emb.mean(axis=0), emb.std(axis=0)
    x = (emb - mean) / (std + 1e-6)
    rows = [n for n, i in enumerate(items) if i["key"] in manual]
    y = np.array([LEVELS.index(manual[items[n]["key"]]["stock_level"]) for n in rows])
    key_row = {i["key"]: n for n, i in enumerate(items)}
    fill_keys = [k for k in fill if k in key_row]
    return {
        "train_classes": head_classes(items),
        "emb_mean": mean,
        "emb_std": std,
        "classifier": fit_classifier(x[rows], y, 0.1),
        "fill_model": Ridge(alpha=300.0).fit(
            x[[key_row[k] for k in fill_keys]], np.array([fill[k] for k in fill_keys], dtype=float)
        ),
    }


def method_metrics(
    level: np.ndarray, ok: np.ndarray, truth: np.ndarray, true_ok: np.ndarray
) -> dict:
    """Per-area points under the brief's scoring, exact stock share and the stock confusion matrix."""
    stock = float(POINTS[level, truth].mean())
    identification = float(np.where(np.asarray(ok) == true_ok, 1, -2).mean())
    half = truth == 1
    confusion = np.zeros((3, 3), dtype=int)
    np.add.at(confusion, (truth, level), 1)
    return {
        "points_per_area": stock + identification,
        "stock_points": stock,
        "identification_points": identification,
        "exact_stock": float(np.mean(level == truth)),
        "half_found": [int((level[half] == 1).sum()), int(half.sum())],
        "confusion": {"labels": list(LEVELS), "rows_truth_columns_predicted": confusion.tolist()},
    }


def head_classes(items: list[dict]) -> list[str]:
    return sorted({product_class(i["product_name"]) for i in items})


def check_head_classes(work_dir: Path, items: list[dict]) -> None:
    """Stop unless the classes of `items` are, in order, the ones model.pt was trained with."""
    path = work_dir / CLASSES.name
    if not path.exists():
        raise typer.BadParameter(
            f"{path} is missing, so the class order of model.pt is unknown: run finetune --final"
        )
    trained, current = json.loads(path.read_text()), head_classes(items)
    if trained != current:
        raise typer.BadParameter(
            f"model.pt was trained with {trained}, the crops index now gives {current}: "
            "a bundle built from it would score the wrong products. Run finetune --final"
        )


def load_fill_labels(out_dir: Path = OUT) -> dict:
    return {
        k: v["fill"]
        for k, v in json.loads((out_dir / "fill_labels.json").read_text()).items()
        if not v.get("wrong")
    }


def regenerate_labels(items, manual, emb, zs_probs, zero, held=frozenset()) -> dict:
    """Training labels: manual where labelled, else the CLIP classifier fitted on the manual labels.

    Crops of the `held` frames get no label, so nothing about them is trained on.
    """
    x = features(emb, zs_probs, False)
    rows = [n for n, i in enumerate(items) if i["key"] in manual]
    y = np.array([LEVELS.index(manual[items[n]["key"]]["stock_level"]) for n in rows])
    p = fit_proba(x[rows], y, x, 3, 0.1)
    labels = {}
    for n, item in enumerate(items):
        k = item["key"]
        if frame_of(k) in held:
            continue
        if k in manual:
            m = manual[k]
            labels[k] = (
                m["planogram_compliant"],
                m["stock_level"],
                "manual",
                not m.get("unsure", False),
            )
        else:
            level = LEVELS[int((p[n] @ POINTS.T).argmax())]
            labels[k] = (zero[k]["planogram_compliant"], level, "clip", bool(p[n].max() >= 0.6))
    return labels


@app.command()
def main(
    final: bool = False,
    bundle_only: bool = False,
    backbone: str = "resnet50.a1_in1k",
    size: int = 224,
    epochs: int = 8,
    batch: int = 32,
    manual_weight: float = 3.0,
    seed: int = 0,
    ignore: Path | None = None,
    split: Path = SPLIT,
    out_dir: Path = OUT,
    pretrained: bool = True,
) -> None:
    params = {
        "backbone": backbone,
        "size": size,
        "epochs": epochs,
        "batch": batch,
        "manual_weight": manual_weight,
        "seed": seed,
        "pretrained": pretrained,
        "final": final,
        "ignore": str(ignore) if ignore else None,
        "out_dir": repo_path(out_dir),
    }
    evaluate = not (final or bundle_only)
    if evaluate and not split.exists():
        raise typer.BadParameter(
            f"{split} is missing: run python -m src.split_labels once, then evaluate"
        )
    inputs = {
        "labels": out_dir / "labels.json",
        "fill_labels": out_dir / "fill_labels.json",
        "split": split if evaluate else None,
    }
    torch.manual_seed(seed)
    np.random.seed(seed)
    work_dir = out_dir / "finetune"
    items = json.loads((out_dir / "crops" / "index.json").read_text())
    manual = json.loads((out_dir / "labels.json").read_text())
    zero = json.loads((out_dir / "clip_zeroshot.json").read_text())
    emb = np.load(out_dir / "clip_embeddings.npz")["embeddings"]
    zs_probs = np.load(out_dir / "clip_level_probs.npy")
    # --ignore keys.json: drop these manual labels from TRAINING only (the test still scores them),
    # to measure what a batch of labels added.
    dropped = set(json.loads(ignore.read_text())) if ignore else set()
    held = load_test_frames(split) if evaluate else set()
    excluded = held_out_window(held, {frame_of(i["key"]) for i in items})
    manual_fit = {k: v for k, v in manual.items() if frame_of(k) not in excluded}
    manual_train = {k: v for k, v in manual_fit.items() if k not in dropped}
    fill = {k: v for k, v in load_fill_labels(out_dir).items() if frame_of(k) not in excluded}
    if bundle_only:
        check_head_classes(work_dir, items)
        joblib.dump(build_bundle(items, manual_fit, emb, fill), work_dir / BUNDLE.name)
        typer.echo(f"wrote {work_dir / BUNDLE.name}")
        return
    labels = regenerate_labels(items, manual_train, emb, zs_probs, zero, excluded)

    classes = head_classes(items)
    cls = torch.tensor([classes.index(product_class(i["product_name"])) for i in items])
    imgs = load_crops(items, size, out_dir / "crops", work_dir)

    train_idx = [n for n, i in enumerate(items) if i["key"] in labels and labels[i["key"]][3]]
    y_stock = torch.tensor([LEVELS.index(labels[items[n]["key"]][1]) for n in train_idx])
    y_ok = torch.tensor([float(labels[items[n]["key"]][0]) for n in train_idx])
    weight = torch.tensor(
        [manual_weight if labels[items[n]["key"]][2] == "manual" else 1.0 for n in train_idx]
    )
    n_manual = sum(labels[items[n]["key"]][2] == "manual" for n in train_idx)
    typer.echo(
        f"training on {len(train_idx)} confident labels ({n_manual} manual, "
        f"{len(train_idx) - n_manual} clip); held-out test frames: {len(held)}, "
        f"frames excluded from training: {len(excluded)}"
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Net(backbone, len(classes), pretrained=pretrained).to(device)
    head = (
        list(model.product.parameters())
        + list(model.stock.parameters())
        + list(model.ok.parameters())
    )
    opt = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": 1e-4}, {"params": head, "lr": 1e-3}],
        weight_decay=0.05,
    )
    steps = epochs * ((len(train_idx) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[1e-4, 1e-3], total_steps=steps, pct_start=0.15
    )

    test_idx = [n for n, i in enumerate(items) if i["key"] in manual and frame_of(i["key"]) in held]
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train_idx))
        total = 0.0
        for b in range(0, len(perm), batch):
            sel = perm[b : b + batch]
            rows = torch.tensor([train_idx[j] for j in sel])
            s, o = model(prepare(imgs[rows], True).to(device), cls[rows].to(device))
            w = weight[sel].to(device)
            loss = (
                F.cross_entropy(s, y_stock[sel].to(device), reduction="none", label_smoothing=0.1)
                * w
            ).mean()
            loss = (
                loss
                + (
                    F.binary_cross_entropy_with_logits(o, y_ok[sel].to(device), reduction="none")
                    * w
                ).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(sel)
        msg = f"epoch {epoch + 1}/{epochs} loss {total / len(perm):.3f}"
        typer.echo(msg)

    if final:
        work_dir.mkdir(exist_ok=True)
        p_stock, p_ok = predict(model, imgs, cls, device)
        level = (p_stock @ POINTS.T).argmax(1)
        predictions: dict = {}
        for n, item in enumerate(items):
            lv = LEVELS[level[n]]
            predictions.setdefault(item["camera"], {}).setdefault(item["image"], {})[
                item["area_id"]
            ] = {
                "planogram_compliant": bool(p_ok[n] >= 0.5),
                "stock_level": lv,
                "stock_value": STOCK_VALUES[lv],
            }
        merge_fallbacks(predictions, out_dir / "crops")
        (work_dir / "predictions.json").write_text(json.dumps(predictions, indent=1))
        np.save(work_dir / "stock_probs.npy", p_stock)
        np.save(work_dir / "ok_probs.npy", p_ok)
        torch.save(model.state_dict(), work_dir / "model.pt")
        (work_dir / CLASSES.name).write_text(json.dumps(classes))
        joblib.dump(build_bundle(items, manual_fit, emb, fill), work_dir / BUNDLE.name)
        metadata = provenance(params, inputs) | {
            "trained_on": {
                "labels": len(train_idx),
                "manual": n_manual,
                "clip": len(train_idx) - n_manual,
                "test_frames_excluded": len(held),
            }
        }
        (work_dir / "metadata.json").write_text(json.dumps(metadata, indent=1))
        typer.echo(
            f"wrote out/finetune/predictions.json, model.pt, bundle.joblib and metadata.json for "
            f"{len(items)} areas (out/predictions.json untouched)"
        )
        return

    # Held-out comparison on manual labels only.
    truth = np.array([LEVELS.index(manual[items[n]["key"]]["stock_level"]) for n in test_idx])
    true_ok = np.array([int(manual[items[n]["key"]]["planogram_compliant"]) for n in test_idx])
    p_stock, p_ok = predict(model, imgs[test_idx], cls[test_idx], device)
    keys = [items[n]["key"] for n in test_idx]

    methods: dict[str, dict] = {}

    def report(name, level, ok):
        m = methods[name] = method_metrics(level, np.asarray(ok), truth, true_ok)
        found, halves = m["half_found"]
        typer.echo(
            f"  {name:28} stock {m['stock_points']:+.2f}  exact {m['exact_stock']:4.0%}  "
            f"half found {found}/{halves}  identification {m['identification_points']:+.2f}"
        )

    typer.echo(
        f"\nheld out ({len(held)} frames): "
        f"{len(test_idx)} manual labels "
        f"(stock empty/half/full {np.bincount(truth, minlength=3).tolist()}, wrong product {int((true_ok == 0).sum())})"
    )
    zl = np.array([LEVELS.index(zero[k]["stock_level"]) for k in keys])
    zo = np.array([int(zero[k]["planogram_compliant"]) for k in keys])
    rows = [n for n, i in enumerate(items) if i["key"] in manual_train]
    xs = features(emb, zs_probs, False)
    y_tr = np.array([LEVELS.index(manual_train[items[n]["key"]]["stock_level"]) for n in rows])
    p_lr = fit_proba(xs[rows], y_tr, xs[test_idx], 3, 0.1)
    report("zero-shot CLIP", zl, zo)
    report("CLIP + classifier (current)", (p_lr @ POINTS.T).argmax(1), zo)
    report("fine-tuned network", (p_stock @ POINTS.T).argmax(1), (p_ok >= 0.5).astype(int))
    report("fine-tuned stock + zero-shot id", (p_stock @ POINTS.T).argmax(1), zo)
    for w in sorted(
        {0.25, 0.5, W_NET}
    ):  # average the two models' stock probabilities, w = network's share
        blend = w * p_stock + (1 - w) * p_lr
        report(f"average, network {w:.0%}", (blend @ POINTS.T).argmax(1), zo)

    metrics = provenance(params, inputs) | {
        "test_set": {
            "frames": len(held),
            "frames_excluded_from_training": len(excluded),
            "areas": len(test_idx),
            "stock_empty_half_full": np.bincount(truth, minlength=3).tolist(),
            "wrong_product": int((true_ok == 0).sum()),
        },
        "shipped": SHIPPED,
        "methods": methods,
    }
    (work_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    typer.echo(
        f"\n{SHIPPED} (shipped, before the fill rule): "
        f"{methods[SHIPPED]['points_per_area']:+.2f} points per area; wrote {work_dir / 'metrics.json'}"
    )


if __name__ == "__main__":
    app()
