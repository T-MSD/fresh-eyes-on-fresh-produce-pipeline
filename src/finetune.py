"""Fine-tune a vision network on the confident labels (manual + CLIP) and test it on manual labels.

Inputs per crop: the image and the product the planogram expects there (a learned embedding of its
generic class), because stock and "right product" both depend on it. Outputs: stock level (3
classes) and planogram_compliant.

  evaluate  (default) holds out whole cameras. Nothing from them is trained on, and the CLIP
            labels for the other cameras are regenerated from the training cameras' manual labels
            alone, so no information about the held-out labels leaks in. Scores only the held-out
            MANUAL labels, next to zero-shot CLIP and the current classifier on the same cameras.
  --final   regenerates confident labels from out/labels.json and CLIP outputs for every camera,
            then writes out/finetune/predictions.json, model.pt and the inference bundle
            out/finetune/bundle.joblib next to it. It does not touch out/predictions.json.
  --bundle-only  rebuilds bundle.joblib from the current crops index and labels, without training.

    uv run --extra ml --env-file .env python -m src.finetune --holdout 6371,6443,6591
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

from src.clip_predict import LEVELS, POINTS, STOCK_VALUES, product_class
from src.prediction_io import merge_fallbacks
from src.train_classifier import features, fit_classifier, fit_proba

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
WORK = OUT / "finetune"
BUNDLE = WORK / "bundle.joblib"
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
        "train_classes": sorted({product_class(i["product_name"]) for i in items}),
        "emb_mean": mean,
        "emb_std": std,
        "classifier": fit_classifier(x[rows], y, 0.1),
        "fill_model": Ridge(alpha=300.0).fit(
            x[[key_row[k] for k in fill_keys]], np.array([fill[k] for k in fill_keys], dtype=float)
        ),
    }


def load_fill_labels(out_dir: Path = OUT) -> dict:
    return {
        k: v["fill"]
        for k, v in json.loads((out_dir / "fill_labels.json").read_text()).items()
        if not v.get("wrong")
    }


def regenerate_labels(items, manual, train_cams, emb, zs_probs, zero) -> dict:
    """CLIP labels as before, but fitted on the manual labels of the training cameras only."""
    x = features(emb, zs_probs, False)
    rows = [n for n, i in enumerate(items) if i["key"] in manual and i["camera"] in train_cams]
    y = np.array([LEVELS.index(manual[items[n]["key"]]["stock_level"]) for n in rows])
    p = fit_proba(x[rows], y, x, 3, 0.1)
    labels = {}
    for n, item in enumerate(items):
        k = item["key"]
        if item["camera"] not in train_cams:
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
    holdout: str = "6371,6443,6591",
    final: bool = False,
    bundle_only: bool = False,
    backbone: str = "resnet50.a1_in1k",
    size: int = 224,
    epochs: int = 8,
    batch: int = 32,
    manual_weight: float = 3.0,
    seed: int = 0,
    ignore: Path | None = None,
    out_dir: Path = OUT,
    pretrained: bool = True,
) -> None:
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
    manual_train = {k: v for k, v in manual.items() if k not in dropped}
    if bundle_only:
        work_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            build_bundle(items, manual, emb, load_fill_labels(out_dir)), work_dir / BUNDLE.name
        )
        typer.echo(f"wrote {work_dir / BUNDLE.name}")
        return
    cams = sorted({i["camera"] for i in items})
    test_cams = set() if final else set(holdout.split(","))
    train_cams = set(cams) - test_cams

    labels = regenerate_labels(items, manual_train, train_cams, emb, zs_probs, zero)

    classes = sorted({product_class(i["product_name"]) for i in items})
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
        f"{len(train_idx) - n_manual} clip); held-out cameras: {sorted(test_cams) or 'none'}"
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

    test_idx = [n for n, i in enumerate(items) if i["camera"] in test_cams and i["key"] in manual]
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
        if test_idx and (epoch + 1) % 2 == 0:
            p_stock, _ = predict(model, imgs[test_idx], cls[test_idx], device)
            truth = np.array(
                [LEVELS.index(manual[items[n]["key"]]["stock_level"]) for n in test_idx]
            )
            msg += f" | held-out stock points {POINTS[(p_stock @ POINTS.T).argmax(1), truth].mean():+.2f}"
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
        joblib.dump(
            build_bundle(items, manual, emb, load_fill_labels(out_dir)), work_dir / BUNDLE.name
        )
        typer.echo(
            f"wrote out/finetune/predictions.json, model.pt and bundle.joblib for {len(items)} areas "
            "(out/predictions.json untouched)"
        )
        return

    # Held-out comparison on manual labels only.
    truth = np.array([LEVELS.index(manual[items[n]["key"]]["stock_level"]) for n in test_idx])
    true_ok = np.array([int(manual[items[n]["key"]]["planogram_compliant"]) for n in test_idx])
    p_stock, p_ok = predict(model, imgs[test_idx], cls[test_idx], device)
    keys = [items[n]["key"] for n in test_idx]

    def report(name, level, ok):
        pts = POINTS[level, truth].mean()
        ident = np.where(np.asarray(ok) == true_ok, 1, -2).mean()
        half = truth == 1
        typer.echo(
            f"  {name:28} stock {pts:+.2f}  exact {np.mean(level == truth):4.0%}  "
            f"half found {int((level[half] == 1).sum())}/{int(half.sum())}  identification {ident:+.2f}"
        )

    typer.echo(
        f"\nheld-out cameras {sorted(test_cams)}: {len(test_idx)} manual labels "
        f"(stock empty/half/full {np.bincount(truth, minlength=3).tolist()}, wrong product {int((true_ok == 0).sum())})"
    )
    zl = np.array([LEVELS.index(zero[k]["stock_level"]) for k in keys])
    zo = np.array([int(zero[k]["planogram_compliant"]) for k in keys])
    rows = [
        n for n, i in enumerate(items) if i["key"] in manual_train and i["camera"] in train_cams
    ]
    xs = features(emb, zs_probs, False)
    y_tr = np.array([LEVELS.index(manual_train[items[n]["key"]]["stock_level"]) for n in rows])
    p_lr = fit_proba(xs[rows], y_tr, xs[test_idx], 3, 0.1)
    report("zero-shot CLIP", zl, zo)
    report("CLIP + classifier (current)", (p_lr @ POINTS.T).argmax(1), zo)
    report("fine-tuned network", (p_stock @ POINTS.T).argmax(1), (p_ok >= 0.5).astype(int))
    report("fine-tuned stock + zero-shot id", (p_stock @ POINTS.T).argmax(1), zo)
    for w in (0.25, 0.5, 0.75):  # average the two models' stock probabilities, w = network's share
        blend = w * p_stock + (1 - w) * p_lr
        report(f"average, network {w:.0%}", (blend @ POINTS.T).argmax(1), zo)


if __name__ == "__main__":
    app()
