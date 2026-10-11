"""Zero-shot CLIP predictions for every cropped planogram area.

For each crop, using the product the planogram expects there:
  planogram_compliant  expected product is among CLIP's top-k guesses (over all products in the
                       store), or the shelf looks empty (nothing to judge, so compliant)
  stock_level          CLIP picks between "empty", "a few" and "full" prompts for that product;
                       the level is chosen to maximise expected points under the assignment's
                       scoring (+1 correct, -2 too low, -3 too high), so doubt leans low

Writes out/predictions.json (assignment format), out/prelabels.json (pre-fills the review tool),
out/clip_zeroshot.json (same, kept as a baseline) and out/clip_embeddings.npz (image embeddings, reusable to train a classifier on manual labels).

    uv run --extra ml --env-file .env python -m src.clip_predict
"""

import json
import unicodedata
from pathlib import Path

import numpy as np
import typer

from src.prediction_io import merge_fallbacks

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
LEVELS = ("empty", "half", "full")
STOCK_VALUES = {"empty": 0, "half": 0.5, "full": 1}
# points[chosen][truth], from the assignment: correct +1, below real -2, above real -3
POINTS = np.array([[1, -2, -2], [-3, 1, -2], [-3, -3, 1]])

# Ordered (tokens that must all appear, English class); the first match wins.
CLASS_RULES: list[tuple[set[str], str]] = [
    ({"BATATA", "DOCE"}, "sweet potatoes"),
    ({"BATATA", "VERMELHA"}, "red potatoes"),
    ({"BATATA"}, "potatoes"),
    ({"CEBOLA", "ROXA"}, "red onions"),
    ({"CEBOLINHA"}, "shallots"),
    ({"CEBOLA"}, "onions"),
    ({"COUVE", "FLOR"}, "cauliflower"),
    ({"COUVE", "BROCOLO"}, "broccoli"),
    ({"COUVE", "ROXA"}, "red cabbage"),
    ({"COUVE", "BRANCA"}, "white cabbage"),
    ({"COUVE"}, "collard greens"),
    ({"KIWI", "GOLD"}, "yellow kiwis"),
    ({"KIWI"}, "kiwis"),
    ({"MACA", "GRANNY"}, "green apples"),
    ({"MACA", "GOLDEN"}, "yellow apples"),
    ({"MACA"}, "red apples"),
    ({"MELANCIA"}, "watermelon"),
    ({"MELAO"}, "honeydew melon"),
    ({"MELOA", "GALIA"}, "galia melon"),
    ({"MELOA"}, "cantaloupe melon"),
    ({"TOMATE", "CHERRY"}, "cherry tomatoes"),
    ({"MINI", "TOMATE"}, "cherry tomatoes"),
    ({"TOMATE", "MINI"}, "cherry tomatoes"),
    ({"TOMATE"}, "tomatoes"),
    ({"PIMENTO", "VERDE"}, "green bell peppers"),
    ({"PIMENTO"}, "red bell peppers"),
    ({"UVA", "BRANCA"}, "green grapes"),
    ({"UVA"}, "red grapes"),
    ({"ABACATE"}, "avocados"),
    ({"ABACAXI"}, "pineapples"),
    ({"ABOBORA"}, "pumpkins"),
    ({"ALFACE"}, "lettuce"),
    ({"ALHO"}, "garlic"),
    ({"ALPERCE"}, "apricots"),
    ({"BANANA"}, "bananas"),
    ({"BERINGELA"}, "eggplants"),
    ({"CEREJA"}, "cherries"),
    ({"CHUCHU"}, "chayote"),
    ({"CITR"}, "oranges"),
    ({"CITRINOS"}, "oranges"),
    ({"CLEMENTINA"}, "clementines"),
    ({"CURGETE"}, "zucchini"),
    ({"FEIJAO"}, "green beans"),
    ({"GENGIBRE"}, "ginger"),
    ({"GOIABA"}, "guavas"),
    ({"GRANADILHA"}, "passion fruit"),
    ({"MARACUJA"}, "passion fruit"),
    ({"LIMAO"}, "lemons"),
    ({"LIMA"}, "limes"),
    ({"MAMAO"}, "papayas"),
    ({"PAPAIA"}, "papayas"),
    ({"MANGA"}, "mangoes"),
    ({"MORANGO"}, "strawberries"),
    ({"NECTARINA"}, "nectarines"),
    ({"PEPINO"}, "cucumbers"),
    ({"PERA"}, "pears"),
    ({"PESSEGO"}, "peaches"),
    ({"PITAYA"}, "dragon fruit"),
]

app = typer.Typer(add_completion=False)


def product_class(name: str) -> str:
    """Portuguese catalogue name -> generic English class, e.g. 'MAÇÃ GALA ...' -> 'red apples'."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper()
    tokens = set(plain.replace("/", " ").replace(".", " ").replace(":", " ").split())
    for needed, cls in CLASS_RULES:
        if needed <= tokens:
            return cls
    raise ValueError(f"no class rule for product {name!r}")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def embed_images(paths: list[Path], model_name: str, pretrained: str, batch: int):
    import open_clip
    import torch
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    def encode_text(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            t = model.encode_text(tokenizer(texts).to(device))
            return torch.nn.functional.normalize(t, dim=-1).cpu().numpy()

    chunks = []
    for i in range(0, len(paths), batch):
        imgs = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch]]
        ).to(device)
        with torch.no_grad():
            f = torch.nn.functional.normalize(model.encode_image(imgs), dim=-1)
        chunks.append(f.cpu().numpy())
        typer.echo(f"  embedded {min(i + batch, len(paths))}/{len(paths)}")
    return np.concatenate(chunks), encode_text, float(model.logit_scale.exp().item())


@app.command()
def main(
    model_name: str = "ViT-B-16-quickgelu",
    pretrained: str = "openai",
    batch: int = 32,
    topk: int = 3,
    reuse: bool = True,
) -> None:
    items = json.loads((OUT / "crops" / "index.json").read_text())
    if not items:
        predictions: dict = {}
        merge_fallbacks(predictions, OUT / "crops")
        (OUT / "predictions.json").write_text(json.dumps(predictions, indent=1))
        (OUT / "prelabels.json").write_text("{}")
        (OUT / "clip_zeroshot.json").write_text("{}")
        return
    classes = [product_class(i["product_name"]) for i in items]
    all_classes = sorted(set(classes))
    typer.echo(f"{len(items)} crops, {len(all_classes)} product classes")

    cache = OUT / "clip_embeddings.npz"
    paths = [OUT / "crops" / i["crop"] for i in items]
    if reuse and cache.exists() and list(np.load(cache)["keys"]) == [i["key"] for i in items]:
        img = np.load(cache)["embeddings"]
        import open_clip
        import torch

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        tok = open_clip.get_tokenizer(model_name)

        def encode_text(texts: list[str]) -> np.ndarray:
            with torch.no_grad():
                t = model.encode_text(tok(texts))
                return torch.nn.functional.normalize(t, dim=-1).numpy()

        scale = float(model.logit_scale.exp().item())
    else:
        img, encode_text, scale = embed_images(paths, model_name, pretrained, batch)
        np.savez(cache, keys=np.array([i["key"] for i in items]), embeddings=img)

    # Which product is it? One prompt per class, shared by all crops.
    class_txt = encode_text([f"a photo of {c} on a supermarket shelf" for c in all_classes])
    class_prob = softmax(scale * img @ class_txt.T)
    class_idx = {c: n for n, c in enumerate(all_classes)}

    # How full is it? Three prompts per class, used for the crops that should hold that class.
    level_txt = {
        c: encode_text(
            [
                f"an empty supermarket shelf with no {c}",
                f"a supermarket shelf with only a few {c}",
                f"a supermarket shelf full of piled {c}",
            ]
        )
        for c in all_classes
    }

    predictions = {}
    pre: dict = {}
    level_probs = np.zeros((len(items), 3))
    for n, (item, cls) in enumerate(zip(items, classes, strict=False)):
        p_level = softmax(scale * img[n] @ level_txt[cls].T)
        level_probs[n] = p_level
        level = LEVELS[int(np.argmax(p_level @ POINTS.T))]
        top = np.argsort(class_prob[n])[::-1][:topk]
        compliant = level == "empty" or class_idx[cls] in top
        result = {
            "planogram_compliant": bool(compliant),
            "stock_level": level,
            "stock_value": STOCK_VALUES[level],
        }
        predictions.setdefault(item["camera"], {}).setdefault(item["image"], {})[
            item["area_id"]
        ] = result
        pre[item["key"]] = {
            "planogram_compliant": result["planogram_compliant"],
            "stock_level": level,
            "unsure": bool(p_level.max() < 0.6),
        }

    merge_fallbacks(predictions, OUT / "crops")
    (OUT / "predictions.json").write_text(json.dumps(predictions, indent=1))
    (OUT / "prelabels.json").write_text(json.dumps(pre, indent=1))
    np.save(OUT / "clip_level_probs.npy", level_probs)  # extra features for train_classifier.py
    (OUT / "clip_zeroshot.json").write_text(
        json.dumps(pre, indent=1)
    )  # baseline for train_classifier.py
    levels = [p["stock_level"] for p in pre.values()]
    typer.echo(
        f"{len(pre)} areas -> out/predictions.json; "
        + ", ".join(f"{lv} {levels.count(lv)}" for lv in LEVELS)
        + f"; non-compliant {sum(not p['planogram_compliant'] for p in pre.values())}"
    )


if __name__ == "__main__":
    app()
