"""Count loose items in the crops of every product the table marks method "count" (out/owl_counts.json).

OWL-ViT v2 with the prompt "a photo of a <noun>", confident boxes only (score >= 0.2), NMS 0.5, and boxes
that touch the crop border and are much smaller than the typical item dropped (they are pieces of the
neighbouring tray). Checked by hand on 24 crops: 54% exact, 75% within one item. Resumable.

    uv run --env-file .env python -m src.owl_count
"""

import json
from pathlib import Path

import cv2
import torch
import typer
from PIL import Image
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
SCORE, EDGE, SMALL = 0.2, 0.03, 0.6
NOUNS = [  # first match wins
    ("COUVE FLOR", "cauliflower"),
    ("BROCOLO", "broccoli"),
    ("COUVE", "cabbage"),
    ("MELANCIA", "watermelon"),
    ("MELAO", "melon"),
    ("MELOA", "melon"),
    ("MANGA", "mango"),
    ("MAMAO", "papaya"),
    ("PAPAIA", "papaya"),
    ("ABACAXI", "pineapple"),
    ("ABÓBORA", "pumpkin"),
    ("ABACATE", "avocado"),
    ("MAÇ", "apple"),
    ("MACA", "apple"),
    ("LARANJA", "orange"),
    ("CLEMENTINA", "clementine"),
    ("LIMA", "lime"),
    ("PERA", "pear"),
    ("PESSEGO", "peach"),
    ("NECTARINA", "nectarine"),
    ("ALPERCE", "apricot"),
    ("KIWI", "kiwi"),
    ("GOIABA", "guava"),
    ("GRANADILHA", "passion fruit"),
    ("MARACUJA", "passion fruit"),
    ("BATATA", "potato"),
    ("CEBOLA", "onion"),
    ("PIMENTO", "pepper"),
    ("CURGETE", "zucchini"),
    ("PEPINO", "cucumber"),
    ("BERINGELA", "eggplant"),
    ("CHUCHU", "chayote"),
    ("GENGIBRE", "ginger"),
    ("FEIJÃO", "green beans"),
    ("ALFACE", "lettuce"),
]

app = typer.Typer(add_completion=False)


def area(b: list[float]) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def touches(b: list[float], w: int, h: int) -> bool:
    """Whether a box reaches the crop border (a piece of the neighbouring tray)."""
    return b[0] <= w * EDGE or b[1] <= h * EDGE or b[2] >= w * (1 - EDGE) or b[3] >= h * (1 - EDGE)


def noun_of(name: str) -> str | None:
    return next((n for k, n in NOUNS if k in name), None)


@app.command()
def main() -> None:
    items = json.loads((OUT / "crops" / "index.json").read_text())
    groups = json.loads((OUT / "product_groups.json").read_text())
    todo = [
        i
        for i in items
        if groups[i["product_id"]]["method"] == "count" and noun_of(i["product_name"])
    ]
    skipped = sorted(
        {
            i["product_name"]
            for i in items
            if groups[i["product_id"]]["method"] == "count" and not noun_of(i["product_name"])
        }
    )
    typer.echo(f"{len(todo)} crops to count; no noun for: {skipped}")
    dest = OUT / "owl_counts.json"
    counts = json.loads(dest.read_text()) if dest.exists() else {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = (
        Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        .to(device)
        .eval()
    )
    for n, it in enumerate(todo):
        if it["key"] in counts:
            continue
        im = cv2.imread(str(OUT / "crops" / it["crop"]))
        pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        h, w = im.shape[:2]
        with torch.no_grad():
            out = owl(
                **proc(
                    text=[[f"a photo of a {noun_of(it['product_name'])}"]],
                    images=pil,
                    return_tensors="pt",
                ).to(device)
            )
        side = max(pil.size)
        res = proc.image_processor.post_process_object_detection(
            out, threshold=SCORE, target_sizes=torch.tensor([[side, side]]).to(device)
        )[0]
        boxes = [
            (b.tolist(), float(s))
            for b, s in zip(res["boxes"], res["scores"], strict=False)
            if b[0] < w and b[1] < h
        ]
        if boxes:
            keep = nms(
                torch.tensor([b for b, _ in boxes]), torch.tensor([s for _, s in boxes]), 0.5
            )
            boxes = [boxes[i][0] for i in keep]

        inner = [b for b in boxes if not touches(b, w, h)]
        if inner:
            med = sorted(area(b) for b in inner)[len(inner) // 2]
            boxes = [b for b in boxes if not (touches(b, w, h) and area(b) < SMALL * med)]
        counts[it["key"]] = len(boxes)
        if n % 50 == 0:
            dest.write_text(json.dumps(counts))
            typer.echo(f"{n}/{len(todo)}")
    dest.write_text(json.dumps(counts))
    typer.echo(f"{len(counts)} counts -> {dest}")


if __name__ == "__main__":
    app()
