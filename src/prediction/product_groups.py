"""Assign every planogram product to a purchase-size group (out/product_groups.json).

Rule: empty = one typical purchase would leave the tray bare or nearly bare. Groups give the typical
purchase and the method: "count" (OWL-ViT on loose items) or "fill" (CLIP probe, for packages/bunches).
First matching keyword wins, so order matters. Edit GROUPS, rerun, and review ambiguous names.

    uv run --env-file .env python -m src.prediction.product_groups
"""

import json
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "out"

# group: (typical purchase, "empty" means at most this many visible units, method)
GROUPS = {
    "A_very_large": ("1 item", 2, "count"),
    "B_large": ("1-2 items", 3, "count"),
    "C_medium_fruit": ("4-6 items", 6, "count"),
    "D_medium_veg": ("4-8 items", 8, "count"),
    "H_loose_tomato": ("4-6 items", 6, "fill"),  # OWL counts of dense tomato trays were 0% exact
    "E_small_pack": ("1-2 packs", 3, "fill"),
    "F_bunch": ("1 bunch", 2, "fill"),
    "G_bag": ("1 bag", 2, "fill"),
}
PACK = (
    "EMB",
    "SC ",
    "SACO",
    " 250",
    " 200",
    " 300",
    " 400",
    " 500",
    "CNT 1KG",
    "1.5 KG",
    "3KG",
    "4KG",
    " 5 KG",
    " 2 KG",
)
# Sold packaged on the shelf although the name says nothing (checked by eye in the review dashboards).
PACKAGED = {
    "MAÇA GOLDEN NACIONAL CAL. 65/70 1 KG",
    "MAÇA FUJI CARIOCA 65/75 1 KG IMP",
    "MAÇÂ GALA CNT 60/70 1 KG IMP",
    "CEBOLA NOVA CNT KG",
}
# Judge by how full the tray looks, never by counting items.
FILL_ONLY = {
    "KIWI ZESPRI KG",
    "LIMÃO CNT 60/75 KG",
    "MACA GRANNY SMITH CNT CAL.73/85 KG",
    "ALHO SECO KG",
}
RULES = [  # (substring in product name, group); checked in order
    ("UVA", "G_bag"),
    ("BANANA", "F_bunch"),
    ("CEBOLINHA", "F_bunch"),
    ("TOMATE CACHO", "F_bunch"),
    ("MELANCIA", "A_very_large"),
    ("ABÓBORA", "A_very_large"),
    ("MAMAO", "A_very_large"),
    ("PAPAIA", "A_very_large"),
    ("ABACAXI", "A_very_large"),
    ("COUVE", "A_very_large"),
    ("MELAO", "B_large"),
    ("MELOA", "B_large"),
    ("MANGA", "B_large"),
    ("ABACATE", "B_large"),
    ("ALFACE", "B_large"),
    ("CEREJA", "E_small_pack"),
    ("MORANGO", "E_small_pack"),
    ("TOMATE CHERRY", "E_small_pack"),
    ("TOMATE MINI", "E_small_pack"),
    ("MINI TOMATE", "E_small_pack"),
    ("TOMATE TASTY", "E_small_pack"),
    ("TOMATE", "H_loose_tomato"),
    ("BATATA", "D_medium_veg"),
    ("CEBOLA", "D_medium_veg"),
    ("ALHO", "D_medium_veg"),
    ("PIMENTO", "D_medium_veg"),
    ("CURGETE", "D_medium_veg"),
    ("PEPINO", "D_medium_veg"),
    ("BERINGELA", "D_medium_veg"),
    ("CHUCHU", "D_medium_veg"),
    ("GENGIBRE", "D_medium_veg"),
    ("FEIJÃO", "D_medium_veg"),
]


def group_of(name: str) -> tuple[str, bool]:
    """Return (group, ambiguous). Packaged names go to bag/small-pack groups; ambiguous = needs a human look."""
    if name in PACKAGED:
        return "G_bag", False
    packed = any(p in name for p in PACK)
    for key, g in RULES:
        if key in name:
            if packed and g == "D_medium_veg":
                return "G_bag", "CNT 1KG" in name or "500" in name
            if packed and g in ("C_medium_fruit", "H_loose_tomato"):
                return "E_small_pack", True
            return g, packed and g == "F_bunch"
    return ("G_bag" if packed else "C_medium_fruit"), packed


app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    items = json.loads((OUT / "crops" / "index.json").read_text())
    products = {i["product_id"]: i["product_name"] for i in items}
    table = {}
    for pid, name in sorted(products.items(), key=lambda x: x[1]):
        g, amb = group_of(name)
        typical, empty_max, method = GROUPS[g]
        if name in FILL_ONLY:
            method = "fill"
        table[pid] = dict(
            name=name,
            group=g,
            typical_purchase=typical,
            empty_max_units=empty_max,
            method=method,
            check=amb,
        )
    (OUT / "product_groups.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    by = {}
    for v in table.values():
        by.setdefault(v["group"], []).append(v["name"])
    for g, names in sorted(by.items()):
        typer.echo(
            f"{g} ({len(names)}): " + "; ".join(names[:6]) + (" ..." if len(names) > 6 else "")
        )
    typer.echo(f"{sum(v['check'] for v in table.values())} products flagged check=true")


if __name__ == "__main__":
    app()
