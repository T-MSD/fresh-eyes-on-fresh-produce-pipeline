from typing import Literal

from src.shelf.status import AreaStatus, StockLevel, area_status

TaskKind = Literal["restock", "move_product", "top_up"]

TASK_KINDS: dict[AreaStatus, TaskKind] = {
    "empty": "restock",
    "wrong_product": "move_product",
    "half": "top_up",
}


def task_kind(stock_level: StockLevel, planogram_compliant: bool) -> TaskKind | None:
    return TASK_KINDS.get(area_status(stock_level, planogram_compliant))
