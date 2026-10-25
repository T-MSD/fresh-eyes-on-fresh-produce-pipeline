from typing import Literal

StockLevel = Literal["empty", "half", "full"]
AreaStatus = Literal["empty", "wrong_product", "half", "ok", "unknown"]

STOCK_VALUES: dict[StockLevel, float] = {"empty": 0.0, "half": 0.5, "full": 1.0}


def area_status(stock_level: StockLevel | None, planogram_compliant: bool | None) -> AreaStatus:
    """What an area needs: `unknown` without a stock level, else `empty`, else `wrong_product`
    when not compliant, else `half`, else `ok`."""
    if stock_level is None:
        return "unknown"
    if stock_level == "empty":
        return "empty"
    if not planogram_compliant:
        return "wrong_product"
    if stock_level == "half":
        return "half"
    return "ok"
