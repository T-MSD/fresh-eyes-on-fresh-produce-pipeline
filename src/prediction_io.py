from pathlib import Path
from typing import Literal

from pydantic import BaseModel, TypeAdapter, computed_field

StockLevel = Literal["empty", "half", "full"]
VALUES = {"empty": 0.0, "half": 0.5, "full": 1.0}


class Label(BaseModel):
    planogram_compliant: bool
    stock_level: StockLevel

    @computed_field
    @property
    def stock_value(self) -> float:
        return VALUES[self.stock_level]


class NetworkCrop(BaseModel):
    product_name: str
    crop: str


Submission = dict[str, dict[str, dict[str, Label]]]
SUBMISSION = TypeAdapter(Submission)
MANUAL_LABELS = TypeAdapter(dict[str, Label])


def merge_fallbacks(result: dict, crops_dir: Path) -> None:
    path = crops_dir / "fallbacks.json"
    if not path.exists():
        return
    fallbacks = SUBMISSION.validate_json(path.read_bytes())
    for camera, frames in fallbacks.items():
        for filename, areas in frames.items():
            target = result.setdefault(camera, {}).setdefault(filename, {})
            for area, label in areas.items():
                target.setdefault(area, label.model_dump())


def score_submission(predictions: Submission, labels: dict[str, Label]) -> float:
    if not labels:
        raise ValueError("Cannot score an empty label set")
    total = 0
    for key, truth in labels.items():
        camera, filename, area = key.split("/")
        prediction = predictions.get(camera, {}).get(filename, {}).get(area)
        if prediction is None:
            total -= 5
            continue
        total += 1 if prediction.planogram_compliant == truth.planogram_compliant else -2
        if prediction.stock_level == truth.stock_level:
            total += 1
        else:
            total += -2 if prediction.stock_value < truth.stock_value else -3
    return total / len(labels)
