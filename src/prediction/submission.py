from pathlib import Path

from pydantic import BaseModel, TypeAdapter, computed_field

from src.shelf.status import STOCK_VALUES, StockLevel


class Label(BaseModel):
    planogram_compliant: bool
    stock_level: StockLevel

    @computed_field
    @property
    def stock_value(self) -> float:
        return STOCK_VALUES[self.stock_level]


class NetworkCrop(BaseModel):
    product_name: str
    crop: str


Submission = dict[str, dict[str, dict[str, Label]]]
SUBMISSION = TypeAdapter(Submission)


def fallback_keys(path: Path) -> set[tuple[str, str, str]]:
    """The (camera, frame, area) triples of a fallbacks.json file written by crop_areas."""
    return {
        (camera, frame, area)
        for camera, by_frame in SUBMISSION.validate_json(path.read_bytes()).items()
        for frame, by_area in by_frame.items()
        for area in by_area
    }


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
