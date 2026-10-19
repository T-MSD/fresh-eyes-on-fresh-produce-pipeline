from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from src.prediction_io import SUBMISSION, Label, Submission
from src.split_labels import load_test_frames

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_AREAS = 289
app = typer.Typer(add_completion=False)


@dataclass(frozen=True)
class Tally:
    identification_correct: int = 0
    identification_wrong: int = 0
    stock_correct: int = 0
    stock_below: int = 0
    stock_above: int = 0
    missing: int = 0

    @property
    def areas(self) -> int:
        return self.identification_correct + self.identification_wrong

    @property
    def points(self) -> int:
        return (
            self.identification_correct
            - 2 * self.identification_wrong
            + self.stock_correct
            - 2 * self.stock_below
            - 3 * self.stock_above
        )


@dataclass(frozen=True)
class Estimate:
    tally: Tally
    per_area: float
    expected_total: float
    low: float
    high: float
    validation_areas: int


def score_area(prediction: Label | None, truth: Label) -> Tally:
    if prediction is None:
        return Tally(identification_wrong=1, stock_above=1, missing=1)
    compliant = prediction.planogram_compliant == truth.planogram_compliant
    if prediction.stock_value == truth.stock_value:
        stock = {"stock_correct": 1}
    elif prediction.stock_value < truth.stock_value:
        stock = {"stock_below": 1}
    else:
        stock = {"stock_above": 1}
    return Tally(
        identification_correct=int(compliant), identification_wrong=int(not compliant), **stock
    )


def add(a: Tally, b: Tally) -> Tally:
    return Tally(**{f: getattr(a, f) + getattr(b, f) for f in a.__dataclass_fields__})


def frame_tallies(predictions: Submission, labels: Submission) -> list[Tally]:
    tallies = []
    for camera, frames in labels.items():
        for filename, areas in frames.items():
            shown = predictions.get(camera, {}).get(filename, {})
            frame = Tally()
            for area, truth in areas.items():
                frame = add(frame, score_area(shown.get(area), truth))
            tallies.append(frame)
    return tallies


def tally(predictions: Submission, labels: Submission) -> Tally:
    total = Tally()
    for frame in frame_tallies(predictions, labels):
        total = add(total, frame)
    return total


def estimate(
    predictions: Submission,
    labels: Submission,
    validation_areas: int = VALIDATION_AREAS,
    resamples: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Expected validation score: points per labelled area times the validation area count.

    The 90% interval resamples labelled frames, since areas in one frame share their errors.
    """
    frames = [frame for frame in frame_tallies(predictions, labels) if frame.areas]
    if not frames:
        raise ValueError("Cannot estimate a score from an empty label set")
    points = np.array([frame.points for frame in frames], dtype=float)
    areas = np.array([frame.areas for frame in frames], dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(frames), (resamples, len(frames)))
    resampled = points[picks].sum(1) / areas[picks].sum(1) * validation_areas
    total = tally(predictions, labels)
    per_area = total.points / total.areas
    low, high = np.percentile(resampled, [5, 95])
    return Estimate(total, per_area, per_area * validation_areas, low, high, validation_areas)


@app.command()
def main(
    predictions: Annotated[Path, typer.Argument()] = ROOT / "out" / "test" / "predictions.json",
    labels: Path = ROOT / "labels" / "dev_labels.json",
    validation_areas: int = VALIDATION_AREAS,
    split: Path | None = None,
) -> None:
    truth = SUBMISSION.validate_json(labels.read_bytes())
    if split:
        held = load_test_frames(split)
        truth = {
            camera: {f: areas for f, areas in frames.items() if f"{camera}/{f}" in held}
            for camera, frames in truth.items()
        }
    result = estimate(SUBMISSION.validate_json(predictions.read_bytes()), truth, validation_areas)
    t = result.tally
    rows = [
        ("Identification correct", t.identification_correct, 1),
        ("Identification wrong", t.identification_wrong, -2),
        ("Stock level correct", t.stock_correct, 1),
        ("Stock level below the real stock", t.stock_below, -2),
        ("Stock level above the real stock", t.stock_above, -3),
    ]
    typer.echo(f"{predictions} against {labels} ({t.areas} labelled areas, {t.missing} missing)")
    for name, count, each in rows:
        typer.echo(f"  {name:34} {count:5}  {each:+d}  {count * each:+6d}")
    typer.echo(f"  {'Total':34} {'':5}      {t.points:+6d}")
    typer.echo(f"{result.per_area:+.3f} per area")
    typer.echo(
        f"Validation estimate ({result.validation_areas} areas): {result.expected_total:+.0f}"
        f" (90% interval {result.low:+.0f} to {result.high:+.0f}),"
        f" maximum {2 * result.validation_areas:+d}"
    )


if __name__ == "__main__":
    app()
