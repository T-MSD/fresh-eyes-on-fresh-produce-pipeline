from pathlib import Path

import typer

from src.estimate_score import tally
from src.prediction_io import MANUAL_LABELS, SUBMISSION, Submission

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
app = typer.Typer(add_completion=False)


@app.command()
def main(
    labels: Path = OUT / "labels.json",
    before: Path = OUT / "final_predictions.before_hybrid.json",
    after: Path = OUT / "final_predictions.json",
) -> None:
    manual = MANUAL_LABELS.validate_json(labels.read_bytes())
    baseline = SUBMISSION.validate_json(before.read_bytes())
    candidate = SUBMISSION.validate_json(after.read_bytes())
    truth: Submission = {}
    for key, label in manual.items():
        camera, filename, area = key.split("/")
        truth.setdefault(camera, {}).setdefault(filename, {})[area] = label
    old, new = (t.points / t.areas for t in (tally(baseline, truth), tally(candidate, truth)))
    typer.echo(f"labels: {labels}")
    typer.echo(f"before: {old:+.6f} points per area ({old * len(manual):+.0f} total)")
    typer.echo(f"after:  {new:+.6f} points per area ({new * len(manual):+.0f} total)")
    typer.echo(f"change: {new - old:+.6f} points per area")
    typer.echo(
        "These labels train strategy A; this is a training-label score, not held-out accuracy."
    )


if __name__ == "__main__":
    app()
