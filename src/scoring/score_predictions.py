from pathlib import Path

import typer

from src.prediction.submission import SUBMISSION
from src.scoring.score import MANUAL_LABELS, score_submission

ROOT = Path(__file__).resolve().parent.parent.parent
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
    old, new = score_submission(baseline, manual), score_submission(candidate, manual)
    typer.echo(f"labels: {labels}")
    typer.echo(f"before: {old:+.6f} points per area ({old * len(manual):+.0f} total)")
    typer.echo(f"after:  {new:+.6f} points per area ({new * len(manual):+.0f} total)")
    typer.echo(f"change: {new - old:+.6f} points per area")
    typer.echo(
        "These labels train strategy A; this is a training-label score, not held-out accuracy."
    )


if __name__ == "__main__":
    app()
