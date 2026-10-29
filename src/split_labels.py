"""Split the hand labels in time: per camera, the latest frames become the test set.

The split is frozen: once labels/split.json exists it is only rewritten with --force, so every
score is measured on the same test frames. Test frames are scored, never trained or tuned on, and
no frame from a camera's first test frame onwards is trained on, labelled or not.

    uv run --env-file .env python -m src.split_labels
"""

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import typer
from pydantic import BaseModel, TypeAdapter

from src.prediction_io import MANUAL_LABELS

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "labels" / "split.json"
TEST_SHARE = 0.2
app = typer.Typer(add_completion=False)


class Split(BaseModel):
    test_frames: list[str]


SPLIT_FILE = TypeAdapter(Split)


def frame_of(key: str) -> str:
    return key.rsplit("/", 1)[0]


def held_out_frames(label_keys: Iterable[str], share: float = TEST_SHARE) -> set[str]:
    """Frames (`camera/<unix_ts>.jpg`) held out for testing: each camera's latest `share` of labelled frames."""
    frames: dict[str, set[str]] = defaultdict(set)
    for key in label_keys:
        frame = frame_of(key)
        frames[frame.split("/", 1)[0]].add(frame)
    held: set[str] = set()
    for camera_frames in frames.values():
        ordered = sorted(camera_frames, key=timestamp_of)
        held.update(ordered[len(ordered) - round(len(ordered) * share) :])
    return held


def timestamp_of(frame: str) -> int:
    return int(frame.split("/", 1)[1].removesuffix(".jpg"))


def held_out_window(test_frames: set[str], frames: Iterable[str]) -> set[str]:
    """Every frame at or after its camera's first test frame, so no later frame leaks into training."""
    start: dict[str, int] = {}
    for frame in test_frames:
        camera = frame.split("/", 1)[0]
        start[camera] = min(start.get(camera, timestamp_of(frame)), timestamp_of(frame))
    return {
        f
        for f in frames
        if f.split("/", 1)[0] in start and timestamp_of(f) >= start[f.split("/", 1)[0]]
    }


def load_test_frames(path: Path | None) -> set[str]:
    return set(SPLIT_FILE.validate_json(path.read_bytes()).test_frames) if path else set()


@app.command()
def main(
    labels: Path = ROOT / "out" / "labels.json", out: Path = SPLIT, force: bool = False
) -> None:
    if out.exists() and not force:
        raise typer.BadParameter(
            f"{out} is frozen: scores are only comparable on the same test frames. "
            "Pass --force to replace it"
        )
    held = held_out_frames(MANUAL_LABELS.validate_json(labels.read_bytes()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"test_frames": sorted(held)}, indent=1))
    typer.echo(f"wrote {out}: {len(held)} test frames")


if __name__ == "__main__":
    app()
