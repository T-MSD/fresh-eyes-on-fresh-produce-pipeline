import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from config.db.connection import make_engine
from src.dataset.errors import InvalidInput
from src.dataset.load import load_dataset
from src.dataset.read import Dataset, read_dataset
from src.prediction.submission import SUBMISSION, fallback_keys
from src.runs.errors import UnusableRun
from src.runs.save import latest_dataset_id, save_run
from src.tasks.derive import derive_tasks

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Fresh eyes on fresh produce: pipeline commands."""


@app.command()
def load(
    ctx: typer.Context,
    planogram: Annotated[
        Path,
        typer.Argument(
            envvar="PLANOGRAM_PATH",
            metavar="PLANOGRAM",
            show_envvar=True,
            exists=True,
            dir_okay=False,
            readable=True,
            help="Planogram JSON.",
        ),
    ],
    images_dir: Annotated[
        Path,
        typer.Argument(
            envvar="IMAGES_DIR",
            metavar="IMAGES_DIR",
            show_envvar=True,
            exists=True,
            file_okay=False,
            help="Folder of <camera>/<unix_ts>.jpg frames.",
        ),
    ],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Check the inputs; leave the database alone.")
    ] = False,
) -> None:
    """Add this planogram and its frames to the database as the newest dataset load."""
    for name, label, path in (
        ("planogram", "planogram", planogram),
        ("images_dir", "images", images_dir),
    ):
        source = ctx.get_parameter_source(name)
        origin = "env" if source is not None and source.name == "ENVIRONMENT" else "argument"
        typer.echo(f"{label:<10} {path.resolve()} ({origin})")

    try:
        dataset = read_dataset(planogram, images_dir)
    except InvalidInput as error:
        fail(error.problems)
    print_dataset(dataset)

    if dry_run:
        typer.echo("dry run: database not changed")
        return

    engine = make_engine()
    try:
        with engine.begin() as conn:
            report = load_dataset(conn, dataset)
    finally:
        engine.dispose()
    typer.echo(f"loaded at {report.loaded_at.isoformat()}")


@app.command("save-run")
def save_run_command(
    predictions: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help="Submission-shaped predictions JSON."
        ),
    ],
    method: Annotated[str, typer.Option(help="Name stored with the run.")] = "file",
    dataset_id: Annotated[
        int | None, typer.Option(help="Dataset load to attach to; default is the newest.")
    ] = None,
    fallbacks: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            readable=True,
            help="fallbacks.json from crop_areas: its areas are stored with is_fallback set.",
        ),
    ] = None,
) -> None:
    """Store a predictions file as a run on a loaded dataset."""
    try:
        raw = json.loads(predictions.read_text())
        SUBMISSION.validate_python(raw)
        fallback = fallback_keys(fallbacks) if fallbacks else set()
    except json.JSONDecodeError as error:
        fail([f"not valid JSON: {error}"])
    except ValidationError as error:
        fail([f"{'/'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()[:10]])
    engine = make_engine()
    try:
        with engine.begin() as conn:
            target = dataset_id if dataset_id is not None else latest_dataset_id(conn)
            run_id = save_run(
                conn,
                target,
                raw,
                method=method,
                params={"predictions": str(predictions)},
                fallback=fallback,
            )
    except UnusableRun as error:
        fail(error.problems)
    finally:
        engine.dispose()
    typer.echo(f"stored run {run_id} on dataset {target}")


@app.command("derive-tasks")
def derive_tasks_command(
    run_id: Annotated[int, typer.Argument(help="Run stored without tasks.")],
) -> None:
    """Open and close the tasks of a run stored before save-run derived them."""
    engine = make_engine()
    try:
        with engine.begin() as conn:
            count = derive_tasks(conn, run_id)
    except UnusableRun as error:
        fail(error.problems)
    finally:
        engine.dispose()
    typer.echo(f"stored {count} tasks for run {run_id}")


def print_dataset(dataset: Dataset) -> None:
    planogram = dataset.planogram
    unreadable = [frame for frame in dataset.frames if frame.size is None]
    typer.echo(
        f"store      {planogram.store}: {len(planogram.cameras)} cameras,"
        f" {len(planogram.products)} products, {len(planogram.areas)} areas,"
        f" {len(dataset.frames)} frames ({len(unreadable)} unreadable)"
    )
    for frame in unreadable:
        typer.echo(f"unreadable {dataset.images_dir / frame.camera / frame.filename}")
    for path in dataset.skipped:
        typer.echo(f"skipped    {path}")
    for camera in dataset.cameras_without_frames:
        typer.echo(f"warning    camera {camera} has no frames")


def fail(problems: tuple[str, ...] | list[str]) -> NoReturn:
    for problem in problems:
        typer.echo(f"error      {problem}", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
