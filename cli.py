from pathlib import Path
from typing import Annotated, NoReturn

import typer

from config.db.connection import make_engine
from config.db.load import load_dataset
from ingest.dataset import Dataset, read_dataset
from ingest.errors import InvalidInput

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
