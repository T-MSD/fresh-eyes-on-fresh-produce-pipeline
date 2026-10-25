import json
from pathlib import Path

import pytest
from sqlalchemy import Engine, delete, select
from typer.testing import CliRunner

from cli import app
from config.db.schema import Task
from src.dataset.load import load_dataset
from src.dataset.read import read_dataset
from src.runs.save import save_run
from tests.builders import area, label, write_dataset, write_frame_bytes

NO_PATHS = {"PLANOGRAM_PATH": None, "IMAGES_DIR": None}


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    return write_dataset(
        tmp_path,
        {"6215": [area("6215_1"), area("6215_2")], "6383": [area("6383_1")]},
        {"6215": [1751052241]},
    )


def run(*args: str, env: dict[str, str | None] = NO_PATHS):
    return CliRunner().invoke(app, ["load", *args], env=env)


def test_dry_run_checks_the_dataset_from_arguments(paths):
    planogram_path, images_dir = paths

    result = run(str(planogram_path), str(images_dir), "--dry-run")

    assert result.exit_code == 0, result.output
    assert f"{planogram_path.resolve()} (argument)" in result.output
    assert f"{images_dir.resolve()} (argument)" in result.output
    assert "2 cameras, 1 products, 3 areas, 1 frames (0 unreadable)" in result.output
    assert "camera 6383 has no frames" in result.output
    assert "database not changed" in result.output


def test_lists_unreadable_frames(paths):
    planogram_path, images_dir = paths
    broken = write_frame_bytes(images_dir, "6383", 1751052241, b"not an image")

    result = run(str(planogram_path), str(images_dir), "--dry-run")

    assert result.exit_code == 0, result.output
    assert "2 frames (1 unreadable)" in result.output
    assert f"unreadable {broken.resolve()}" in result.output


def test_falls_back_to_environment_paths(paths):
    planogram_path, images_dir = paths

    result = run(
        "--dry-run", env={"PLANOGRAM_PATH": str(planogram_path), "IMAGES_DIR": str(images_dir)}
    )

    assert result.exit_code == 0, result.output
    assert f"{planogram_path.resolve()} (env)" in result.output
    assert f"{images_dir.resolve()} (env)" in result.output


def test_argument_wins_over_environment(paths, tmp_path: Path):
    planogram_path, images_dir = paths
    other = tmp_path / "other.json"
    other.write_bytes(planogram_path.read_bytes())

    result = run(
        str(other),
        "--dry-run",
        env={"PLANOGRAM_PATH": str(planogram_path), "IMAGES_DIR": str(images_dir)},
    )

    assert result.exit_code == 0, result.output
    assert f"{other.resolve()} (argument)" in result.output
    assert f"{images_dir.resolve()} (env)" in result.output


def test_fails_without_argument_or_environment():
    result = run("--dry-run")
    assert result.exit_code == 2
    assert "PLANOGRAM" in result.output


def test_fails_on_missing_images_folder(paths, tmp_path: Path):
    result = run(str(paths[0]), str(tmp_path / "missing"), "--dry-run")
    assert result.exit_code == 2


def test_reports_every_planogram_problem_and_exits(tmp_path: Path):
    planogram_path, images_dir = write_dataset(
        tmp_path, {"6215": [area("6215_0"), area("6383_1")]}, {"6215": [1751052241]}
    )

    result = run(str(planogram_path), str(images_dir), "--dry-run")

    assert result.exit_code == 1
    assert "6215_0" in result.output
    assert "6383_1" in result.output


def run_save(*args: str):
    return CliRunner().invoke(app, ["save-run", *args], env=NO_PATHS)


def test_save_run_reports_an_invalid_stock_level_without_a_traceback(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text(
        json.dumps(
            {"6215": {"1.jpg": {"6215_1": {"planogram_compliant": True, "stock_level": "x"}}}}
        )
    )

    result = run_save(str(predictions))

    assert result.exit_code == 1
    assert "6215/1.jpg/6215_1/stock_level" in result.output
    assert "Traceback" not in result.output


def test_save_run_reports_a_file_that_is_not_json(tmp_path: Path):
    predictions = tmp_path / "predictions.json"
    predictions.write_text("{")

    result = run_save(str(predictions))

    assert result.exit_code == 1
    assert "not valid JSON" in result.output


def run_derive_tasks(engine: Engine, run_id: int):
    env = {**NO_PATHS, "DATABASE_URL": engine.url.render_as_string(hide_password=False)}
    return CliRunner().invoke(app, ["derive-tasks", str(run_id)], env=env)


def commit_run(engine: Engine, root: Path, *, with_tasks: bool) -> int:
    with engine.begin() as db:
        dataset = read_dataset(
            *write_dataset(root, {"6215": [area("6215_1")]}, {"6215": [1751052241]})
        )
        dataset_id = load_dataset(db, dataset).dataset_id
        predictions = {"6215": {"1751052241.jpg": {"6215_1": label("empty")}}}
        run_id = save_run(db, dataset_id, predictions, method="run_test")
        if not with_tasks:
            db.execute(delete(Task).where(Task.run_id == run_id))
    return run_id


def test_derive_tasks_stores_the_tasks_of_a_run_without_any(committed: Engine, tmp_path: Path):
    run_id = commit_run(committed, tmp_path, with_tasks=False)

    result = run_derive_tasks(committed, run_id)

    assert result.exit_code == 0, result.output
    assert f"stored 1 tasks for run {run_id}" in result.output
    with committed.connect() as db:
        assert db.scalars(select(Task.kind).where(Task.run_id == run_id)).all() == ["restock"]


def test_derive_tasks_refuses_a_run_that_already_has_tasks(committed: Engine, tmp_path: Path):
    run_id = commit_run(committed, tmp_path, with_tasks=True)

    result = run_derive_tasks(committed, run_id)

    assert result.exit_code == 1
    assert f"run {run_id} already has tasks" in result.output
    assert "Traceback" not in result.output


def test_derive_tasks_refuses_an_unknown_run(committed: Engine):
    result = run_derive_tasks(committed, 999999)

    assert result.exit_code == 1
    assert "run 999999 does not exist" in result.output
    assert "Traceback" not in result.output
