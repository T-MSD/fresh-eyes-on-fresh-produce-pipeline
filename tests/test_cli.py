from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app
from tests.builders import area, write_dataset, write_frame_bytes

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
