import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_sha(root: Path = ROOT) -> str | None:
    """The HEAD commit, suffixed `-dirty` when tracked changes or untracked files are present."""
    try:
        head = git(root, "rev-parse", "HEAD")
        dirty = git(root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return None
    if not head:
        return None
    return f"{head}-dirty" if dirty else head


def repo_path(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(path)


def provenance(params: dict, inputs: dict[str, Path | None]) -> dict:
    """What produced a model or a score: time, git commit, params and the sha256 of each input file."""
    return {
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "params": params,
        "inputs": {
            name: None
            if path is None
            else {"path": repo_path(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in inputs.items()
        },
    }
