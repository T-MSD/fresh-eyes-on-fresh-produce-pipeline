import hashlib
import subprocess

from src.provenance import git_sha, provenance


def git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def commit_repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    git(tmp_path, "add", "a.py")
    git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "commit",
        "-q",
        "-m",
        "init",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    return head.stdout.strip()


def test_git_sha_is_the_commit_on_a_clean_tree(tmp_path):
    head = commit_repo(tmp_path)
    assert git_sha(tmp_path) == head


def test_git_sha_marks_uncommitted_changes(tmp_path):
    head = commit_repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    assert git_sha(tmp_path) == f"{head}-dirty"


def test_git_sha_marks_untracked_files(tmp_path):
    head = commit_repo(tmp_path)
    (tmp_path / "b.py").write_text("y = 1\n")
    assert git_sha(tmp_path) == f"{head}-dirty"


def test_git_sha_outside_a_repository_is_none(tmp_path):
    assert git_sha(tmp_path) is None


def test_provenance_hashes_each_input_and_records_the_params(tmp_path):
    labels = tmp_path / "labels.json"
    labels.write_text("{}")

    record = provenance({"epochs": 8}, {"labels": labels, "split": None})

    assert record["params"] == {"epochs": 8}
    assert record["inputs"]["labels"] == {
        "path": str(labels),
        "sha256": hashlib.sha256(b"{}").hexdigest(),
    }
    assert record["inputs"]["split"] is None
    assert record["created_at"].endswith("Z")
    assert "git_sha" in record
