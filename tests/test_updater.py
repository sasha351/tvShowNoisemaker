import subprocess
from pathlib import Path

import pytest

from tvshow_noisemaker import updater


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True,
    ).stdout.strip()


def commit(repo: Path, name: str, text: str, msg: str) -> None:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg)


@pytest.fixture
def clones(tmp_path):
    """(origin working copy that 'pushes', local clone the updater runs in)."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    upstream = tmp_path / "upstream"
    subprocess.run(["git", "clone", "-q", str(origin), str(upstream)], check=True)
    git(upstream, "checkout", "-q", "-b", "main")
    commit(upstream, "a.txt", "1", "first")
    git(upstream, "push", "-q", "origin", "main")
    local = tmp_path / "local"
    subprocess.run(["git", "clone", "-q", str(origin), str(local)], check=True)
    return upstream, local


def push(upstream: Path, msg: str, name: str = "a.txt") -> None:
    commit(upstream, name, msg, msg)
    git(upstream, "push", "-q", "origin", "main")


def test_fast_forwards_and_lists_changes(clones):
    upstream, local = clones
    push(upstream, "second")
    push(upstream, "third")
    result = updater.update(local, install_deps=False)
    assert result.status == "updated"
    assert result.changes == ["third", "second"]
    assert git(local, "rev-parse", "HEAD") == git(upstream, "rev-parse", "HEAD")


def test_current_when_nothing_new(clones):
    _, local = clones
    assert updater.update(local, install_deps=False).status == "current"


def test_skips_uncommitted_changes(clones):
    upstream, local = clones
    push(upstream, "second")
    (local / "a.txt").write_text("edited")
    result = updater.update(local, install_deps=False)
    assert (result.status, result.reason) == ("skipped", "uncommitted changes")
    assert (local / "a.txt").read_text() == "edited"


def test_skips_other_branch(clones):
    _, local = clones
    git(local, "checkout", "-q", "-b", "feature")
    assert updater.update(local, install_deps=False).reason == "on branch feature"


def test_skips_diverged_history(clones):
    upstream, local = clones
    push(upstream, "remote work")
    commit(local, "b.txt", "x", "local work")
    before = git(local, "rev-parse", "HEAD")
    result = updater.update(local, install_deps=False)
    assert (result.status, result.reason) == ("skipped", "local commits not on GitHub")
    assert git(local, "rev-parse", "HEAD") == before


def test_not_a_checkout(tmp_path):
    assert updater.update(tmp_path, install_deps=False).status == "skipped"


def test_state_round_trip(tmp_path):
    path = tmp_path / "s" / "update.json"
    updater.save(updater.UpdateResult("updated", old="a" * 40, new="b" * 40, changes=["x"]), path)
    loaded = updater.load(path)
    assert loaded.status == "updated" and loaded.changes == ["x"] and loaded.seen is False
    assert updater.load(tmp_path / "missing.json") is None


def test_version_label_has_commit(tmp_path, clones):
    _, local = clones
    label = updater.version_label(local)
    assert label.startswith(updater.__version__ + " · ")
    assert updater.version_label(tmp_path) == updater.__version__
