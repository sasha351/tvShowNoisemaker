"""Self-update from ``origin/main`` at launch, and a label for the version that is running.

The macOS app launcher runs ``python -m tvshow_noisemaker.updater`` before the GUI.  It only
ever fast-forwards: on another branch, with uncommitted edits, with local commits, or offline it
leaves the checkout alone and records why.  The result goes to a small JSON file the GUI reads,
so the app can say what changed (or why it did not update) on the next launch.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from tvshow_noisemaker import __version__

REPO = Path(__file__).resolve().parent.parent
STATE = Path.home() / ".cache" / "tvshow_noisemaker" / "update.json"
BRANCH = "main"


@dataclass
class UpdateResult:
    status: str  # "updated" | "current" | "skipped" | "failed"
    reason: str = ""
    old: str = ""
    new: str = ""
    changes: List[str] = field(default_factory=list)
    at: float = field(default_factory=time.time)
    seen: bool = False  # set once the GUI has shown an "updated" result


def _git(repo: Path, *args: str, timeout: float = 10) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=timeout, check=True,
    ).stdout.strip()


def _is_ancestor(repo: Path, old: str, new: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", old, new], capture_output=True, timeout=10,
    ).returncode == 0


def update(repo: Path = REPO, branch: str = BRANCH, install_deps: bool = True) -> UpdateResult:
    """Fast-forward *repo* to ``origin/<branch>`` when that is safe; never merge or discard work."""
    try:
        if not (repo / ".git").exists():
            return UpdateResult("skipped", "not a git checkout")
        current = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if current != branch:
            return UpdateResult("skipped", f"on branch {current}")
        if _git(repo, "status", "--porcelain", "--untracked-files=no"):
            return UpdateResult("skipped", "uncommitted changes")
        _git(repo, "fetch", "--quiet", "origin", branch, timeout=20)
        old = _git(repo, "rev-parse", "HEAD")
        new = _git(repo, "rev-parse", f"origin/{branch}")
        if old == new:
            return UpdateResult("current", old=old, new=new)
        if not _is_ancestor(repo, old, new):
            return UpdateResult("skipped", "local commits not on GitHub", old=old, new=new)
        changes = _git(repo, "log", "--format=%s", f"{old}..{new}").splitlines()
        deps_changed = bool(_git(repo, "diff", "--name-only", old, new, "--", "pyproject.toml"))
        _git(repo, "merge", "--ff-only", "--quiet", new)
        if deps_changed and install_deps:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "-e", str(repo)],
                capture_output=True, timeout=600, check=False,
            )
        return UpdateResult("updated", old=old, new=new, changes=changes)
    except subprocess.TimeoutExpired:
        return UpdateResult("failed", "GitHub did not answer in time")
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        return UpdateResult("failed", detail.strip().splitlines()[-1] if detail.strip() else "git error")


def save(result: UpdateResult, path: Path = STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result)))


def load(path: Path = STATE) -> Optional[UpdateResult]:
    try:
        return UpdateResult(**json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        return None


def version_label(repo: Path = REPO) -> str:
    """``"2.0.0 · ed07ed4 · 2026-11-04"``; just the version outside a git checkout."""
    try:
        short, date = _git(repo, "log", "-1", "--format=%h %cs").split()
        return f"{__version__} · {short} · {date}"
    except (OSError, ValueError, subprocess.SubprocessError):
        return __version__


def main() -> None:
    result = update()
    previous = load()
    # Keep an unseen "updated" note when a later launch finds nothing new, so it is not lost.
    if result.status == "current" and previous and previous.status == "updated" and not previous.seen:
        return
    save(result)
    print(f"update: {result.status}{' – ' + result.reason if result.reason else ''}")


if __name__ == "__main__":
    main()
