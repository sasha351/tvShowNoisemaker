"""Self-update to the newest tagged release at launch, and a label for the version that is running.

Releases are ``vX.Y.Z`` tags on ``main`` (cut with ``tools/release.sh``).  The macOS app launcher
runs ``python -m tvshow_noisemaker.updater`` before the GUI, which fast-forwards ``main`` to the
newest release tag.  Commits pushed to ``main`` without a tag are never picked up.  It only ever
fast-forwards: on another branch, with uncommitted edits, with local commits, or offline it
leaves the checkout alone and records why.  The result goes to a small JSON file the GUI reads,
so the app can say what changed (or why it did not update) on the next launch.
"""
from __future__ import annotations

import json
import re
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
_RELEASE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


@dataclass
class UpdateResult:
    status: str  # "updated" | "current" | "skipped" | "failed"
    reason: str = ""
    old: str = ""
    new: str = ""
    changes: List[str] = field(default_factory=list)
    tag: str = ""  # the release this result is about, e.g. "v2.1.0"
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


def latest_release(repo: Path) -> Optional[str]:
    """The highest ``vX.Y.Z`` tag, by number (so v2.10.0 beats v2.9.0)."""
    tags = [t for t in _git(repo, "tag", "--list", "v[0-9]*").splitlines() if _RELEASE.match(t)]
    return max(tags, key=lambda t: tuple(int(n) for n in _RELEASE.match(t).groups()), default=None)


def update(repo: Path = REPO, branch: str = BRANCH, install_deps: bool = True) -> UpdateResult:
    """Fast-forward *branch* to the newest release tag when that is safe; never merge or discard work."""
    try:
        if not (repo / ".git").exists():
            return UpdateResult("skipped", "not a git checkout")
        current = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
        if current != branch:
            return UpdateResult("skipped", f"on branch {current}")
        if _git(repo, "status", "--porcelain", "--untracked-files=no"):
            return UpdateResult("skipped", "uncommitted changes")
        _git(repo, "fetch", "--quiet", "--tags", "--force", "origin", timeout=20)
        tag = latest_release(repo)
        if tag is None:
            return UpdateResult("skipped", "no releases yet")
        old = _git(repo, "rev-parse", "HEAD")
        new = _git(repo, "rev-parse", f"{tag}^{{commit}}")
        if old == new or _is_ancestor(repo, new, old):  # at or past the release already
            return UpdateResult("current", old=old, new=new, tag=tag)
        if not _is_ancestor(repo, old, new):
            return UpdateResult("skipped", "local commits not in the release", old=old, new=new, tag=tag)
        changes = _git(repo, "log", "--format=%s", f"{old}..{new}").splitlines()
        deps_changed = bool(_git(repo, "diff", "--name-only", old, new, "--", "pyproject.toml"))
        _git(repo, "merge", "--ff-only", "--quiet", new)
        if deps_changed and install_deps:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "-e", str(repo)],
                capture_output=True, timeout=600, check=False,
            )
        return UpdateResult("updated", old=old, new=new, changes=changes, tag=tag)
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
    """What is running: ``"2.1.0 · 2026-11-04"`` on a release, ``"2.1.0 + 3 · abc1234"`` past one.

    Falls back to the package version (plus the commit, inside a checkout with no release tags).
    """
    try:
        date = _git(repo, "log", "-1", "--format=%cs")
        short = _git(repo, "rev-parse", "--short", "HEAD")
    except (OSError, subprocess.SubprocessError):
        return __version__
    try:
        described = _git(repo, "describe", "--tags", "--match", "v[0-9]*")
    except (OSError, subprocess.SubprocessError):
        return f"{__version__} · {short} · {date}"
    m = re.match(r"^v(\d+\.\d+\.\d+)(?:-(\d+)-g[0-9a-f]+)?$", described)
    if not m:
        return f"{__version__} · {short} · {date}"
    version, ahead = m.groups()
    return f"{version} + {ahead} · {short}" if ahead else f"{version} · {date}"


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
