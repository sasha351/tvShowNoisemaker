"""Updates for the packaged macOS app, from GitHub Releases.

The downloadable ``TV Show Noisemaker.app`` (built by ``.github/workflows/macos-app.yml``) has no
git checkout to fast-forward, so it updates the way most Mac apps distributed on GitHub do: ask
the Releases API for the newest ``vX.Y.Z`` release, download its zip for this CPU, check its
size and SHA-256, then hand off to a tiny shell script that waits for the app to quit, swaps the
bundle in place and reopens it.  Source checkouts keep using :mod:`tvshow_noisemaker.updater`.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from tvshow_noisemaker import __version__

REPO_SLUG = os.environ.get("TVSN_UPDATE_REPO", "sasha351/tvShowNoisemaker")
APP_NAME = "TV Show Noisemaker"
_RELEASE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def asset_name(arch: Optional[str] = None) -> str:
    """The release zip the updater installs, e.g. ``TV-Show-Noisemaker-macOS-arm64.zip``."""
    return f"{APP_NAME.replace(' ', '-')}-macOS-{arch or platform.machine()}.zip"


def is_bundled() -> bool:
    """True when running from the PyInstaller-built .app rather than a Python install."""
    return bool(getattr(sys, "frozen", False))


def bundle_path() -> Optional[Path]:
    """The running ``.app`` directory, or None outside a bundle."""
    if not is_bundled():
        return None
    app = Path(sys.executable).resolve().parents[2]  # <App>.app/Contents/MacOS/<exe>
    return app if app.suffix == ".app" else None


def extend_path() -> None:
    """Make ffmpeg/ffprobe/fpcalc findable when launched from Finder.

    Finder gives apps a bare PATH (``/usr/bin:/bin:...``), so Homebrew's tools are invisible.
    Homebrew comes first (its ffmpeg has Apple's AAC encoder); the copies bundled in the app last.
    """
    parts = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin").split(os.pathsep)
    front = [p for p in ("/opt/homebrew/bin", "/usr/local/bin") if p not in parts]
    back = []
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "bin"
    if is_bundled() and bundled.is_dir() and str(bundled) not in parts:
        back.append(str(bundled))
    os.environ["PATH"] = os.pathsep.join(front + parts + back)


def parse_version(text: str) -> Optional[Tuple[int, int, int]]:
    m = _RELEASE.match(text.strip())
    return tuple(int(n) for n in m.groups()) if m else None  # type: ignore[return-value]


@dataclass
class Release:
    tag: str
    version: str
    url: str  # the zip for this machine
    size: int
    sha256: str  # "" when GitHub did not report a digest
    notes: str
    page: str


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # bundled with the app; python.org builds ship no CA store of their own
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _open(url: str, timeout: float):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"tvshow-noisemaker/{__version__}",
    })
    return urllib.request.urlopen(req, timeout=timeout, context=_ssl_context())


def pick_release(data: dict, current: str = __version__, arch: Optional[str] = None) -> Optional[Release]:
    """The release in the API's ``releases/latest`` payload, if it is newer and has our zip."""
    tag = data.get("tag_name", "")
    newest, running = parse_version(tag), parse_version(current)
    if newest is None or running is None or newest <= running or data.get("draft") or data.get("prerelease"):
        return None
    want = asset_name(arch)
    for asset in data.get("assets", []):
        if asset.get("name") == want:
            digest = asset.get("digest") or ""
            return Release(
                tag=tag, version=".".join(map(str, newest)), url=asset["browser_download_url"],
                size=int(asset.get("size") or 0),
                sha256=digest.split(":", 1)[1] if digest.startswith("sha256:") else "",
                notes=data.get("body") or "", page=data.get("html_url", ""),
            )
    return None


def check(timeout: float = 10) -> Optional[Release]:
    """The newer release for this app, or None when up to date.  Raises on network errors."""
    with _open(f"https://api.github.com/repos/{REPO_SLUG}/releases/latest", timeout) as resp:
        return pick_release(json.load(resp))


def download(release: Release, dest: Path, progress: Callable[[float], None] = lambda f: None) -> Path:
    """Fetch the release zip into *dest*, verifying size and SHA-256; return the zip path."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / asset_name()
    digest, done = hashlib.sha256(), 0
    with _open(release.url, 60) as resp, open(zip_path, "wb") as out:
        total = release.size or int(resp.headers.get("Content-Length") or 0)
        while chunk := resp.read(1 << 16):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total:
                progress(done / total)
    if release.size and done != release.size:
        raise OSError(f"download incomplete ({done} of {release.size} bytes)")
    if release.sha256 and digest.hexdigest() != release.sha256:
        raise OSError("download does not match the release checksum")
    return zip_path


def unpack(zip_path: Path, dest: Path) -> Path:
    """Expand the zip with ``ditto`` (keeps the bundle's symlinks and signatures); return the .app."""
    subprocess.run(["ditto", "-x", "-k", str(zip_path), str(dest)], check=True, capture_output=True)
    app = dest / f"{APP_NAME}.app"
    if not (app / "Contents" / "MacOS").is_dir():
        raise OSError("the release zip does not contain the app")
    return app


# Waits for the running app to quit, swaps the new bundle in (rolling back if the move fails),
# clears the quarantine flag and reopens the app.  Runs detached so it outlives the app.
SWAP_SCRIPT = r"""#!/bin/bash
pid="$1"; new="$2"; app="$3"; work="$4"
for _ in $(seq 1 600); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
old="$work/previous.app"
if mv "$app" "$old"; then
    if ! mv "$new" "$app"; then mv "$old" "$app"; fi
fi
xattr -dr com.apple.quarantine "$app" 2>/dev/null
open "$app"
rm -rf "$work"
"""


def install(release: Release, progress: Callable[[float], None] = lambda f: None) -> None:
    """Download and stage *release*, then start the swap script.  The caller must quit right after."""
    app = bundle_path()
    if app is None:
        raise OSError("not running from the app bundle")
    if not os.access(app.parent, os.W_OK):
        raise OSError(f"no permission to replace the app in {app.parent}")
    work = Path(tempfile.mkdtemp(prefix="tvsn-update-"))
    try:
        new = unpack(download(release, work, progress), work / "new")
        script = work / "swap.sh"
        script.write_text(SWAP_SCRIPT)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    with open(work / "swap.log", "w") as log:
        subprocess.Popen(
            ["/bin/bash", str(script), str(os.getpid()), str(new), str(app), str(work)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
        )
