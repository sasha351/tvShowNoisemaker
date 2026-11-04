"""Updates for the packaged desktop apps, from GitHub Releases.

Each release (built by ``.github/workflows/macos-app.yml``) carries one installer per platform:

* macOS: ``TV-Show-Noisemaker-macOS-<arm64|x86_64>.dmg``, a drag-to-Applications disk image.
* Windows: ``TV-Show-Noisemaker-Windows-x64-Setup.exe``, a per-user installer (no admin needed).

The packaged app asks the Releases API for the newest ``vX.Y.Z`` release and downloads the
installer for this machine, checking its size and SHA-256.  On macOS it copies the app out of the
disk image and hands off to a tiny shell script that waits for the app to quit, swaps the bundle
in place and reopens it.  On Windows it runs the installer silently, which closes the app,
upgrades it in place and starts it again.  Source checkouts keep using
:mod:`tvshow_noisemaker.updater`.
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
_SLUG = APP_NAME.replace(" ", "-")
_RELEASE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def asset_name(system: Optional[str] = None, arch: Optional[str] = None) -> str:
    """The installer this machine downloads from a release."""
    system = system or sys.platform
    if system == "win32":
        return f"{_SLUG}-Windows-x64-Setup.exe"
    return f"{_SLUG}-macOS-{arch or platform.machine()}.dmg"


def is_bundled() -> bool:
    """True when running from the PyInstaller-built app rather than a Python install."""
    return bool(getattr(sys, "frozen", False))


def bundle_path() -> Optional[Path]:
    """The running macOS ``.app`` directory, or None outside one."""
    if not is_bundled() or sys.platform != "darwin":
        return None
    app = Path(sys.executable).resolve().parents[2]  # <App>.app/Contents/MacOS/<exe>
    return app if app.suffix == ".app" else None


def extend_path() -> None:
    """Make ffmpeg/ffprobe/fpcalc findable however the app was launched.

    Finder gives apps a bare PATH (``/usr/bin:/bin:...``), so Homebrew's tools are invisible:
    Homebrew comes first (its ffmpeg has Apple's AAC encoder).  The copies bundled in the app
    come last, as the fallback on every platform.
    """
    parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    front = []
    if sys.platform == "darwin":
        front = [p for p in ("/opt/homebrew/bin", "/usr/local/bin") if p not in parts]
        parts = parts or ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    back = []
    bundled = Path(getattr(sys, "_MEIPASS", "")) / "bin"
    if is_bundled() and bundled.is_dir() and str(bundled) not in parts:
        back.append(str(bundled))
    os.environ["PATH"] = os.pathsep.join(front + parts + back)


def hide_child_consoles() -> None:
    """Windows GUI apps flash a console window for every console child (ffmpeg, fpcalc...).

    Every tool runs through :mod:`subprocess`, so default ``CREATE_NO_WINDOW`` on there once
    instead of at each of the many call sites.
    """
    if sys.platform != "win32" or getattr(subprocess.Popen, "_tvsn_no_window", False):
        return
    original = subprocess.Popen.__init__

    def __init__(self, *args, **kwargs):
        if not kwargs.get("creationflags"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        original(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__  # type: ignore[method-assign]
    subprocess.Popen._tvsn_no_window = True  # type: ignore[attr-defined]


def parse_version(text: str) -> Optional[Tuple[int, int, int]]:
    m = _RELEASE.match(text.strip())
    return tuple(int(n) for n in m.groups()) if m else None  # type: ignore[return-value]


@dataclass
class Release:
    tag: str
    version: str
    url: str  # the installer for this machine
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


def pick_release(data: dict, current: str = __version__, asset: Optional[str] = None) -> Optional[Release]:
    """The release in the API's ``releases/latest`` payload, if it is newer and has our installer."""
    tag = data.get("tag_name", "")
    newest, running = parse_version(tag), parse_version(current)
    if newest is None or running is None or newest <= running or data.get("draft") or data.get("prerelease"):
        return None
    want = asset or asset_name()
    for item in data.get("assets", []):
        if item.get("name") == want:
            digest = item.get("digest") or ""
            return Release(
                tag=tag, version=".".join(map(str, newest)), url=item["browser_download_url"],
                size=int(item.get("size") or 0),
                sha256=digest.split(":", 1)[1] if digest.startswith("sha256:") else "",
                notes=data.get("body") or "", page=data.get("html_url", ""),
            )
    return None


def check(timeout: float = 10) -> Optional[Release]:
    """The newer release for this app, or None when up to date.  Raises on network errors."""
    with _open(f"https://api.github.com/repos/{REPO_SLUG}/releases/latest", timeout) as resp:
        return pick_release(json.load(resp))


def download(release: Release, dest: Path, progress: Callable[[float], None] = lambda f: None) -> Path:
    """Fetch the release installer into *dest*, verifying size and SHA-256; return its path."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / asset_name()
    digest, done = hashlib.sha256(), 0
    with _open(release.url, 60) as resp, open(path, "wb") as out:
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
    return path


def unpack(dmg: Path, dest: Path) -> Path:
    """Copy the app out of the release disk image into *dest*; return the copied .app."""
    mount = Path(tempfile.mkdtemp(prefix="tvsn-dmg-"))
    subprocess.run(
        ["hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", str(mount), str(dmg)],
        check=True, capture_output=True,
    )
    try:
        src = mount / f"{APP_NAME}.app"
        if not (src / "Contents" / "MacOS").is_dir():
            raise OSError("the disk image does not contain the app")
        dest.mkdir(parents=True, exist_ok=True)
        app = dest / src.name
        subprocess.run(["ditto", str(src), str(app)], check=True, capture_output=True)  # keeps signatures
        return app
    finally:
        subprocess.run(["hdiutil", "detach", "-force", str(mount)], capture_output=True)
        shutil.rmtree(mount, ignore_errors=True)


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

# Inno Setup flags: no wizard pages, no message boxes, close the running app, never reboot.
# The installer's [Run] entry starts the app again once it is done.
WINDOWS_INSTALLER_ARGS = ["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"]


def install(release: Release, progress: Callable[[float], None] = lambda f: None) -> None:
    """Download *release* and start replacing this app with it.  The caller must quit right after."""
    if not is_bundled():
        raise OSError("not running from the packaged app")
    work = Path(tempfile.mkdtemp(prefix="tvsn-update-"))
    try:
        if sys.platform == "win32":
            setup = download(release, work, progress)
            subprocess.Popen(
                [str(setup), *WINDOWS_INSTALLER_ARGS], close_fds=True,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            return  # the installer lives in %TEMP%; Windows cleans that up
        app = bundle_path()
        if app is None:
            raise OSError("not running from the app bundle")
        if not os.access(app.parent, os.W_OK):
            raise OSError(f"no permission to replace the app in {app.parent}")
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
