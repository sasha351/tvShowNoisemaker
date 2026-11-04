import hashlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tvshow_noisemaker import appupdate


def payload(tag="v2.2.0", arch="arm64", **asset):
    return {
        "tag_name": tag, "draft": False, "prerelease": False, "body": "- notes",
        "html_url": f"https://github.com/x/y/releases/tag/{tag}",
        "assets": [
            {"name": "TV-Show-Noisemaker-macOS-arm64.dmg", "browser_download_url": "https://dl/dmg", "size": 9},
            {"name": appupdate.asset_name(arch), "browser_download_url": "https://dl/zip", "size": 5, **asset},
        ],
    }


def test_parse_version():
    assert appupdate.parse_version("v2.10.0") == (2, 10, 0)
    assert appupdate.parse_version("2.1.0") == (2, 1, 0)
    assert appupdate.parse_version("v2.1.0-beta") is None


def test_picks_newer_release_zip_for_this_cpu():
    rel = appupdate.pick_release(payload(digest="sha256:" + "ab" * 32), current="2.1.0", arch="arm64")
    assert (rel.tag, rel.version, rel.url, rel.size) == ("v2.2.0", "2.2.0", "https://dl/zip", 5)
    assert rel.sha256 == "ab" * 32


def test_numeric_order_and_no_downgrade():
    assert appupdate.pick_release(payload("v2.10.0"), current="2.9.0", arch="arm64") is not None
    assert appupdate.pick_release(payload("v2.1.0"), current="2.1.0", arch="arm64") is None
    assert appupdate.pick_release(payload("v2.0.0"), current="2.1.0", arch="arm64") is None


def test_ignores_release_without_our_zip_and_prereleases():
    assert appupdate.pick_release(payload(arch="x86_64"), current="2.1.0", arch="arm64") is None
    pre = payload()
    pre["prerelease"] = True
    assert appupdate.pick_release(pre, current="2.1.0", arch="arm64") is None
    assert appupdate.pick_release({"tag_name": "nightly"}, current="2.1.0") is None


class FakeResp(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def release(data: bytes, sha: str = "") -> appupdate.Release:
    return appupdate.Release("v9.0.0", "9.0.0", "https://dl/zip", len(data), sha, "", "")


def test_download_verifies_checksum(tmp_path, monkeypatch):
    data = b"zip bytes"
    monkeypatch.setattr(appupdate, "_open", lambda url, timeout: FakeResp(data))
    seen = []
    path = appupdate.download(release(data, hashlib.sha256(data).hexdigest()), tmp_path, seen.append)
    assert path.read_bytes() == data and seen[-1] == 1.0
    with pytest.raises(OSError, match="checksum"):
        appupdate.download(release(data, "0" * 64), tmp_path)


def test_download_rejects_truncated_file(tmp_path, monkeypatch):
    monkeypatch.setattr(appupdate, "_open", lambda url, timeout: FakeResp(b"short"))
    rel = release(b"short")
    rel.size = 100
    with pytest.raises(OSError, match="incomplete"):
        appupdate.download(rel, tmp_path)


def test_not_bundled_outside_the_app():
    assert appupdate.is_bundled() is False and appupdate.bundle_path() is None
    with pytest.raises(OSError, match="not running from the app"):
        appupdate.install(release(b""))


def test_extend_path_adds_homebrew_first(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    appupdate.extend_path()
    assert os.environ["PATH"].split(os.pathsep)[:3] == ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]


def bundle(path: Path, marker: str) -> Path:
    (path / "Contents" / "MacOS").mkdir(parents=True)
    (path / "Contents" / "marker").write_text(marker)
    return path


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_swap_script_replaces_bundle_after_app_quits(tmp_path):
    shims = tmp_path / "shims"  # stand-ins for macOS's open and xattr
    shims.mkdir()
    for tool in ("open", "xattr"):
        (shims / tool).write_text(f'#!/bin/bash\necho "{tool} $*" >> "{tmp_path}/calls"\n')
        (shims / tool).chmod(0o755)
    app = bundle(tmp_path / "Applications" / "TV Show Noisemaker.app", "old")
    work = tmp_path / "work"
    new = bundle(work / "new" / "TV Show Noisemaker.app", "new")
    (work / "swap.sh").write_text(appupdate.SWAP_SCRIPT)
    running = subprocess.Popen(["sleep", "0.5"])
    env = {**os.environ, "PATH": f"{shims}:{os.environ['PATH']}"}
    t0 = time.time()
    swap = subprocess.Popen(
        ["/bin/bash", str(work / "swap.sh"), str(running.pid), str(new), str(app), str(work)], env=env,
    )
    running.wait()  # reap it, as launchd does when the app exits; a zombie still answers kill -0
    assert swap.wait(timeout=30) == 0
    assert time.time() - t0 >= 0.4  # waited for the app to exit first
    assert (app / "Contents" / "marker").read_text() == "new"
    assert not work.exists()
    calls = (tmp_path / "calls").read_text()
    assert f"open {app}" in calls and "xattr -dr com.apple.quarantine" in calls
