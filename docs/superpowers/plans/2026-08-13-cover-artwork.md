# Embedded Cover Artwork Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Converted audio files carry cover art matching the TV show, sourced from Jellyfin sidecar images with fallbacks, so they display artwork when played on an iPod.

**Architecture:** A new `artwork.py` module resolves one cover per run through three tiers (Jellyfin sidecar image → cover embedded in the video container → frame grabbed from the episode), normalizing every result to a single 600px JPEG. `processor.py` embeds that JPEG as a second ffmpeg input during the existing encode pass. `cli.py` resolves once before the encode pool and shares the path across worker threads.

**Tech Stack:** Python 3.9+, ffmpeg/ffprobe via `subprocess`, pytest with `unittest.mock`.

Spec: `docs/superpowers/specs/2026-08-13-cover-artwork-design.md`

## Global Constraints

- **No new dependencies.** ffmpeg and ffprobe are already required by this project; everything here uses them plus the standard library.
- **Python >= 3.9.** Every module must keep `from __future__ import annotations` at the top so builtin generics (`list[str]`, `tuple[str, str]`) work in annotations.
- **Artwork never fails a conversion.** Every failure path logs a warning and returns a falsy value. No function in `artwork.py` raises.
- **Always on.** No CLI flag, no GUI control, no config key. Do not modify `gui.py`, `config.py`, or the argument parser.
- **WAV never receives artwork.** The wav muxer rejects video streams and the encode fails outright if an image is mapped into it.
- **`-vn` must never appear in an ffmpeg encode command.** It silently discards mapped cover art.
- Cover normalization target: long edge **600** px, aspect ratio preserved, no cropping.
- Sidecar base names, in preference order: `poster`, `folder`, `cover`, `default`. Extensions: `.jpg`, `.jpeg`, `.png`.
- Existing test style: `unittest.mock.patch("subprocess.run")` returning a `MagicMock(returncode=0, stdout="", stderr="")`, then asserting against `mock_run.call_args[0][0]`. Follow it.

---

### Task 1: Embed artwork in the encode command

Adds artwork support to `processor.py` and removes the `-vn` flag that would silently suppress it. Self-contained: no dependency on `artwork.py`, which does not exist yet.

**Files:**
- Modify: `tvshow_noisemaker/processor.py` (`process_file` at 39-128, `_build_simple_cmd` at 131-155, `_build_concat_cmd` at 158-186, new `_build_artwork_args`)
- Test: `tests/test_processor.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `_build_artwork_args(artwork_path: Optional[str], output_format: str) -> tuple[list[str], list[str]]` — returns `(input_args, output_args)`.
  - `process_file(..., artwork_path: Optional[str] = None)` — new final keyword parameter. Task 5 calls this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_processor.py`:

```python
# ---------------------------------------------------------------------------
# cover artwork
# ---------------------------------------------------------------------------


def _run_process_file(tmp_path, **kwargs):
    """Call process_file with a mocked ffmpeg and return the command list."""
    params = {
        "input_path": "/input/ep01.mp4",
        "output_path": str(tmp_path / "ep01.mp3"),
        "intro_end": 30.0,
        "outro_start": 2550.0,
    }
    params.update(kwargs)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(**params)
    return mock_run.call_args[0][0]


def test_simple_cmd_has_no_vn_flag(tmp_path):
    """-vn silently discards mapped cover art, so it must never be emitted."""
    cmd = _run_process_file(tmp_path)
    assert "-vn" not in cmd


def test_concat_cmd_has_no_vn_flag(tmp_path):
    cmd = _run_process_file(tmp_path, intro_start=165.0, intro_end=200.0)
    assert "-filter_complex" in cmd
    assert "-vn" not in cmd


def test_simple_cmd_embeds_artwork(tmp_path):
    cmd = _run_process_file(tmp_path, artwork_path="/work/cover.jpg")
    assert "/work/cover.jpg" in cmd
    idx = cmd.index("-map", cmd.index("-ac"))
    assert cmd[idx:idx + 2] == ["-map", "1:v"]
    assert "-disposition:v" in cmd
    assert "attached_pic" in cmd
    c_idx = cmd.index("-c:v")
    assert cmd[c_idx:c_idx + 2] == ["-c:v", "copy"]


def test_concat_cmd_embeds_artwork(tmp_path):
    cmd = _run_process_file(
        tmp_path, intro_start=165.0, intro_end=200.0, artwork_path="/work/cover.jpg"
    )
    assert "-filter_complex" in cmd
    assert "/work/cover.jpg" in cmd
    assert "1:v" in cmd
    assert "attached_pic" in cmd


def test_artwork_input_comes_after_source_input(tmp_path):
    """-ss/-t precede input 0 and must not apply to the cover image."""
    cmd = _run_process_file(tmp_path, artwork_path="/work/cover.jpg")
    assert cmd.index("/input/ep01.mp4") < cmd.index("/work/cover.jpg")


def test_no_artwork_args_when_path_is_none(tmp_path):
    cmd = _run_process_file(tmp_path)
    assert "1:v" not in cmd
    assert "-disposition:v" not in cmd


def test_artwork_skipped_for_wav(tmp_path):
    """The wav muxer rejects video streams; mapping an image breaks the encode."""
    cmd = _run_process_file(
        tmp_path,
        output_path=str(tmp_path / "ep01.wav"),
        output_format="wav",
        artwork_path="/work/cover.jpg",
    )
    assert "/work/cover.jpg" not in cmd
    assert "1:v" not in cmd
    assert "-disposition:v" not in cmd


@pytest.mark.parametrize("fmt", ["mp3", "m4a", "flac"])
def test_artwork_embedded_for_non_wav_formats(tmp_path, fmt):
    cmd = _run_process_file(
        tmp_path,
        output_path=str(tmp_path / f"ep01.{fmt}"),
        output_format=fmt,
        artwork_path="/work/cover.jpg",
    )
    assert "/work/cover.jpg" in cmd
    assert "1:v" in cmd


def test_build_artwork_args_returns_empty_pairs_for_wav():
    assert _build_artwork_args("/work/cover.jpg", "wav") == ([], [])


def test_build_artwork_args_returns_empty_pairs_for_none():
    assert _build_artwork_args(None, "mp3") == ([], [])


def test_build_artwork_args_populated_for_mp3():
    inputs, outputs = _build_artwork_args("/work/cover.jpg", "mp3")
    assert inputs == ["-i", "/work/cover.jpg"]
    assert outputs == [
        "-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic",
    ]
```

Update the import at the top of `tests/test_processor.py` (line 8):

```python
from tvshow_noisemaker.processor import (
    _build_artwork_args,
    build_output_path,
    process_file,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_processor.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_artwork_args'`

- [ ] **Step 3: Add `_build_artwork_args` to `processor.py`**

Insert after `_build_metadata_args` (after line 200):

```python
def _build_artwork_args(
    artwork_path: Optional[str], output_format: str
) -> tuple[list[str], list[str]]:
    """Return ``(input_args, output_args)`` for embedding cover art.

    WAV gets nothing: the wav muxer rejects video streams outright, so mapping
    an image into it fails the encode rather than being ignored.
    """
    if not artwork_path or output_format == "wav":
        return [], []
    return (
        ["-i", artwork_path],
        ["-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic"],
    )
```

- [ ] **Step 4: Thread `artwork_path` through the command builders**

Replace `_build_simple_cmd` (lines 131-155) with:

```python
def _build_simple_cmd(
    input_path: str,
    output_path: str,
    intro_end: float,
    outro_start: float,
    mp3_bitrate: int,
    output_format: str = "mp3",
    metadata: Optional[dict[str, str]] = None,
    artwork_path: Optional[str] = None,
) -> list:
    """Build an ffmpeg command that seeks past the intro and encodes to outro_start."""
    duration = outro_start - intro_end
    art_in, art_out = _build_artwork_args(artwork_path, output_format)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(intro_end),
        "-t", str(duration),
        "-i", input_path,
        *art_in,
        "-map", "0:a:0",
        *_codec_args(output_format, mp3_bitrate),
        "-ac", "2",
        *art_out,
    ]
    cmd.extend(_build_metadata_args(metadata, output_format))
    cmd.append(output_path)
    return cmd
```

Replace `_build_concat_cmd` (lines 158-186) with:

```python
def _build_concat_cmd(
    input_path: str,
    output_path: str,
    intro_start: float,
    intro_end: float,
    outro_start: float,
    mp3_bitrate: int,
    output_format: str = "mp3",
    metadata: Optional[dict[str, str]] = None,
    artwork_path: Optional[str] = None,
) -> list:
    """Build an ffmpeg command that keeps [0..intro_start] + [intro_end..outro_start]."""
    filter_complex = (
        f"[0:a]atrim=start=0:end={intro_start},asetpts=PTS-STARTPTS[pre];"
        f"[0:a]atrim=start={intro_end}:end={outro_start},asetpts=PTS-STARTPTS[post];"
        f"[pre][post]concat=n=2:v=0:a=1[out]"
    )
    art_in, art_out = _build_artwork_args(artwork_path, output_format)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        *art_in,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        *_codec_args(output_format, mp3_bitrate),
        "-ac", "2",
        *art_out,
    ]
    cmd.extend(_build_metadata_args(metadata, output_format))
    cmd.append(output_path)
    return cmd
```

Both `-vn` flags are gone. They are redundant — each path already maps its
output streams explicitly (`-map 0:a:0` and `-map [out]`) — and `-vn` discards
the mapped cover image regardless of argument order.

- [ ] **Step 5: Add the parameter to `process_file` and pass it down**

In `process_file`, add the parameter after `metadata` (line 47):

```python
    metadata: Optional[dict[str, str]] = None,
    artwork_path: Optional[str] = None,
) -> None:
```

Add to the docstring, after the `metadata:` entry (line 71):

```
    artwork_path:
        Optional path to a cover image to embed.  Ignored for wav output,
        whose muxer rejects video streams.
```

Update both builder calls (lines 91-102):

```python
    if has_cold_open:
        cmd = _build_concat_cmd(
            input_path, output_path,
            intro_start, intro_end, outro_start,
            mp3_bitrate, output_format, metadata, artwork_path,
        )
    else:
        cmd = _build_simple_cmd(
            input_path, output_path,
            intro_end, outro_start,
            mp3_bitrate, output_format, metadata, artwork_path,
        )
```

- [ ] **Step 6: Run the full processor suite**

Run: `.venv/bin/python -m pytest tests/test_processor.py -v`
Expected: PASS — all tests, including the pre-existing ones.

- [ ] **Step 7: Commit**

```bash
git add tvshow_noisemaker/processor.py tests/test_processor.py
git commit -m "feat: embed cover art in encode command, drop -vn

-vn silently discards a mapped cover image even with an explicit
-map 1:v. It is redundant with the explicit output maps, so it is
removed from both builders rather than made conditional."
```

---

### Task 2: Sidecar image lookup

Creates `artwork.py` with tier 1 only. No ffmpeg involved — pure filesystem lookup, so it tests fast.

**Files:**
- Create: `tvshow_noisemaker/artwork.py`
- Create: `tests/test_artwork.py`

**Interfaces:**
- Consumes: `_extract_season_number(name: str) -> Optional[int]` from `tvshow_noisemaker.metadata`.
- Produces: `_find_sidecar(input_dir: str) -> Optional[Path]`, plus the module constants `_MAX_EDGE`, `_SIDECAR_NAMES`, `_SIDECAR_EXTS`, `_COVER_CODECS`, `_FRAME_GRAB_OFFSET`, `_FFMPEG_TIMEOUT` used by Tasks 3 and 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_artwork.py`:

```python
"""Tests for tvshow_noisemaker.artwork."""

from pathlib import Path

import pytest

from tvshow_noisemaker.artwork import _find_sidecar


def _touch(path: Path) -> Path:
    """Create an empty file, making parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


# ---------------------------------------------------------------------------
# _find_sidecar
# ---------------------------------------------------------------------------


def test_sidecar_prefers_season_folder(tmp_path):
    season = tmp_path / "Family Guy" / "Season 02"
    _touch(season / "poster.jpg")
    _touch(tmp_path / "Family Guy" / "poster.jpg")
    assert _find_sidecar(str(season)) == season / "poster.jpg"


def test_sidecar_falls_back_to_season_numbered_poster(tmp_path):
    """Jellyfin writes per-season art to the show root as seasonNN-poster.jpg."""
    season = tmp_path / "Family Guy" / "Season 02"
    season.mkdir(parents=True)
    _touch(tmp_path / "Family Guy" / "season02-poster.jpg")
    _touch(tmp_path / "Family Guy" / "poster.jpg")
    assert _find_sidecar(str(season)) == tmp_path / "Family Guy" / "season02-poster.jpg"


def test_sidecar_falls_back_to_show_poster(tmp_path):
    season = tmp_path / "Family Guy" / "Season 02"
    season.mkdir(parents=True)
    _touch(tmp_path / "Family Guy" / "folder.jpg")
    assert _find_sidecar(str(season)) == tmp_path / "Family Guy" / "folder.jpg"


def test_sidecar_name_preference_order(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    _touch(season / "cover.jpg")
    _touch(season / "default.jpg")
    _touch(season / "folder.jpg")
    _touch(season / "poster.jpg")
    assert _find_sidecar(str(season)).name == "poster.jpg"


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png"])
def test_sidecar_accepts_each_extension(tmp_path, ext):
    season = tmp_path / "Show" / "Season 01"
    _touch(season / f"poster{ext}")
    assert _find_sidecar(str(season)).name == f"poster{ext}"


def test_sidecar_does_not_escape_a_non_season_folder(tmp_path):
    """A plain folder must not inherit artwork belonging to the library root."""
    show = tmp_path / "Family Guy"
    show.mkdir(parents=True)
    _touch(tmp_path / "poster.jpg")
    assert _find_sidecar(str(show)) is None


def test_sidecar_returns_none_when_absent(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    assert _find_sidecar(str(season)) is None


def test_sidecar_ignores_a_directory_named_like_an_image(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    (season / "poster.jpg").mkdir(parents=True)
    assert _find_sidecar(str(season)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_artwork.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tvshow_noisemaker.artwork'`

- [ ] **Step 3: Create `artwork.py`**

```python
"""Resolve cover artwork for a conversion run.

Three tiers, first hit wins:

1. A Jellyfin sidecar image in the season or show folder.
2. A cover image embedded in the video container (a Matroska attachment,
   which ffmpeg exposes as a video stream).
3. A frame grabbed from the episode itself.

Every tier funnels through the same normalization step, so callers get one
uniform JPEG regardless of source.  Nothing here raises: artwork is a nicety
and must never fail a conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

from tvshow_noisemaker.metadata import _extract_season_number

logger = logging.getLogger(__name__)

# Long edge of the embedded cover, in pixels.
_MAX_EDGE = 600

# Sidecar base names, in preference order, and the extensions tried for each.
_SIDECAR_NAMES = ("poster", "folder", "cover", "default")
_SIDECAR_EXTS = (".jpg", ".jpeg", ".png")

# Image codecs that can appear as an attached cover inside a video container.
_COVER_CODECS = {"mjpeg", "png"}

# Seconds past the detected intro to seek when grabbing a fallback frame.
_FRAME_GRAB_OFFSET = 60.0

_FFMPEG_TIMEOUT = 60


def _first_existing(directory: Path, names: Sequence[str]) -> Optional[Path]:
    """Return the first ``<name><ext>`` file that exists in *directory*."""
    for name in names:
        for ext in _SIDECAR_EXTS:
            candidate = directory / f"{name}{ext}"
            if candidate.is_file():
                return candidate
    return None


def _find_sidecar(input_dir: str) -> Optional[Path]:
    """Find a Jellyfin sidecar image for *input_dir*, searching outward."""
    season_dir = Path(input_dir)

    found = _first_existing(season_dir, _SIDECAR_NAMES)
    if found:
        return found

    # Only look at the parent when this really is a season folder.  Otherwise
    # the parent is the library root, whose artwork belongs to another show.
    season = _extract_season_number(season_dir.name)
    show_dir = season_dir.parent
    if season is None or show_dir == season_dir:
        return None

    found = _first_existing(show_dir, (f"season{season:02d}-poster",))
    if found:
        return found
    return _first_existing(show_dir, _SIDECAR_NAMES)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_artwork.py -v`
Expected: PASS — 10 tests (the extension test parametrizes to 3).

- [ ] **Step 5: Commit**

```bash
git add tvshow_noisemaker/artwork.py tests/test_artwork.py
git commit -m "feat: add Jellyfin sidecar artwork lookup"
```

---

### Task 3: Locate a cover embedded in the video

Tier 2's probe. Split from Task 4 because the ffprobe JSON parsing is where the real risk lives: the cover's stream index varies between files (3 and 4 both observed in `test_media`), and the cover carries no `attached_pic` disposition, so only the `filename`/`mimetype` tags distinguish it from the primary video stream.

**Files:**
- Modify: `tvshow_noisemaker/artwork.py`
- Test: `tests/test_artwork.py` (append)

**Interfaces:**
- Consumes: `_COVER_CODECS`, `_FFMPEG_TIMEOUT` from Task 2.
- Produces: `_find_embedded_cover_index(video_path: str) -> Optional[int]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artwork.py`:

```python
# ---------------------------------------------------------------------------
# _find_embedded_cover_index
# ---------------------------------------------------------------------------


def _ffprobe_result(streams):
    """A MagicMock standing in for a successful ffprobe run."""
    return MagicMock(
        returncode=0, stdout=json.dumps({"streams": streams}), stderr=""
    )


def test_embedded_cover_skips_the_primary_video_stream():
    streams = [
        {"index": 0, "codec_name": "hevc", "disposition": {"attached_pic": 0}},
        {
            "index": 4,
            "codec_name": "mjpeg",
            "disposition": {"attached_pic": 0},
            "tags": {"filename": "cover.jpg", "mimetype": "image/jpeg"},
        },
    ]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") == 4


def test_embedded_cover_index_is_probed_not_assumed():
    """Real files put the cover at varying indices; 3 and 4 both observed."""
    streams = [
        {"index": 0, "codec_name": "hevc", "disposition": {}},
        {"index": 3, "codec_name": "mjpeg", "disposition": {},
         "tags": {"filename": "cover.jpg"}},
    ]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") == 3


def test_embedded_cover_accepts_attached_pic_without_tags():
    streams = [
        {"index": 0, "codec_name": "h264", "disposition": {}},
        {"index": 1, "codec_name": "png", "disposition": {"attached_pic": 1}},
    ]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") == 1


def test_embedded_cover_ignores_untagged_non_cover_image_stream():
    """An mjpeg stream with neither tags nor disposition is not a cover."""
    streams = [
        {"index": 0, "codec_name": "hevc", "disposition": {}},
        {"index": 1, "codec_name": "mjpeg", "disposition": {}},
    ]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_when_no_image_stream():
    streams = [{"index": 0, "codec_name": "hevc", "disposition": {}}]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_on_unparseable_output():
    result = MagicMock(returncode=0, stdout="not json at all", stderr="")
    with patch("subprocess.run", return_value=result):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_on_ffprobe_nonzero_exit():
    result = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch("subprocess.run", return_value=result):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_when_ffprobe_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError("no ffprobe")):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_on_timeout():
    exc = subprocess.TimeoutExpired("ffprobe", 60)
    with patch("subprocess.run", side_effect=exc):
        assert _find_embedded_cover_index("/v.mkv") is None
```

Add to the imports at the top of `tests/test_artwork.py`:

```python
import json
import subprocess
from unittest.mock import MagicMock, patch
```

and extend the module import:

```python
from tvshow_noisemaker.artwork import _find_embedded_cover_index, _find_sidecar
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_artwork.py -v`
Expected: FAIL — `ImportError: cannot import name '_find_embedded_cover_index'`

- [ ] **Step 3: Implement the probe**

Add `import json` and `import subprocess` to `artwork.py`'s imports, then append:

```python
def _find_embedded_cover_index(video_path: str) -> Optional[int]:
    """Return the stream index of a cover image inside *video_path*.

    Matroska attachments surface as video streams, so the primary video track
    has to be told apart from the cover.  On the observed library the cover
    carries ``filename``/``mimetype`` tags and no ``attached_pic`` disposition,
    and its index varies between files -- hence the probe.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=index,codec_name,disposition",
        "-show_entries", "stream_tags=filename,mimetype",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Artwork: ffprobe failed on %s: %s", video_path, exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "Artwork: ffprobe exited %d on %s", result.returncode, video_path
        )
        return None

    try:
        streams = json.loads(result.stdout).get("streams", [])
    except (ValueError, AttributeError) as exc:
        logger.warning("Artwork: could not parse ffprobe output: %s", exc)
        return None

    # ffprobe emits streams in index order, so the first match is the lowest.
    for stream in streams:
        if stream.get("codec_name") not in _COVER_CODECS:
            continue
        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        if tags.get("filename") or tags.get("mimetype") or disposition.get("attached_pic"):
            return stream.get("index")
    return None
```

`json.JSONDecodeError` subclasses `ValueError`, so the one clause covers both
malformed JSON and a non-dict payload.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_artwork.py -v`
Expected: PASS — all sidecar and probe tests.

- [ ] **Step 5: Commit**

```bash
git add tvshow_noisemaker/artwork.py tests/test_artwork.py
git commit -m "feat: locate cover art embedded in video containers"
```

---

### Task 4: Normalization and tier orchestration

Completes `artwork.py`: the shared ffmpeg normalization step and the public `resolve_artwork` that walks the three tiers.

**Files:**
- Modify: `tvshow_noisemaker/artwork.py`
- Test: `tests/test_artwork.py` (append)

**Interfaces:**
- Consumes: `_find_sidecar` (Task 2), `_find_embedded_cover_index` (Task 3), and the module constants.
- Produces:
  - `_normalize(input_args: list[str], dest: str) -> bool`
  - `resolve_artwork(input_dir: str, sample_video: str, intro_end: float, workdir: str) -> Optional[tuple[str, str]]` returning `(jpeg_path, source_description)`. Task 5 calls this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_artwork.py`:

```python
# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_builds_a_scale_filter(tmp_path):
    dest = tmp_path / "cover.jpg"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        dest.write_bytes(b"jpegdata")
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is True
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "/src/poster.jpg" in cmd
    assert "scale=w=600:h=600:force_original_aspect_ratio=decrease" in cmd
    assert cmd[-1] == str(dest)


def test_normalize_false_on_ffmpeg_failure(tmp_path):
    dest = tmp_path / "cover.jpg"
    result = MagicMock(returncode=1, stdout="", stderr="bad image")
    with patch("subprocess.run", return_value=result):
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False


def test_normalize_false_when_output_missing(tmp_path):
    """ffmpeg can exit 0 without writing anything usable."""
    dest = tmp_path / "cover.jpg"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False


def test_normalize_false_when_output_empty(tmp_path):
    dest = tmp_path / "cover.jpg"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        dest.write_bytes(b"")
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False


def test_normalize_false_when_ffmpeg_missing(tmp_path):
    dest = tmp_path / "cover.jpg"
    with patch("subprocess.run", side_effect=FileNotFoundError("no ffmpeg")):
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False


# ---------------------------------------------------------------------------
# resolve_artwork
# ---------------------------------------------------------------------------


def test_resolve_uses_sidecar_first(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    _touch(season / "poster.jpg")
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=True) as norm, \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index") as probe:
        result = resolve_artwork(str(season), "/v.mkv", 30.0, str(work))
    assert result is not None
    path, source = result
    assert path == str(work / "cover.jpg")
    assert "poster.jpg" in source
    probe.assert_not_called()
    assert str(season / "poster.jpg") in norm.call_args[0][0]


def test_resolve_falls_through_to_embedded_cover(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=True) as norm, \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=3):
        result = resolve_artwork(str(season), "/v.mkv", 30.0, str(work))
    assert result is not None
    assert "embedded" in result[1]
    assert "0:3" in norm.call_args[0][0]


def test_resolve_falls_through_to_frame_grab(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=True) as norm, \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=None):
        result = resolve_artwork(str(season), "/v.mkv", 30.0, str(work))
    assert result is not None
    assert "frame" in result[1]
    args = norm.call_args[0][0]
    assert "-ss" in args
    assert "90.0" in args  # intro_end 30.0 + 60s offset
    assert "0:v:0" in args


def test_resolve_frame_grab_seek_never_negative(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=True) as norm, \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=None):
        resolve_artwork(str(season), "/v.mkv", -500.0, str(work))
    args = norm.call_args[0][0]
    seek = float(args[args.index("-ss") + 1])
    assert seek >= 0.0


def test_resolve_falls_through_when_sidecar_is_corrupt(tmp_path):
    """A sidecar that fails to normalize must not block the embedded tier."""
    season = tmp_path / "Show" / "Season 01"
    _touch(season / "poster.jpg")
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", side_effect=[False, True]), \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=3):
        result = resolve_artwork(str(season), "/v.mkv", 30.0, str(work))
    assert result is not None
    assert "embedded" in result[1]


def test_resolve_returns_none_when_every_tier_fails(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=False), \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=None):
        assert resolve_artwork(str(season), "/v.mkv", 30.0, str(work)) is None
```

Extend the module import:

```python
from tvshow_noisemaker.artwork import (
    _find_embedded_cover_index,
    _find_sidecar,
    _normalize,
    resolve_artwork,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_artwork.py -v`
Expected: FAIL — `ImportError: cannot import name '_normalize'`

- [ ] **Step 3: Implement `_normalize`**

Append to `artwork.py`:

```python
def _normalize(input_args: list[str], dest: str) -> bool:
    """Write a single downscaled JPEG to *dest*.  Return True on success.

    Every tier goes through here, so the embedded image is the same shape and
    size no matter where it came from.  Aspect ratio is preserved and nothing
    is cropped, so poster text stays intact.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        *input_args,
        "-frames:v", "1",
        "-vf",
        f"scale=w={_MAX_EDGE}:h={_MAX_EDGE}:force_original_aspect_ratio=decrease",
        dest,
    ]
    logger.debug("Artwork cmd: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Artwork: ffmpeg failed: %s", exc)
        return False

    if result.returncode != 0:
        logger.warning(
            "Artwork: ffmpeg exited %d: %s", result.returncode, result.stderr[-300:]
        )
        return False

    output = Path(dest)
    return output.is_file() and output.stat().st_size > 0
```

- [ ] **Step 4: Implement `resolve_artwork`**

Append to `artwork.py`:

```python
def resolve_artwork(
    input_dir: str,
    sample_video: str,
    intro_end: float,
    workdir: str,
) -> Optional[tuple[str, str]]:
    """Resolve one cover image for a run.

    Returns ``(path_to_jpeg, source_description)``, or ``None`` when no
    artwork could be produced.  *sample_video* is one episode standing in for
    the whole run: every episode of a season shares an album, and probing each
    file costs seconds.  *intro_end* is that episode's detected intro
    boundary, used to seek past the intro when grabbing a frame.
    """
    dest = str(Path(workdir) / "cover.jpg")

    sidecar = _find_sidecar(input_dir)
    if sidecar is not None and _normalize(["-i", str(sidecar)], dest):
        return dest, f"sidecar image ({sidecar.name})"

    index = _find_embedded_cover_index(sample_video)
    if index is not None and _normalize(
        ["-i", sample_video, "-map", f"0:{index}"], dest
    ):
        return dest, f"embedded cover (stream {index})"

    # ponytail: no blackframe detection -- a dull frame beats no art, and this
    # tier never fired across the test library.  Add a blackframe filter or a
    # TMDB lookup here if real runs start producing bad covers.
    seek = max(intro_end + _FRAME_GRAB_OFFSET, 0.0)
    if _normalize(["-ss", str(seek), "-i", sample_video, "-map", "0:v:0"], dest):
        return dest, f"video frame at {seek:.0f}s"

    logger.warning("Artwork: no cover could be resolved for %s", input_dir)
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_artwork.py -v`
Expected: PASS — all artwork tests.

- [ ] **Step 6: Commit**

```bash
git add tvshow_noisemaker/artwork.py tests/test_artwork.py
git commit -m "feat: normalize covers and orchestrate artwork tiers"
```

---

### Task 5: Wire artwork into the CLI

Resolves artwork once per run and hands the path to every encode.

**Files:**
- Modify: `tvshow_noisemaker/cli.py` (imports at 19-39, Phase 2 header at 530, `_encode_one` at 542-563, pool teardown after ~605)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `resolve_artwork` (Task 4), `process_file(..., artwork_path=...)` (Task 1).
- Produces: no new public API.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

These use the existing `run()` + `_make_args()` harness already in the file, so
they exercise the real wiring rather than a stand-in.

```python
# ---------------------------------------------------------------------------
# cover artwork wiring
# ---------------------------------------------------------------------------


def _artwork_run(tmp_path, resolved, episode_count=3, output_format=None):
    """Run a conversion with artwork resolution patched. Returns (mocks, exit_code)."""
    show_dir = tmp_path / "Family Guy" / "Season 2"
    show_dir.mkdir(parents=True)
    for n in range(1, episode_count + 1):
        (show_dir / f"Family.Guy.S02E{n:02d}.Episode.mp4").touch()

    args = _make_args(
        input_dir=str(show_dir),
        output_dir=str(tmp_path / "out"),
        output_format=output_format,
        fingerprint=False,
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
        patch("tvshow_noisemaker.cli.resolve_artwork", return_value=resolved) as mock_art,
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)
    return mock_art, mock_proc, exit_code


def test_run_resolves_artwork_once_for_the_whole_run(tmp_path):
    """Probing every episode costs seconds each; one cover serves the season."""
    mock_art, mock_proc, exit_code = _artwork_run(
        tmp_path, ("/work/cover.jpg", "sidecar image (poster.jpg)")
    )
    assert exit_code == 0
    mock_art.assert_called_once()
    assert mock_proc.call_count == 3


def test_run_passes_artwork_path_to_every_encode(tmp_path):
    _, mock_proc, _ = _artwork_run(
        tmp_path, ("/work/cover.jpg", "sidecar image (poster.jpg)")
    )
    for _, kwargs in mock_proc.call_args_list:
        assert kwargs["artwork_path"] == "/work/cover.jpg"


def test_run_resolves_artwork_from_first_job_path_and_intro_end(tmp_path):
    mock_art, _, _ = _artwork_run(
        tmp_path, ("/work/cover.jpg", "embedded cover (stream 4)")
    )
    input_dir_arg, sample_video, intro_end, workdir = mock_art.call_args[0]
    assert sample_video.endswith(".mp4")
    assert intro_end == 30.0
    assert Path(input_dir_arg).name == "Season 2"
    assert isinstance(workdir, str)


def test_run_skips_artwork_entirely_for_wav(tmp_path):
    """The wav muxer rejects video streams, so resolving one is wasted work."""
    mock_art, mock_proc, exit_code = _artwork_run(
        tmp_path, ("/work/cover.jpg", "sidecar image (poster.jpg)"),
        output_format="wav",
    )
    assert exit_code == 0
    mock_art.assert_not_called()
    for _, kwargs in mock_proc.call_args_list:
        assert kwargs["artwork_path"] is None


def test_run_continues_when_no_artwork_is_found(tmp_path):
    mock_art, mock_proc, exit_code = _artwork_run(tmp_path, None)
    assert exit_code == 0
    mock_art.assert_called_once()
    assert mock_proc.call_count == 3
    for _, kwargs in mock_proc.call_args_list:
        assert kwargs["artwork_path"] is None


def test_run_removes_the_artwork_temp_dir(tmp_path):
    mock_art, _, _ = _artwork_run(
        tmp_path, ("/work/cover.jpg", "sidecar image (poster.jpg)")
    )
    workdir = mock_art.call_args[0][3]
    assert not Path(workdir).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -k artwork -v`
Expected: FAIL — `AttributeError: <module 'tvshow_noisemaker.cli'> does not have the attribute 'resolve_artwork'`, raised by `patch`.

- [ ] **Step 3: Add the imports**

In `cli.py`, add `import tempfile` to the stdlib imports (alphabetically, after `import sys` at line 26), and add the artwork import after the `metadata` import (line 38):

```python
from tvshow_noisemaker.artwork import resolve_artwork
```

Place it before the `metadata` import to keep the block alphabetical:

```python
from tvshow_noisemaker.artwork import resolve_artwork
from tvshow_noisemaker.config import VALID_OUTPUT_FORMATS, build_config, load_config
from tvshow_noisemaker.detector import (
    detect_by_fingerprinting,
    get_last_fingerprint_stats,
    get_duration,
    resolve_boundaries,
)
from tvshow_noisemaker.metadata import infer_track_metadata
from tvshow_noisemaker.processor import build_output_path, process_file
```

- [ ] **Step 4: Resolve artwork before the encode pool**

In `cli.py`, immediately after the Phase 2 comment (line 530) and before
`max_workers = cfg.workers or _default_encode_workers()`, insert:

```python
    # Resolve cover art once: every episode in a season shares an album, and
    # probing each file costs seconds.  wav is skipped -- its muxer rejects
    # video streams outright.
    artwork_path: Optional[str] = None
    artwork_dir = tempfile.mkdtemp(prefix="tvshow-art-")
    if jobs and cfg.output_format != "wav":
        resolved = resolve_artwork(str(input_dir), jobs[0][0], jobs[0][3], artwork_dir)
        if resolved is not None:
            artwork_path, artwork_source = resolved
            con.info("Artwork: %s", artwork_source)
        else:
            con.info("Artwork: none found — encoding without cover art")
```

`jobs` entries are tuples of
`(filepath, basename, intro_start, intro_end, outro_start, content_duration, trimmed)`,
so `jobs[0][0]` is the first episode's path and `jobs[0][3]` its `intro_end`.

- [ ] **Step 5: Pass the path into the encode**

In `_encode_one`, add the argument to the `process_file` call (around line 556):

```python
        process_file(
            input_path=filepath,
            output_path=output_path,
            intro_end=intro_end,
            outro_start=outro_start,
            intro_start=intro_start,
            mp3_bitrate=cfg.mp3_bitrate,
            output_format=cfg.output_format,
            metadata=metadata,
            artwork_path=artwork_path,
        )
```

`_encode_one` is a closure over the enclosing function's scope, so it reads
`artwork_path` directly. The value is assigned before the closure is defined
and never reassigned, and it is read-only across the pool, so no locking is
needed.

- [ ] **Step 6: Clean up the temp directory**

Immediately after the `with concurrent.futures.ThreadPoolExecutor(...)` block
closes (after the progress-bar loop, before the summary section around line
605), add at the same indentation as the `with`:

```python
    # ponytail: cleaned up after the pool rather than via try/finally, which
    # would mean re-indenting the whole encode phase.  A crash leaks one 26 KB
    # JPEG into the OS temp dir, which the OS reclaims.
    shutil.rmtree(artwork_dir, ignore_errors=True)
```

`shutil` is already imported at line 25.

- [ ] **Step 7: Run the CLI suite**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS — new artwork tests plus all pre-existing CLI tests.

- [ ] **Step 8: Run the whole suite**

Run: `.venv/bin/python -m pytest -v`
Expected: PASS — no regressions in detector, config, metadata, processor, or gui parser tests.

- [ ] **Step 9: Commit**

```bash
git add tvshow_noisemaker/cli.py tests/test_cli.py
git commit -m "feat: resolve cover art once per run and embed it"
```

---

### Task 6: End-to-end verification against real media

Everything above runs on mocks. This task proves the ffmpeg commands actually work on the real files, and guards the wav path that fails at the muxer rather than at any Python boundary.

**Files:**
- Create: `tests/test_artwork_integration.py`

**Interfaces:**
- Consumes: `resolve_artwork` (Task 4), `process_file` (Task 1).
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_artwork_integration.py`:

```python
"""End-to-end artwork checks against the bundled test media.

Skipped when ffmpeg, ffprobe, or test_media are unavailable.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from tvshow_noisemaker.artwork import resolve_artwork
from tvshow_noisemaker.processor import process_file

_TEST_MEDIA = Path(__file__).resolve().parent.parent / "test_media" / "Season 02"

needs_media = pytest.mark.skipif(
    shutil.which("ffmpeg") is None
    or shutil.which("ffprobe") is None
    or not _TEST_MEDIA.is_dir(),
    reason="requires ffmpeg, ffprobe and test_media",
)


def _sample_episode() -> str:
    episodes = sorted(_TEST_MEDIA.glob("*.mkv"))
    assert episodes, f"no .mkv files in {_TEST_MEDIA}"
    return str(episodes[0])


def _probe_streams(path: str) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "stream=codec_type,codec_name",
         "-of", "csv", path],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@needs_media
def test_resolves_cover_from_real_media(tmp_path):
    """test_media has no sidecar, so this exercises the embedded tier."""
    resolved = resolve_artwork(str(_TEST_MEDIA), _sample_episode(), 10.0, str(tmp_path))
    assert resolved is not None
    path, source = resolved
    assert "embedded" in source
    assert Path(path).is_file()
    assert Path(path).stat().st_size > 0
    assert "mjpeg" in _probe_streams(path)


@needs_media
def test_encoded_mp3_contains_cover_art(tmp_path):
    resolved = resolve_artwork(str(_TEST_MEDIA), _sample_episode(), 10.0, str(tmp_path))
    assert resolved is not None
    out = str(tmp_path / "out.mp3")
    process_file(
        input_path=_sample_episode(),
        output_path=out,
        intro_end=40.0,
        outro_start=70.0,
        mp3_bitrate=128,
        metadata={"artist": "Family Guy", "album": "Season 2"},
        artwork_path=resolved[0],
    )
    streams = _probe_streams(out)
    assert "audio" in streams
    assert "mjpeg" in streams


@needs_media
def test_wav_encode_succeeds_with_artwork_supplied(tmp_path):
    """The wav muxer rejects video streams; artwork must be dropped, not passed."""
    resolved = resolve_artwork(str(_TEST_MEDIA), _sample_episode(), 10.0, str(tmp_path))
    assert resolved is not None
    out = str(tmp_path / "out.wav")
    process_file(
        input_path=_sample_episode(),
        output_path=out,
        intro_end=40.0,
        outro_start=55.0,
        output_format="wav",
        artwork_path=resolved[0],
    )
    streams = _probe_streams(out)
    assert "audio" in streams
    assert "mjpeg" not in streams
```

- [ ] **Step 2: Run the integration tests**

Run: `.venv/bin/python -m pytest tests/test_artwork_integration.py -v`
Expected: PASS — 3 tests. Allow up to a minute; ffprobe on these HEVC files takes several seconds each.

If they are reported as skipped, confirm `ffmpeg`/`ffprobe` are on PATH and
`test_media/Season 02` exists — do not treat a skip as a pass.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/python -m pytest -v`
Expected: PASS, no skips in the artwork integration file.

- [ ] **Step 4: Verify by ear and eye**

Run a real conversion and confirm the log names an artwork source:

```bash
.venv/bin/python -m tvshow_noisemaker.cli \
  --input-dir test_media/Season\ 02 \
  --output-dir /tmp/artwork-check \
  --intro-duration 30 --outro-duration 20
```

Expected: the run logs a line like `Artwork: embedded cover (stream 4)`.
Then confirm the output carries the image:

```bash
ffprobe -v error -show_entries stream=codec_type,codec_name -of csv /tmp/artwork-check/*.mp3 | head
```

Expected: each file reports both an `audio` stream and an `mjpeg` stream.

- [ ] **Step 5: Commit**

```bash
git add tests/test_artwork_integration.py
git commit -m "test: verify cover art end to end against real media"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Always on, no flags/GUI | Global constraints; Task 5 touches only `cli.py` |
| One cover per run | Task 5 step 4 |
| Tier 1 sidecar, search order | Task 2 |
| Tier 2 embedded, probed index | Task 3 |
| Tier 3 frame grab past intro | Task 4 step 4 |
| Normalization to 600px, aspect preserved | Task 4 step 3 |
| `_build_artwork_args`, cover input after source | Task 1 steps 3-4 |
| `-vn` removal + regression test | Task 1 steps 1, 4 |
| WAV excluded | Task 1 step 3; Task 5 step 4; Task 6 step 1 |
| Never raises / falls through | Tasks 3, 4 error tests |
| Temp dir cleanup | Task 5 step 6 |
| Format support table | Task 1 parametrized test; Task 6 |

**Type consistency:** `resolve_artwork` returns `Optional[tuple[str, str]]` in Task 4 and is unpacked as a 2-tuple in Task 5 and Task 6. `_normalize(input_args, dest) -> bool` is called with a list first, path second in all three call sites. `_build_artwork_args` returns a 2-tuple of lists in Task 1 and is unpacked as `art_in, art_out` in both builders. `artwork_path` is the parameter name in `process_file`, both builders, and the Task 5 call site.

**Note on `_extract_season_number`:** Task 2 imports this private name from `metadata.py`. It is reused rather than duplicated so season parsing stays in one place; if a reviewer prefers a public alias, that is a one-line rename in `metadata.py` plus the import.
