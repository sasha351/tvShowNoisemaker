"""End-to-end artwork checks against the bundled test media.

Skipped when ffmpeg, ffprobe, or test_media are unavailable.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from tvshow_noisemaker.artwork import resolve_artwork
from tvshow_noisemaker.processor import process_file

_TEST_MEDIA = Path(__file__).resolve().parent.parent / "test_media" / "FamilyGuy02"

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
@pytest.mark.parametrize("output_format", ["mp3", "m4a", "flac"])
def test_encoded_output_contains_cover_art(tmp_path, output_format):
    """m4a and flac muxers handle attached pictures differently from mp3.

    m4a legitimately also carries a bin_data stream (the iTunes metadata
    atom), so this only asserts the cover is present, not that nothing else is.
    """
    resolved = resolve_artwork(str(_TEST_MEDIA), _sample_episode(), 10.0, str(tmp_path))
    assert resolved is not None
    out = str(tmp_path / f"out.{output_format}")
    process_file(
        input_path=_sample_episode(),
        output_path=out,
        intro_end=40.0,
        outro_start=70.0,
        output_format=output_format,
        mp3_bitrate=128,
        metadata={"artist": "Family Guy", "album": "Season 2"},
        artwork_path=resolved[0],
    )
    streams = _probe_streams(out)
    assert "audio" in streams
    assert "mjpeg" in streams


@needs_media
def test_concat_path_with_artwork_keeps_trim_intact(tmp_path):
    """The cold-open path mixes -filter_complex with an unfiltered -map 1:v
    -c:v copy for the artwork stream -- exactly the interaction a mocked
    test can't validate."""
    resolved = resolve_artwork(str(_TEST_MEDIA), _sample_episode(), 10.0, str(tmp_path))
    assert resolved is not None
    out = str(tmp_path / "out.mp3")
    process_file(
        input_path=_sample_episode(),
        output_path=out,
        intro_start=20.0,
        intro_end=40.0,
        outro_start=65.0,
        mp3_bitrate=128,
        artwork_path=resolved[0],
    )
    streams = _probe_streams(out)
    assert "audio" in streams
    assert "mjpeg" in streams

    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-show_entries", "format=duration",
         "-of", "csv=p=0", out],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    duration = float(result.stdout.strip())
    # [0..20] + [40..65] = 20s + 25s = 45s.
    assert duration == pytest.approx(45.0, abs=0.5)


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
