"""m4a stream copy: eligibility, fallbacks, and exact cuts against real ffmpeg on generated media."""

import shutil
import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tvshow_noisemaker import processor
from tvshow_noisemaker.processor import Terminated, can_stream_copy, probe_audio, process_file


@pytest.mark.parametrize("info, kbps, expected", [
    ({"codec": "aac", "channels": 2, "bit_rate": None}, 256, True),
    ({"codec": "aac", "channels": 2, "bit_rate": 192_000}, 256, True),
    ({"codec": "aac", "channels": 2, "bit_rate": 320_000}, 256, False),  # user asked for smaller files
    ({"codec": "aac", "channels": 6, "bit_rate": None}, 256, False),     # needs a downmix
    ({"codec": "ac3", "channels": 2, "bit_rate": None}, 256, False),
    (None, 256, False),
])
def test_can_stream_copy(info, kbps, expected):
    assert can_stream_copy(info, kbps) is expected


def test_probe_audio_parses_ffprobe_json():
    out = '{"streams": [{"codec_name": "aac", "channels": 2, "bit_rate": "128000"}]}'
    with patch("tvshow_noisemaker.processor.subprocess.run", return_value=MagicMock(returncode=0, stdout=out)):
        assert probe_audio("/x.mkv") == {"codec": "aac", "channels": 2, "bit_rate": 128000}
    with patch("tvshow_noisemaker.processor.subprocess.run", return_value=MagicMock(returncode=1, stdout="")):
        assert probe_audio("/x.mkv") is None


_AAC = {"codec": "aac", "channels": 2, "bit_rate": None}


def test_copy_failure_falls_back_to_encode(tmp_path):
    out = str(tmp_path / "ep.m4a")
    with patch("tvshow_noisemaker.processor.probe_audio", return_value=_AAC), \
         patch("tvshow_noisemaker.processor._copy_trim", side_effect=RuntimeError("boom")), \
         patch("tvshow_noisemaker.processor._run_ffmpeg") as run:
        process_file("/in.mkv", out, 30.0, 100.0, output_format="m4a")
    run.assert_called_once()
    assert "-acodec" in run.call_args[0][0]


def test_cancelled_copy_does_not_fall_back(tmp_path):
    out = str(tmp_path / "ep.m4a")
    with patch("tvshow_noisemaker.processor.probe_audio", return_value=_AAC), \
         patch("tvshow_noisemaker.processor._copy_trim", side_effect=Terminated("cancelled")), \
         patch("tvshow_noisemaker.processor._run_ffmpeg") as run:
        with pytest.raises(Terminated):
            process_file("/in.mkv", out, 30.0, 100.0, output_format="m4a")
    run.assert_not_called()


@pytest.mark.parametrize("fmt, stream_copy", [("mp3", True), ("m4a", False)])
def test_no_copy_for_mp3_or_when_disabled(tmp_path, fmt, stream_copy):
    with patch("tvshow_noisemaker.processor.probe_audio", return_value=_AAC) as probe, \
         patch("tvshow_noisemaker.processor._copy_trim") as copy, \
         patch("tvshow_noisemaker.processor._run_ffmpeg"):
        process_file("/in.mkv", str(tmp_path / f"ep.{fmt}"), 30.0, 100.0,
                     output_format=fmt, stream_copy=stream_copy)
    copy.assert_not_called()
    probe.assert_not_called()


def test_terminate_all_marks_process_as_terminated():
    proc = MagicMock()
    proc.communicate.return_value = ("", "")
    proc.returncode = -15

    def _popen(*_a, **_k):
        return proc

    def _communicate(timeout=None):
        processor.terminate_all()  # a cancel arrives while ffmpeg runs
        return ("", "")

    proc.communicate.side_effect = _communicate
    with patch("tvshow_noisemaker.processor.subprocess.Popen", side_effect=_popen):
        with pytest.raises(Terminated):
            processor._run_ffmpeg(["ffmpeg"], "/in.mkv")
    assert proc not in processor._terminated


# ── real ffmpeg ─────────────────────────────────────────────────────────────

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="requires ffmpeg",
)
_SR = 48_000
# 0–10 s cold open 440 Hz, 10–40 intro 880 Hz, 40–100 content 1320 Hz, 100–120 credits 1760 Hz
_TONES = [(440, 10), (880, 30), (1320, 60), (1760, 20)]


@pytest.fixture(scope="module")
def aac_episode(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "Show - S01E01 - It's, a test.mkv"
    inputs, labels = [], ""
    for i, (freq, dur) in enumerate(_TONES):
        inputs += ["-f", "lavfi", "-i", f"sine=f={freq}:d={dur}:r={_SR}"]
        labels += f"[{i + 1}]"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=120",  # 10 s GOP: seeks land early
        *inputs,
        "-filter_complex", f"{labels}concat=n=4:v=0:a=1,aformat=channel_layouts=stereo[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "mpeg4", "-g", "250", "-c:a", "aac", "-b:a", "128k",
        str(path),
    ], check=True, capture_output=True)
    return str(path)


def _tone_runs(path):
    """``[(freq, seconds), …]`` of consecutive 50 ms windows with the same dominant tone."""
    pcm = np.frombuffer(subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(_SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout, np.float32)
    win = _SR // 20
    runs = []
    for i in range(0, len(pcm) - win + 1, win):
        freq = int(round(np.argmax(np.abs(np.fft.rfft(pcm[i:i + win]))) * _SR / win / 10) * 10)
        if runs and runs[-1][0] == freq:
            runs[-1][1] += 0.05
        else:
            runs.append([freq, 0.05])
    return [(f, round(d, 2)) for f, d in runs]


def _codec(path):
    return subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name",
         "-of", "csv=p=0", path], capture_output=True, text=True, check=True,
    ).stdout.strip()


@needs_ffmpeg
def test_stream_copy_simple_cut_is_exact(aac_episode, tmp_path):
    out = str(tmp_path / "out.m4a")
    with patch("tvshow_noisemaker.processor._build_simple_cmd") as encode:
        process_file(aac_episode, out, intro_end=40.0, outro_start=100.0, output_format="m4a",
                     metadata={"title": "T"})
    encode.assert_not_called()  # took the copy path
    runs = _tone_runs(out)
    assert runs[0][0] == 1320 and abs(runs[0][1] - 60.0) <= 0.1
    assert sum(d for _, d in runs) == pytest.approx(60.0, abs=0.1)
    assert _codec(out) == "aac"


@needs_ffmpeg
def test_stream_copy_keeps_cold_open(aac_episode, tmp_path):
    out = str(tmp_path / "out.m4a")
    process_file(aac_episode, out, intro_start=10.0, intro_end=40.0, outro_start=100.0, output_format="m4a")
    runs = [(f, d) for f, d in _tone_runs(out) if d > 0.1]  # ignore the splice window
    assert [f for f, _ in runs] == [440, 1320]
    assert runs[0][1] == pytest.approx(10.0, abs=0.1)
    assert runs[1][1] == pytest.approx(60.0, abs=0.15)


@needs_ffmpeg
def test_stream_copy_embeds_artwork(aac_episode, tmp_path):
    art = tmp_path / "cover.jpg"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x64",
                    "-frames:v", "1", str(art)], check=True)
    out = str(tmp_path / "out.m4a")
    process_file(aac_episode, out, intro_end=40.0, outro_start=100.0, output_format="m4a",
                 artwork_path=str(art))
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
                              "-of", "csv=p=0", out], capture_output=True, text=True).stdout.split()
    assert "aac" in streams and "mjpeg" in streams


@needs_ffmpeg
def test_concat_encode_keeps_full_content_with_read_limit(aac_episode, tmp_path):
    out = str(tmp_path / "out.mp3")
    process_file(aac_episode, out, intro_start=10.0, intro_end=40.0, outro_start=100.0, output_format="mp3")
    runs = [(f, d) for f, d in _tone_runs(out) if d > 0.1]
    assert [f for f, _ in runs] == [440, 1320]
    assert runs[1][1] == pytest.approx(60.0, abs=0.15)  # content not cut short by -t
