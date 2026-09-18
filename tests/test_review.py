import logging
import queue
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from tvshow_noisemaker.pipeline import EpisodePlan, Plan
from tvshow_noisemaker.review import (
    QueueHandler, ReviewRow, SnippetPlayer, format_time, parse_time, snippet_cmd, split_plan,
)


@pytest.mark.parametrize("text, secs", [("1:23.4", 83.4), ("83.4", 83.4), ("0:01:23.4", 83.4), ("12:00", 720.0), (" 5 ", 5.0)])
def test_parse_time_accepts_clock_and_seconds(text, secs):
    assert parse_time(text) == pytest.approx(secs)


@pytest.mark.parametrize("text", ["", "abc", "1:2:3:4", "-5", "1:-3"])
def test_parse_time_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_format_time_round_trips():
    assert format_time(83.4) == "1:23.4"
    assert format_time(0.0) == "0:00.0"
    assert format_time(None) == ""
    assert parse_time(format_time(2650.25)) == pytest.approx(2650.3, abs=0.05)


def _ep(name="ep01.mp4", **kw):
    base = dict(path=f"/v/{name}", duration=1300.0, intro_start=0.0, intro_end=30.0, outro_start=1260.0)
    base.update(kw)
    return EpisodePlan(**base)


def test_split_plan_sends_review_and_optionally_everything_to_the_panel():
    plan = Plan(input_dir="/v", episodes=[_ep("a.mp4"), _ep("b.mp4", confidence="review", intro_end=None), _ep("c.mp4", confidence="skip")])
    now, review = split_plan(plan, review_all=False)
    assert [e.basename for e in now] == ["a.mp4", "c.mp4"]   # skips go straight to encode() so they are tallied
    assert [e.basename for e in review] == ["b.mp4"]
    now, review = split_plan(plan, review_all=True)
    assert now == [] and [e.basename for e in review] == ["a.mp4", "b.mp4", "c.mp4"]


def test_review_row_round_trip_marks_manual_only_when_edited():
    row = ReviewRow.from_episode(_ep(confidence="review", flags=["credits matched 1 episode only"]))
    assert (row.intro_start, row.intro_end, row.outro_start) == ("0:00.0", "0:30.0", "21:00.0")
    untouched = row.to_episode()
    assert (untouched.source, untouched.confidence, untouched.intro_end) == ("fingerprint", "high", 30.0)
    row.intro_end = "0:31.5"
    edited = row.to_episode()
    assert (edited.source, edited.confidence, edited.intro_end, edited.outro_start) == ("manual", "high", 31.5, 1260.0)
    assert edited.flags == ["credits matched 1 episode only"]


def test_review_row_untouched_field_keeps_original_precision():
    # format_time rounds to 0.1s, so an untouched 30.037 displays as "0:30.0"; that must
    # not re-parse to 30.0 and get flagged as a manual edit that drops the fingerprint's precision.
    row = ReviewRow.from_episode(_ep(intro_end=30.037, outro_start=1260.0, source="fingerprint"))
    untouched = row.to_episode()
    assert (untouched.intro_end, untouched.source, untouched.confidence) == (30.037, "fingerprint", "high")
    row.intro_end = "0:31.0"
    edited = row.to_episode()
    assert (edited.intro_end, edited.source, edited.outro_start) == (31.0, "manual", 1260.0)


def test_review_row_skip_and_blank_fields():
    row = ReviewRow.from_episode(_ep(confidence="review", intro_end=None))
    assert row.intro_end == ""
    ep = row.to_episode()
    assert ep.intro_end is None and ep.confidence == "high"      # still not encodable; the GUI disables Approve for it
    row.skip = True
    assert row.to_episode().confidence == "skip"


def test_snippet_cmd_spans_three_seconds_each_side():
    cmd = snippet_cmd("/v/ep01.mp4", 83.4, "/tmp/snip.wav")
    assert cmd[:5] == ["ffmpeg", "-y", "-ss", "80.4", "-t"]
    assert cmd[5] == "6.0" and "/v/ep01.mp4" in cmd and cmd[-1] == "/tmp/snip.wav"
    assert snippet_cmd("/v/ep01.mp4", 1.0, "/tmp/s.wav")[3] == "0.0"   # never seeks before the file start


def test_snippet_player_runs_ffmpeg_then_afplay_and_stop_terminates():
    player = SnippetPlayer()
    afplay = MagicMock()
    with patch("tvshow_noisemaker.review.subprocess.run") as run, \
         patch("tvshow_noisemaker.review.subprocess.Popen", return_value=afplay) as popen:
        player._play_blocking("/v/ep01.mp4", 83.4, player._gen)
    assert run.call_args[0][0][0] == "ffmpeg"
    assert popen.call_args[0][0][0] == "afplay"
    player.stop()
    afplay.terminate.assert_called_once()


def test_snippet_player_stop_during_ffmpeg_phase_prevents_late_afplay():
    # play()'s thread runs ffmpeg (subprocess.run) before spawning afplay; a stop() that
    # lands in that window must stop the thread from starting afplay at all.
    player = SnippetPlayer()
    player._gen = 1  # as if play() had already bumped it for this in-flight thread

    def run_then_stop(*args, **kwargs):
        player.stop()  # simulates a stop() call arriving while ffmpeg is still running

    with patch("tvshow_noisemaker.review.subprocess.run", side_effect=run_then_stop), \
         patch("tvshow_noisemaker.review.subprocess.Popen") as popen:
        player._play_blocking("/v/ep01.mp4", 83.4, 1)
    popen.assert_not_called()


def test_snippet_player_play_twice_only_second_generation_spawns():
    player = SnippetPlayer()
    first_ffmpeg_started = threading.Event()
    release_first_ffmpeg = threading.Event()
    run_calls = []

    def fake_run(*args, **kwargs):
        run_calls.append(args)
        if len(run_calls) == 1:
            first_ffmpeg_started.set()
            release_first_ffmpeg.wait(timeout=2)

    afplay = MagicMock()
    with patch("tvshow_noisemaker.review.subprocess.run", side_effect=fake_run), \
         patch("tvshow_noisemaker.review.subprocess.Popen", return_value=afplay) as popen:
        player.play("/v/ep01.mp4", 1.0)
        assert first_ffmpeg_started.wait(timeout=2)
        player.play("/v/ep01.mp4", 2.0)  # stop()s the first (still in ffmpeg, no proc yet) and bumps the generation
        release_first_ffmpeg.set()
        for _ in range(100):  # wait for both threads' subprocess.run to return
            if len(run_calls) >= 2:
                break
            time.sleep(0.02)
        time.sleep(0.1)  # let both threads finish their post-run gen check / Popen call, if any

    popen.assert_called_once()
    assert player._proc is afplay


def test_queue_handler_posts_formatted_lines():
    q = queue.Queue()
    handler = QueueHandler(q)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logging.getLogger("tvshow_noisemaker.test_review").addHandler(handler)
    try:
        logging.getLogger("tvshow_noisemaker.test_review").warning("hello %s", "world")
    finally:
        logging.getLogger("tvshow_noisemaker.test_review").removeHandler(handler)
    assert q.get_nowait() == ("log", "WARNING hello world")


def test_queue_handler_emit_swallows_format_errors():
    q = queue.Queue()
    handler = QueueHandler(q)
    record = logging.LogRecord("tvshow_noisemaker.test_review", logging.WARNING, __file__, 1, "%d", ("x",), None)
    with patch.object(handler, "handleError") as handle_error:
        handler.emit(record)  # "%d" % ("x",) raises TypeError inside self.format()
    handle_error.assert_called_once_with(record)
    assert q.empty()
