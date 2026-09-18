import logging
import queue
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
        player._play_blocking("/v/ep01.mp4", 83.4)
    assert run.call_args[0][0][0] == "ffmpeg"
    assert popen.call_args[0][0][0] == "afplay"
    player.stop()
    afplay.terminate.assert_called_once()


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
