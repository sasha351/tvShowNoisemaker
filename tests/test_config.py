"""Tests for tvshow_noisemaker.config."""

import json
import pytest
from pathlib import Path

from tvshow_noisemaker.config import load_config, build_config, Config


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_none_returns_empty():
    assert load_config(None) == {}


def test_load_config_valid_file(tmp_path):
    cfg_file = tmp_path / "config.json"
    data = {
        "intro_duration": 30.5,
        "outro_duration": 25.0,
        "mp3_bitrate": 128,
        "timestamps": {
            "ep01.mp4": {"intro_end": 32.0, "outro_start": 2580.0}
        },
        "silence_db": -45.0,
        "min_silence_len": 1.0,
        "fingerprint_search_window": 180.0,
        "cut_margin": 0.8,
    }
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    result = load_config(str(cfg_file))
    assert result["intro_duration"] == 30.5
    assert result["outro_duration"] == 25.0
    assert result["mp3_bitrate"] == 128
    assert result["timestamps"]["ep01.mp4"]["intro_end"] == 32.0
    assert result["silence_db"] == -45.0
    assert result["fingerprint_search_window"] == 180.0
    assert result["cut_margin"] == 0.8


def test_load_config_missing_file():
    with pytest.raises(ValueError, match="not found"):
        load_config("/nonexistent/path/config.json")


def test_load_config_invalid_json(tmp_path):
    cfg_file = tmp_path / "bad.json"
    cfg_file.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(str(cfg_file))


def test_load_config_not_object(tmp_path):
    cfg_file = tmp_path / "array.json"
    cfg_file.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_config(str(cfg_file))


def test_load_config_invalid_float_type(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"intro_duration": "thirty"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a number"):
        load_config(str(cfg_file))


def test_load_config_invalid_bitrate(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"mp3_bitrate": -5}), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        load_config(str(cfg_file))


def test_load_config_invalid_cut_margin(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"cut_margin": -0.1}), encoding="utf-8")
    with pytest.raises(ValueError, match="cut_margin"):
        load_config(str(cfg_file))


def test_load_config_valid_output_format(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"output_format": "flac"}), encoding="utf-8")
    result = load_config(str(cfg_file))
    assert result["output_format"] == "flac"


def test_load_config_invalid_output_format(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"output_format": "ogg"}), encoding="utf-8")
    with pytest.raises(ValueError, match="output_format"):
        load_config(str(cfg_file))


def test_load_config_invalid_timestamps_not_object(tmp_path):
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"timestamps": "bad"}), encoding="utf-8")
    with pytest.raises(ValueError, match="timestamps.*must be"):
        load_config(str(cfg_file))


def test_load_config_unknown_keys_warn(tmp_path, caplog):
    import logging
    cfg_file = tmp_path / "cfg.json"
    cfg_file.write_text(json.dumps({"unknown_key": 1}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="tvshow_noisemaker.config"):
        result = load_config(str(cfg_file))
    assert "unknown_key" in caplog.text
    assert result == {}


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------


class _FakeArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_build_config_defaults():
    args = _FakeArgs(
        intro_duration=None,
        outro_duration=None,
        mp3_bitrate=None,
        silence_db=None,
        min_silence_len=None,
        fingerprint_search_window=None,
        fingerprint=True,
        log_file=None,
    )
    cfg = build_config(args, {})
    assert cfg.mp3_bitrate == 256
    assert cfg.silence_db == -40.0
    assert cfg.min_silence_len == 0.5
    assert cfg.fingerprint_search_window == 300.0
    assert cfg.cut_margin == 0.5
    assert cfg.use_fingerprinting is True
    assert cfg.intro_duration is None
    assert cfg.outro_duration is None


def test_build_config_cli_overrides_file():
    args = _FakeArgs(
        intro_duration=10.0,
        outro_duration=20.0,
        mp3_bitrate=320,
        silence_db=None,
        min_silence_len=None,
        fingerprint_search_window=None,
        cut_margin=0.2,
        fingerprint=False,
        log_file="out.log",
    )
    file_data = {"intro_duration": 99.0, "mp3_bitrate": 64, "cut_margin": 0.9}
    cfg = build_config(args, file_data)
    assert cfg.intro_duration == 10.0      # CLI wins
    assert cfg.outro_duration == 20.0
    assert cfg.mp3_bitrate == 320          # CLI wins
    assert cfg.cut_margin == 0.2           # CLI wins
    assert cfg.use_fingerprinting is False
    assert cfg.log_file == "out.log"


def test_build_config_negative_cut_margin_raises():
    args = _FakeArgs(
        intro_duration=None,
        outro_duration=None,
        mp3_bitrate=None,
        silence_db=None,
        min_silence_len=None,
        fingerprint_search_window=None,
        cut_margin=-1,
        fingerprint=True,
        log_file=None,
    )
    with pytest.raises(ValueError, match="cut_margin"):
        build_config(args, {})


def test_build_config_file_fills_missing():
    args = _FakeArgs(
        intro_duration=None,
        outro_duration=None,
        mp3_bitrate=None,
        silence_db=None,
        min_silence_len=None,
        fingerprint_search_window=None,
        fingerprint=True,
        log_file=None,
    )
    file_data = {"mp3_bitrate": 128, "intro_duration": 45.0}
    cfg = build_config(args, file_data)
    assert cfg.mp3_bitrate == 128
    assert cfg.intro_duration == 45.0


def test_stream_copy_defaults_on_and_file_or_cli_turns_it_off(tmp_path):
    assert build_config(_FakeArgs(), {}).stream_copy is True
    assert build_config(_FakeArgs(), {"stream_copy": False}).stream_copy is False
    assert build_config(_FakeArgs(stream_copy=False), {"stream_copy": True}).stream_copy is False
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"stream_copy": "no"}))
    with pytest.raises(ValueError, match="stream_copy"):
        load_config(str(cfg_file))


def test_mp3_quality_default_file_and_cli(tmp_path):
    assert build_config(_FakeArgs(), {}).mp3_quality == 5
    assert build_config(_FakeArgs(), {"mp3_quality": 7}).mp3_quality == 7
    assert build_config(_FakeArgs(mp3_quality=2), {"mp3_quality": 7}).mp3_quality == 2
    for bad in (10, -1, "5", True):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({"mp3_quality": bad}))
        with pytest.raises(ValueError, match="mp3_quality"):
            load_config(str(cfg_file))
