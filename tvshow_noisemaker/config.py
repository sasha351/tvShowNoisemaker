"""Configuration loader for tvshow_noisemaker.

Supports a JSON config file with the following optional keys:

    {
        "intro_duration": 30.5,
        "outro_duration": 25.0,
        "timestamps": {
            "episode_s01e01.mp4": {"intro_end": 30.0, "outro_start": 2580.0}
        },
        "mp3_bitrate": 256,
        "output_format": "mp3",
        "silence_db": -40.0,
        "min_silence_len": 0.5,
        "fingerprint_search_window": 300.0,
        "cut_margin": 0.5
    }

CLI flags always override config file values.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Keys accepted in the config file
_VALID_KEYS = {
    "intro_duration",
    "outro_duration",
    "timestamps",
    "mp3_bitrate",
    "output_format",
    "silence_db",
    "min_silence_len",
    "fingerprint_search_window",
    "cut_margin",
}

VALID_OUTPUT_FORMATS = {"mp3", "m4a", "flac", "wav"}


@dataclass
class Config:
    """Holds all runtime configuration options."""

    intro_duration: Optional[float] = None
    outro_duration: Optional[float] = None
    # timestamps: mapping of filename (basename) → {"intro_end": float, "outro_start": float}
    timestamps: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mp3_bitrate: int = 256
    output_format: str = "mp3"
    silence_db: float = -40.0
    min_silence_len: float = 0.5
    fingerprint_search_window: float = 300.0
    cut_margin: float = 0.5
    use_fingerprinting: bool = True
    log_file: Optional[str] = None
    workers: Optional[int] = None


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load and validate a JSON config file.

    Returns a dict with only the recognised keys.  Returns an empty dict if
    *config_path* is ``None``.

    Raises ``ValueError`` if the file cannot be parsed or contains invalid
    values.
    """
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"Config file not found: {config_path}")

    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("Config file must be a JSON object at the top level")

    unknown = set(raw.keys()) - _VALID_KEYS
    if unknown:
        logger.warning("Unknown config keys (ignored): %s", ", ".join(sorted(unknown)))

    validated: Dict[str, Any] = {}

    for float_key in ("intro_duration", "outro_duration", "silence_db",
                      "min_silence_len", "fingerprint_search_window", "cut_margin"):
        if float_key in raw:
            val = raw[float_key]
            if not isinstance(val, (int, float)):
                raise ValueError(f"Config key '{float_key}' must be a number, got {type(val).__name__}")
            validated[float_key] = float(val)

    if "cut_margin" in validated and validated["cut_margin"] < 0:
        raise ValueError("Config key 'cut_margin' must be >= 0")

    if "mp3_bitrate" in raw:
        val = raw["mp3_bitrate"]
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"Config key 'mp3_bitrate' must be a positive integer, got {val!r}")
        validated["mp3_bitrate"] = val

    if "output_format" in raw:
        val = raw["output_format"]
        if val not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Config key 'output_format' must be one of "
                f"{sorted(VALID_OUTPUT_FORMATS)}, got {val!r}"
            )
        validated["output_format"] = val

    if "timestamps" in raw:
        ts = raw["timestamps"]
        if not isinstance(ts, dict):
            raise ValueError("Config key 'timestamps' must be a JSON object")
        validated["timestamps"] = _validate_timestamps(ts)

    return validated


def _validate_timestamps(ts: Any) -> Dict[str, Dict[str, float]]:
    """Validate the ``timestamps`` mapping from the config file."""
    result: Dict[str, Dict[str, float]] = {}
    for filename, entry in ts.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"timestamps['{filename}'] must be a JSON object with "
                f"'intro_end' and/or 'outro_start' keys"
            )
        cleaned: Dict[str, float] = {}
        for key in ("intro_end", "outro_start"):
            if key in entry:
                val = entry[key]
                if not isinstance(val, (int, float)):
                    raise ValueError(
                        f"timestamps['{filename}']['{key}'] must be a number"
                    )
                cleaned[key] = float(val)
        result[filename] = cleaned
    return result


def build_config(args: Any, config_file_data: Dict[str, Any]) -> Config:
    """Merge CLI *args* (from argparse) with *config_file_data*.

    CLI flags take precedence over config-file values.
    """
    def _pick(cli_val: Any, file_key: str, default: Any) -> Any:
        if cli_val is not None:
            return cli_val
        return config_file_data.get(file_key, default)

    intro_duration = _pick(getattr(args, "intro_duration", None), "intro_duration", None)
    outro_duration = _pick(getattr(args, "outro_duration", None), "outro_duration", None)
    mp3_bitrate = _pick(getattr(args, "mp3_bitrate", None), "mp3_bitrate", 256)
    output_format = _pick(getattr(args, "output_format", None), "output_format", "mp3")
    silence_db = _pick(getattr(args, "silence_db", None), "silence_db", -40.0)
    min_silence_len = _pick(getattr(args, "min_silence_len", None), "min_silence_len", 0.5)
    fingerprint_search_window = _pick(
        getattr(args, "fingerprint_search_window", None),
        "fingerprint_search_window",
        300.0,
    )
    cut_margin = float(_pick(getattr(args, "cut_margin", None), "cut_margin", 0.5))
    if cut_margin < 0:
        raise ValueError("cut_margin must be >= 0")
    timestamps = config_file_data.get("timestamps", {})

    workers_raw = getattr(args, "workers", None)
    workers = int(workers_raw) if workers_raw is not None else None

    return Config(
        intro_duration=intro_duration,
        outro_duration=outro_duration,
        timestamps=timestamps,
        mp3_bitrate=int(mp3_bitrate),
        output_format=output_format,
        silence_db=float(silence_db),
        min_silence_len=float(min_silence_len),
        fingerprint_search_window=float(fingerprint_search_window),
        cut_margin=float(cut_margin),
        use_fingerprinting=getattr(args, "fingerprint", True),
        log_file=getattr(args, "log_file", None),
        workers=workers,
    )
