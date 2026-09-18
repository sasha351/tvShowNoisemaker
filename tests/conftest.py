"""Shared pytest fixtures for the tvshow_noisemaker test suite."""

from __future__ import annotations

import pytest

from tvshow_noisemaker.processor import _aac_encoder


@pytest.fixture(autouse=True)
def _reset_aac_probe():
    _aac_encoder.cache_clear()
    yield
    _aac_encoder.cache_clear()
