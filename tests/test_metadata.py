"""Tests for tvshow_noisemaker.metadata."""

from tvshow_noisemaker.metadata import infer_show_name, infer_track_metadata


def test_infer_show_name_from_season_folder():
    show_name = infer_show_name("/shows/The Office/Season 2")
    assert show_name == "The Office"


def test_infer_show_name_from_plain_folder():
    show_name = infer_show_name("/shows/brooklyn_nine_nine")
    assert show_name == "Brooklyn Nine Nine"


def test_infer_track_metadata_from_standard_episode_pattern():
    metadata = infer_track_metadata(
        "/shows/The Office/Season 2/The.Office.S02E03.Office.Olympics.1080p.WEB-DL.mkv",
        "/shows/The Office/Season 2",
    )

    assert metadata == {
        "artist": "The Office",
        "album": "Season 2",
        "title": "Episode 03 - Office Olympics",
        "track": "3",
        "genre": "TV Show",
    }


def test_infer_track_metadata_without_episode_title():
    metadata = infer_track_metadata(
        "/shows/Severance/Season 1/Severance.S01E09.mkv",
        "/shows/Severance/Season 1",
    )

    assert metadata == {
        "artist": "Severance",
        "album": "Season 1",
        "title": "Episode 09",
        "track": "9",
        "genre": "TV Show",
    }


def test_infer_track_metadata_with_episode_only_pattern():
    metadata = infer_track_metadata(
        "/shows/Futurama/Season 3/Futurama Episode 15 The Problem with Popplers.mp4",
        "/shows/Futurama/Season 3",
    )

    assert metadata == {
        "artist": "Futurama",
        "album": "Season 3",
        "title": "Episode 15 - The Problem with Popplers",
        "track": "15",
        "genre": "TV Show",
    }