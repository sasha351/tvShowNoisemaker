import pytest

from tvshow_noisemaker.gui import (
    output_subdir_for,
    basename_collisions,
    parse_chosen_folders,
    format_folder_progress,
)
from tvshow_noisemaker.pipeline import EpisodePlan, Plan


def test_output_subdir_for_joins_basename():
    assert output_subdir_for("/videos/The Office", "/out") == "/out/The Office (Audio)"


def test_output_subdir_never_equals_source_folder():
    """Destination == source's parent must not write converts next to the videos."""
    assert output_subdir_for("/videos/The Office", "/videos") != "/videos/The Office"


def test_basename_collisions_detects_duplicates():
    assert basename_collisions(["/a/Show", "/b/Show", "/c/Other"]) == ["Show"]
    assert basename_collisions(["/a/Show1", "/b/Show2"]) == []


def test_parse_chosen_folders_splits_and_strips_trailing_slashes():
    output = "/videos/Show One/\n/videos/Show Two/\n"
    assert parse_chosen_folders(output) == ["/videos/Show One", "/videos/Show Two"]


def test_parse_chosen_folders_handles_single_and_empty():
    assert parse_chosen_folders("/videos/Show/\n") == ["/videos/Show"]
    assert parse_chosen_folders("") == []
    assert parse_chosen_folders("   \n") == []


def test_format_folder_progress_before_total_known():
    assert (
        format_folder_progress(1, 3, "Show", 0, 0)
        == "Folder 1/3 (Show): Detecting intro/outro…"
    )


def test_format_folder_progress_with_total():
    assert (
        format_folder_progress(2, 3, "Show", 4, 10)
        == "Folder 2/3 (Show): Converting 4/10…"
    )


def test_format_folder_progress_detecting_with_counts():
    assert format_folder_progress(1, 2, "Show", 3, 21, stage="fingerprint") == "Folder 1/2 (Show): Detecting 3/21…"


def _headless_app():
    ctk = pytest.importorskip("customtkinter")
    from tvshow_noisemaker.gui import App
    try:
        root = ctk.CTk()
    except Exception as exc:  # no display
        pytest.skip(f"no display: {exc}")
    root.withdraw()
    return root, App(root)


def test_app_routes_review_episodes_to_the_panel_and_approval_to_the_encode_queue(tmp_path):
    root, app = _headless_app()
    try:
        app.src_folders = [str(tmp_path)]
        app.dst.set(str(tmp_path / "out"))
        app._begin_batch(start_threads=False)
        plan = Plan(input_dir=str(tmp_path), episodes=[
            EpisodePlan(path=str(tmp_path / "a.mp4"), duration=1300.0, intro_start=0.0, intro_end=30.0, outro_start=1260.0),
            EpisodePlan(path=str(tmp_path / "b.mp4"), duration=1300.0, intro_start=0.0, intro_end=None, outro_start=1260.0,
                        confidence="review", flags=["no intro match (season intro 30 s)"]),
        ])
        app._handle(("plan", str(tmp_path), plan))
        first_job = app.encode_jobs.get_nowait()
        assert [ep.basename for ep in first_job[1].episodes] == ["a.mp4"]
        assert len(app.review_rows) == 1
        app.review_rows[0].entries["intro_end"].insert(0, "0:31.0")
        app._approve(app.review_rows[0])
        approved_job = app.encode_jobs.get_nowait()
        ep = approved_job[1].episodes[0]
        assert (ep.basename, ep.intro_end, ep.source, ep.confidence) == ("b.mp4", 31.0, "manual", "high")
        assert app.review_rows == []
    finally:
        root.destroy()
