from tvshow_noisemaker.gui import (
    parse_total,
    is_file_result,
    output_subdir_for,
    basename_collisions,
    parse_chosen_folders,
    format_folder_progress,
    friendly_result_line,
)


def test_parse_total_matches_conversion_line():
    line = "2026-07-07 10:00:00 [INFO] tvshow_noisemaker.console: Starting conversion  (12 files, 11 to encode, 4 workers, output → /tmp/out)"
    assert parse_total(line) == 12


def test_parse_total_ignores_other_lines():
    assert parse_total("Processing episode01.mkv") is None
    assert parse_total("") is None


def test_is_file_result_done_skipped_failed():
    assert is_file_result("  ✓ Done in 3.2s → /tmp/out/ep01.mp3 (4.1 MB)")
    assert is_file_result("  ✗ Skipped – no boundaries resolved")
    assert is_file_result("  ✗ Failed – ep02.mkv: ffmpeg exited 1")
    assert not is_file_result("Processing episode01.mkv")
    # Summary-section listings use '•', not '✗' — they must not double-count.
    assert not is_file_result("  • ep03.mkv – no boundaries resolved")


def test_output_subdir_for_joins_basename():
    assert output_subdir_for("/videos/The Office", "/out") == "/out/The Office"


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


def test_friendly_result_line_condenses_done_to_filename():
    line = "  ✓ Done in 3.2s → /tmp/out/ep01.mp3 (4.1 MB)"
    assert friendly_result_line(line) == "✓ ep01.mp3"


def test_friendly_result_line_passes_through_skipped_and_failed():
    assert friendly_result_line("  ✗ Skipped – no boundaries resolved") == "✗ Skipped – no boundaries resolved"
    assert friendly_result_line("  ✗ Failed – ep02.mkv: ffmpeg exited 1") == "✗ Failed – ep02.mkv: ffmpeg exited 1"
