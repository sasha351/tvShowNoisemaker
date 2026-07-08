from tvshow_noisemaker.gui import parse_total, is_file_result


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
