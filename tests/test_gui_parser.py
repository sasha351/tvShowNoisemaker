import pytest

from tvshow_noisemaker.gui import (
    output_subdir_for,
    basename_collisions,
    parse_chosen_folders,
    format_folder_progress,
)
from tvshow_noisemaker.pipeline import EpisodePlan, Plan, Summary


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
        app._end_batch()
        root.destroy()


# -- batch-scoped state (review fixes 1-4, 7) ---------------------------------


def _started(tmp_path):
    root, app = _headless_app()
    app.src_folders = [str(tmp_path)]
    app.dst.set(str(tmp_path / "out"))
    app._begin_batch(start_threads=False)
    return root, app


def _plan(tmp_path, *, now=0, review=0):
    eps = [
        EpisodePlan(path=str(tmp_path / f"n{i}.mp4"), duration=1300.0,
                    intro_start=0.0, intro_end=30.0, outro_start=1260.0)
        for i in range(now)
    ] + [
        EpisodePlan(path=str(tmp_path / f"r{i}.mp4"), duration=1300.0,
                    intro_start=0.0, intro_end=None, outro_start=1260.0,
                    confidence="review", flags=["no intro match (season intro 30 s)"])
        for i in range(review)
    ]
    return Plan(input_dir=str(tmp_path), episodes=eps)


def test_a_message_from_an_older_batch_is_dropped(tmp_path):
    """A detect worker outliving its batch must not poison the batch that replaced it."""
    root, app = _started(tmp_path)
    try:
        app.q.put((app._batch - 1, ("plan", str(tmp_path), _plan(tmp_path, now=1, review=1))))
        app.q.put((app._batch - 1, ("detect_done",)))
        app._poll()
        assert app.review_rows == []
        assert app.encode_jobs.empty()
    finally:
        app._end_batch()
        root.destroy()


def test_stop_sentinel_waits_for_the_last_approved_job(tmp_path):
    from tvshow_noisemaker.gui import _STOP
    root, app = _started(tmp_path)
    try:
        app._handle(("plan", str(tmp_path), _plan(tmp_path, review=1)))
        app._handle(("detect_done",))
        assert app.encode_jobs.empty()  # a row still needs review — do not stop the worker
        app.review_rows[0].entries["intro_end"].delete(0, "end")
        app.review_rows[0].entries["intro_end"].insert(0, "0:31.0")
        assert app._approve(app.review_rows[0]) is True
        job = app.encode_jobs.get_nowait()
        assert job is not _STOP and job[1].episodes[0].basename == "r0.mp4"
        assert app.encode_jobs.empty()  # job still in flight
        app._handle(("encoded", str(tmp_path), Summary(converted=["r0.mp4"])))
        assert app.encode_jobs.get_nowait() is _STOP
    finally:
        app._end_batch()
        root.destroy()


def test_skip_ticked_approval_still_enqueues_but_is_not_counted_in_the_eta(tmp_path):
    from tvshow_noisemaker.gui import _STOP
    root, app = _started(tmp_path)
    try:
        app._handle(("plan", str(tmp_path), _plan(tmp_path, review=1)))
        app._handle(("detect_done",))
        app.review_rows[0].skip_var.set(True)
        assert app._approve(app.review_rows[0]) is True
        assert app._enc_total == 0  # skipped episodes are not converted, so not in the ETA
        job = app.encode_jobs.get_nowait()
        assert job is not _STOP and job[1].episodes[0].confidence == "skip"
        assert app.encode_jobs.empty()
        app._handle(("encoded", str(tmp_path), Summary(skipped=[("r0.mp4", "user")])))
        assert app.encode_jobs.get_nowait() is _STOP
    finally:
        app._end_batch()
        root.destroy()


def test_failed_detection_on_the_only_folder_stops_the_worker(tmp_path):
    from tvshow_noisemaker.gui import _STOP
    root, app = _started(tmp_path)
    try:
        app._handle(("detect_failed", str(tmp_path), "ffprobe exploded"))
        app._handle(("detect_done",))
        assert app.encode_jobs.get_nowait() is _STOP
    finally:
        app._end_batch()
        root.destroy()


def test_messages_arriving_after_cancel_are_ignored(tmp_path):
    root, app = _started(tmp_path)
    try:
        app._cancel()
        app._handle(("plan", str(tmp_path), _plan(tmp_path, now=1, review=1)))
        app._handle(("detect_done",))
        assert app.review_rows == []
        assert app.encode_jobs.empty()
        assert app._review_shown is False
    finally:
        app._end_batch()
        root.destroy()


def test_closing_removes_the_log_handler_and_restores_the_level(tmp_path):
    import logging
    from tvshow_noisemaker.review import QueueHandler

    lib_log = logging.getLogger("tvshow_noisemaker")
    before = lib_log.level
    root, app = _started(tmp_path)
    assert any(isinstance(h, QueueHandler) for h in lib_log.handlers)
    app._on_close()  # destroys the root
    assert not any(isinstance(h, QueueHandler) for h in lib_log.handlers)
    assert lib_log.level == before


def test_encode_failure_reason_reaches_the_activity_feed(tmp_path):
    root, app = _started(tmp_path)
    try:
        app._handle(("encoded", str(tmp_path), Summary(failed=[("a.mkv", "ffmpeg died")])))
        assert "ffmpeg died" in app.activity.get("1.0", "end")
    finally:
        app._end_batch()
        root.destroy()


def test_the_encode_worker_exits_only_when_the_tk_thread_sends_stop():
    """No Tk needed: the worker's whole exit condition is the sentinel."""
    import threading
    from tvshow_noisemaker.gui import App, _Batch, _STOP

    b = _Batch(id=7)
    worker = threading.Thread(target=App._encode_worker, args=(b,), daemon=True)
    worker.start()
    worker.join(timeout=0.2)
    assert worker.is_alive()  # idle queue, but the batch is not over until Tk says so
    b.jobs.put(_STOP)
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert b.q.get(timeout=1) == (7, ("idle",))
