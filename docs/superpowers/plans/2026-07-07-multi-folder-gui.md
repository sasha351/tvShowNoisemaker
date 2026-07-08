# Bare-Bones Multi-Folder GUI for tvshow-noisemaker — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A double-clickable, dead-simple GUI: add one or more source folders (e.g. 4 seasons), pick one destination folder, hit Start, watch a per-folder progress bar for each — no terminal, no pasting paths.

**Architecture:** A single tkinter window (`tvshow_noisemaker/gui.py`) that manages a batch of jobs — one CLI subprocess (`python -m tvshow_noisemaker.cli`) per source folder, at most **2 running concurrently** (each CLI run already spawns its own ffmpeg worker pool, so 2 saturates the machine; the rest queue). Each job outputs to `<destination>/<source folder name>/` to keep seasons separated and avoid filename collisions. The GUI parses each subprocess's progress-bar output (`Progress [###...] 3/10 ok=2 skip=1 fail=0`) to drive one ttk.Progressbar per folder. The CLI pipeline (`cli.py:run`) is untouched. Last-used folders are remembered in `~/.tvshow_noisemaker_gui.json`. A minimal `.app` bundle in the repo root gives Finder double-click launch.

**Tech Stack:** Python 3.14 (existing `.venv`), tkinter (stdlib — requires one-time `brew install python-tk@3.14`), subprocess + threading + queue (stdlib), pytest.

## Context

The user runs `tvshow-noisemaker` from the terminal today and has to hunt for folder paths, paste them, and worry about quoting/formatting. They want: source picker(s), destination picker, progress view. Multiple season folders must be handled in one batch. Design decisions confirmed with the user: **capped parallelism (2 concurrent runs)** and **one output subfolder per season**.

Key facts discovered during exploration:
- The CLI takes exactly one `--input-dir` per run, and fingerprinting compares episodes *within* a folder — so N source folders = N separate CLI runs, which the GUI schedules.
- `tvshow_noisemaker/cli.py:651-676` (`_print_progress_bar`) emits `\rProgress [{bar}] {done}/{total} ok={p} skip={s} fail={f} elapsed=… eta=… rate=…` — machine-parseable. Lines are `\r`-terminated (not `\n`), so the GUI reader must split on both `\r` and `\n`.
- The CLI writes a log file into its output dir and prints a summary via logging (flushed per record, so subprocess piping works). `-u` on the child Python avoids block buffering.
- The venv at `.venv/` is Python 3.14.4 (Homebrew). **`import tkinter` currently fails** (`No module named '_tkinter'`) — fixed by `brew install python-tk@3.14`; venvs share the base interpreter's stdlib so no venv rebuild is needed.
- Install shim `/Users/sasha/bin/tvshow-noisemaker` runs `"$VENV/bin/python3" -m tvshow_noisemaker.cli` with `PYTHONPATH=$SRC`. The `.app` launcher mirrors this pattern.
- The repo has `test_media/` for manual end-to-end verification (duplicate it to simulate multiple seasons).

## Global Constraints

- Python `>=3.9` (pyproject floor); code actually runs on the venv's 3.14.
- **No new pip dependencies** — stdlib only (tkinter, subprocess, threading, queue, json, re).
- Do not modify `tvshow_noisemaker/cli.py`, `detector.py`, `processor.py`, `metadata.py`, or `config.py`.
- GUI exposes ONLY: source folder list (Add/Remove), destination folder, Start, Cancel All, per-folder progress, log. No option fields (YAGNI — the CLI keeps them).
- Max concurrent CLI subprocesses: **2** (module constant `MAX_CONCURRENT = 2`).
- All new code lives in `tvshow_noisemaker/gui.py`; tests in `tests/test_gui.py`.
- Run tests with `.venv/bin/python -m pytest` (never bare `pytest` — may resolve to another interpreter).

## File Structure

- Create: `tvshow_noisemaker/gui.py` — pure helpers (`build_command`, `output_dir_for`, `parse_progress_line`, `load_settings`, `save_settings`) + `Job` (one folder's run + its widget row) + `NoisemakerGUI` app class + `main()`.
- Create: `tests/test_gui.py` — unit tests for the pure helpers only (tkinter widget code is verified manually).
- Modify: `pyproject.toml` — add `[project.gui-scripts]` entry point.
- Create: `TV Show Noisemaker.app/Contents/Info.plist` and `TV Show Noisemaker.app/Contents/MacOS/tvshow-noisemaker-gui` (shell launcher) — double-clickable Finder app in the repo root.
- Modify: `README.md` — short "GUI" section.

---

### Task 1: Enable tkinter for the venv

**Files:** none (environment only — nothing to commit).

**Interfaces:**
- Produces: a working `import tkinter` inside `.venv/bin/python3`, required by every later task.

- [ ] **Step 1: Install the Tk bindings for Homebrew Python 3.14**

```bash
brew install python-tk@3.14
```

Expected: brew installs (or reports already installed). This drops `_tkinter` into the Homebrew Python's stdlib; the venv picks it up automatically because venvs share the base interpreter's stdlib.

- [ ] **Step 2: Verify tkinter imports inside the project venv**

```bash
"/Users/sasha/Documents/iPod Project/tvShowNoisemaker/.venv/bin/python3" -c "import tkinter; print('tkinter OK', tkinter.TkVersion)"
```

Expected: `tkinter OK 8.6` (or similar version). If this still fails, stop and report — nothing else in the plan can proceed.

---

### Task 2: Pure helpers — command builder, output layout, progress parser, settings (TDD)

**Files:**
- Create: `tvshow_noisemaker/gui.py` (helpers portion)
- Test: `tests/test_gui.py`

**Interfaces:**
- Consumes: the CLI progress-line format from `tvshow_noisemaker/cli.py:_print_progress_bar` (read-only dependency; not modified).
- Produces (used by Task 3):
  - `MAX_CONCURRENT: int = 2` (module constant)
  - `build_command(input_dir: str, output_dir: str) -> list[str]`
  - `output_dir_for(source_dir: str, dest_root: str) -> str` — returns `<dest_root>/<basename of source_dir>`
  - `parse_progress_line(line: str) -> Optional[dict]` — returns `{"done": int, "total": int, "ok": int, "skip": int, "fail": int}` or `None`
  - `load_settings() -> dict` / `save_settings(settings: dict) -> None` using module constant `SETTINGS_PATH`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui.py`:

```python
"""Tests for the pure helper functions in tvshow_noisemaker.gui."""

import sys

from tvshow_noisemaker import gui


def test_build_command_uses_current_interpreter_and_dirs():
    cmd = gui.build_command("/path/to/episodes", "/path/to/mp3s")
    assert cmd[0] == sys.executable
    assert "-u" in cmd
    assert "-m" in cmd and "tvshow_noisemaker.cli" in cmd
    i = cmd.index("--input-dir")
    assert cmd[i + 1] == "/path/to/episodes"
    o = cmd.index("--output-dir")
    assert cmd[o + 1] == "/path/to/mp3s"


def test_output_dir_for_appends_source_folder_name():
    assert gui.output_dir_for("/shows/Season 1", "/out") == "/out/Season 1"


def test_output_dir_for_ignores_trailing_slash():
    assert gui.output_dir_for("/shows/Season 1/", "/out") == "/out/Season 1"


def test_parse_progress_line_extracts_counters():
    line = (
        "\rProgress [######------------------------] 3/10 "
        "ok=2 skip=1 fail=0 elapsed=5s eta=12s rate=0.60/s"
    )
    assert gui.parse_progress_line(line) == {
        "done": 3, "total": 10, "ok": 2, "skip": 1, "fail": 0,
    }


def test_parse_progress_line_ignores_normal_log_lines():
    assert gui.parse_progress_line("Found 3 video file(s) in /tmp/in") is None
    assert gui.parse_progress_line("") is None


def test_settings_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "SETTINGS_PATH", tmp_path / "settings.json")
    gui.save_settings({"input_dirs": ["/a", "/b"], "output_dir": "/c"})
    assert gui.load_settings() == {"input_dirs": ["/a", "/b"], "output_dir": "/c"}


def test_load_settings_missing_or_corrupt_returns_empty(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(gui, "SETTINGS_PATH", settings_file)
    assert gui.load_settings() == {}          # file missing
    settings_file.write_text("{not json", encoding="utf-8")
    assert gui.load_settings() == {}          # corrupt file
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/sasha/Documents/iPod Project/tvShowNoisemaker" && .venv/bin/python -m pytest tests/test_gui.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'tvshow_noisemaker.gui'` (import error counts as the failing state).

- [ ] **Step 3: Write the helpers**

Create `tvshow_noisemaker/gui.py`:

```python
"""Bare-bones tkinter GUI for tvshow-noisemaker.

Add one or more source folders (one per TV season), pick a destination
folder, hit Start. Each source folder becomes its own CLI subprocess run
(at most MAX_CONCURRENT at a time), outputting MP3s to
<destination>/<source folder name>/. The processing pipeline itself is
untouched — this module only schedules CLI runs and renders progress.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

SETTINGS_PATH = Path.home() / ".tvshow_noisemaker_gui.json"

# Each CLI run spawns its own ffmpeg worker pool (~half the CPU cores),
# so two concurrent runs saturate the machine.
MAX_CONCURRENT = 2

# Matches cli._print_progress_bar output, e.g.
# "Progress [######----] 3/10 ok=2 skip=1 fail=0 elapsed=5s eta=12s rate=0.60/s"
_PROGRESS_RE = re.compile(
    r"Progress \[[#-]*\] (\d+)/(\d+) ok=(\d+) skip=(\d+) fail=(\d+)"
)


def build_command(input_dir: str, output_dir: str) -> List[str]:
    """Build the CLI invocation for one source/destination pair."""
    return [
        sys.executable, "-u", "-m", "tvshow_noisemaker.cli",
        "--input-dir", input_dir,
        "--output-dir", output_dir,
    ]


def output_dir_for(source_dir: str, dest_root: str) -> str:
    """Per-source output folder: <dest_root>/<basename of source_dir>."""
    return str(Path(dest_root) / Path(source_dir).name)


def parse_progress_line(line: str) -> Optional[dict]:
    """Extract progress counters from a CLI progress-bar line, or None."""
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    done, total, ok, skip, fail = (int(g) for g in m.groups())
    return {"done": done, "total": total, "ok": ok, "skip": skip, "fail": fail}


def load_settings() -> dict:
    """Load remembered folders; empty dict if missing or unreadable."""
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    """Persist remembered folders; best-effort, never raises."""
    try:
        SETTINGS_PATH.write_text(json.dumps(settings), encoding="utf-8")
    except OSError:
        pass
```

(Note: no tkinter import yet — that arrives in Task 3, keeping this task green even if Tk breaks.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/sasha/Documents/iPod Project/tvShowNoisemaker" && .venv/bin/python -m pytest tests/test_gui.py -v`
Expected: 7 passed.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `cd "/Users/sasha/Documents/iPod Project/tvShowNoisemaker" && .venv/bin/python -m pytest -q`
Expected: all tests pass (same pass count as before this task, plus 7).

- [ ] **Step 6: Commit**

```bash
git add tvshow_noisemaker/gui.py tests/test_gui.py
git commit -m "feat: add GUI helper functions (command builder, output layout, progress parser, settings)"
```

---

### Task 3: The tkinter window and capped-parallel job runner

**Files:**
- Modify: `tvshow_noisemaker/gui.py` (append the `Job` and `NoisemakerGUI` classes and `main()` below the helpers from Task 2)

**Interfaces:**
- Consumes: `MAX_CONCURRENT`, `build_command`, `output_dir_for`, `parse_progress_line`, `load_settings`, `save_settings` from Task 2 (exact signatures above).
- Produces: `main() -> None` — the entry point Task 4 wires into pyproject and the `.app` launcher.

This is UI + threading glue; it is verified manually (Step 3), not unit-tested. The threading pattern: one reader thread per running subprocess pumps `("line", job_index, text)` / `("exit", job_index, returncode)` tuples into a single shared `queue.Queue`; the Tk main thread drains the queue every 100 ms via `root.after` (tkinter widgets must only be touched from the main thread) and launches queued jobs as slots free up.

- [ ] **Step 1: Append the imports, Job class, app class, and main() to `tvshow_noisemaker/gui.py`**

Add these imports at the top of the file (merge with the existing import block):

```python
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
```

Append below the helpers:

```python
class Job:
    """One source folder's conversion run plus its row of widgets.

    States: queued → running → done | failed | cancelled.
    """

    TERMINAL = {"done", "failed", "cancelled"}

    def __init__(self, source: str, dest_root: str,
                 parent: ttk.Frame, row: int) -> None:
        self.source = source
        self.output_dir = output_dir_for(source, dest_root)
        self.name = Path(source).name
        self.proc: Optional[subprocess.Popen] = None
        self.state = "queued"

        ttk.Label(parent, text=self.name, width=22, anchor="w").grid(
            row=row, column=0, sticky="w", padx=4, pady=2)
        self.bar = ttk.Progressbar(parent, mode="determinate", maximum=1)
        self.bar.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        self.status = tk.StringVar(value="Queued")
        ttk.Label(parent, textvariable=self.status, width=30, anchor="w").grid(
            row=row, column=2, sticky="w", padx=4, pady=2)


class NoisemakerGUI:
    """Single window: source list, destination, Start/Cancel, per-job rows, log."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TV Show Noisemaker")
        self.root.minsize(680, 560)

        self.jobs: List[Job] = []
        self.queue: "queue.Queue[tuple]" = queue.Queue()
        self.polling = False

        settings = load_settings()
        self.dst_var = tk.StringVar(value=settings.get("output_dir", ""))

        pad = {"padx": 8, "pady": 4}
        frame = ttk.Frame(root)
        frame.pack(fill="both", expand=True, **pad)

        # Source folders list
        src_frame = ttk.LabelFrame(frame, text="Source folders (one per season)")
        src_frame.pack(fill="x", **pad)
        list_wrap = ttk.Frame(src_frame)
        list_wrap.pack(side="left", fill="both", expand=True, **pad)
        self.src_list = tk.Listbox(list_wrap, height=5, selectmode="extended")
        src_scroll = ttk.Scrollbar(list_wrap, command=self.src_list.yview)
        self.src_list.configure(yscrollcommand=src_scroll.set)
        src_scroll.pack(side="right", fill="y")
        self.src_list.pack(side="left", fill="both", expand=True)
        for d in settings.get("input_dirs", []):
            self.src_list.insert("end", d)
        btns = ttk.Frame(src_frame)
        btns.pack(side="left", fill="y", **pad)
        ttk.Button(btns, text="Add Folder…", command=self._add_source).pack(
            fill="x", pady=2)
        ttk.Button(btns, text="Remove Selected",
                   command=self._remove_sources).pack(fill="x", pady=2)

        # Destination row
        row2 = ttk.Frame(frame)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Destination folder:", width=16).pack(side="left")
        ttk.Entry(row2, textvariable=self.dst_var).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row2, text="Browse…", command=self._pick_dest).pack(side="left")

        # Buttons + overall status
        row3 = ttk.Frame(frame)
        row3.pack(fill="x", **pad)
        self.start_btn = ttk.Button(row3, text="Start", command=self._start)
        self.start_btn.pack(side="left")
        self.cancel_btn = ttk.Button(
            row3, text="Cancel All", command=self._cancel_all, state="disabled")
        self.cancel_btn.pack(side="left", padx=8)
        self.overall_var = tk.StringVar(value="Idle")
        ttk.Label(row3, textvariable=self.overall_var).pack(side="left", padx=8)

        # Per-job progress rows
        self.jobs_frame = ttk.LabelFrame(frame, text="Progress")
        self.jobs_frame.pack(fill="x", **pad)
        self.jobs_frame.columnconfigure(1, weight=1)

        # Log pane
        log_frame = ttk.Frame(frame)
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(log_frame, height=10, wrap="none", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI callbacks ────────────────────────────────────────────────

    def _add_source(self) -> None:
        chosen = filedialog.askdirectory(initialdir=str(Path.home()))
        if chosen and chosen not in self.src_list.get(0, "end"):
            self.src_list.insert("end", chosen)

    def _remove_sources(self) -> None:
        for idx in reversed(self.src_list.curselection()):
            self.src_list.delete(idx)

    def _pick_dest(self) -> None:
        chosen = filedialog.askdirectory(
            initialdir=self.dst_var.get() or str(Path.home()))
        if chosen:
            self.dst_var.set(chosen)

    def _start(self) -> None:
        sources = list(self.src_list.get(0, "end"))
        dst = self.dst_var.get().strip()
        if not sources:
            messagebox.showerror("TV Show Noisemaker",
                                 "Add at least one source folder.")
            return
        bad = [s for s in sources if not Path(s).is_dir()]
        if bad:
            messagebox.showerror("TV Show Noisemaker",
                                 "Not a folder:\n" + "\n".join(bad))
            return
        if not dst:
            messagebox.showerror("TV Show Noisemaker",
                                 "Please choose a destination folder.")
            return
        names = [Path(s).name for s in sources]
        if len(set(names)) != len(names):
            messagebox.showerror(
                "TV Show Noisemaker",
                "Two source folders have the same name — their output "
                "would collide. Rename one and try again.")
            return

        save_settings({"input_dirs": sources, "output_dir": dst})

        # Reset job rows from any previous batch
        for child in self.jobs_frame.winfo_children():
            child.destroy()
        self.jobs = [Job(src, dst, self.jobs_frame, i)
                     for i, src in enumerate(sources)]

        self._append_log(
            f"Batch of {len(self.jobs)} folder(s) → {dst} "
            f"(max {MAX_CONCURRENT} at a time)\n")
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self._fill_slots()
        self._update_overall()
        if not self.polling:
            self.polling = True
            self.root.after(100, self._poll)

    def _cancel_all(self) -> None:
        for job in self.jobs:
            if job.state == "queued":
                job.state = "cancelled"
                job.status.set("Cancelled")
            elif job.state == "running" and job.proc and job.proc.poll() is None:
                job.proc.terminate()
                job.status.set("Cancelling…")
        self._update_overall()

    def _on_close(self) -> None:
        if any(j.state == "running" for j in self.jobs):
            if not messagebox.askyesno(
                    "TV Show Noisemaker",
                    "Conversions are running. Stop them and quit?"):
                return
            self._cancel_all()
        self.root.destroy()

    # ── Job scheduling / subprocess plumbing ────────────────────────

    def _fill_slots(self) -> None:
        """Launch queued jobs until MAX_CONCURRENT are running."""
        running = sum(1 for j in self.jobs if j.state == "running")
        for idx, job in enumerate(self.jobs):
            if running >= MAX_CONCURRENT:
                break
            if job.state == "queued":
                self._launch(idx)
                running += 1

    def _launch(self, idx: int) -> None:
        job = self.jobs[idx]
        job.state = "running"
        job.status.set("Starting…")
        self._append_log(f"[{job.name}] Starting → {job.output_dir}\n")
        job.proc = subprocess.Popen(
            build_command(job.source, job.output_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(target=self._reader, args=(idx,), daemon=True).start()

    def _reader(self, idx: int) -> None:
        """Worker thread: split one child's output on \\r and \\n."""
        proc = self.jobs[idx].proc
        assert proc is not None and proc.stdout is not None
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if ch == "":
                break
            if ch in "\r\n":
                if buf.strip():
                    self.queue.put(("line", idx, buf))
                buf = ""
            else:
                buf += ch
        if buf.strip():
            self.queue.put(("line", idx, buf))
        proc.wait()
        self.queue.put(("exit", idx, proc.returncode))

    def _poll(self) -> None:
        """Tk main thread, every 100 ms: drain queue, refill slots."""
        try:
            while True:
                kind, idx, payload = self.queue.get_nowait()
                job = self.jobs[idx]
                if kind == "line":
                    p = parse_progress_line(payload)
                    if p:
                        job.bar.configure(
                            maximum=max(p["total"], 1), value=p["done"])
                        job.status.set(
                            f"{p['done']}/{p['total']}  ✓{p['ok']} "
                            f"skip {p['skip']} fail {p['fail']}")
                    else:
                        self._append_log(f"[{job.name}] {payload}\n")
                else:  # "exit"
                    returncode = payload
                    if returncode == 0:
                        job.state = "done"
                        job.status.set("Done ✓")
                    elif returncode < 0:
                        job.state = "cancelled"
                        job.status.set("Cancelled")
                    else:
                        job.state = "failed"
                        job.status.set("Finished with errors — see log")
                    self._fill_slots()
                    self._update_overall()
        except queue.Empty:
            pass

        if all(j.state in Job.TERMINAL for j in self.jobs):
            self.polling = False
            self.start_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self._update_overall()
        else:
            self.root.after(100, self._poll)

    def _update_overall(self) -> None:
        done = sum(1 for j in self.jobs if j.state in Job.TERMINAL)
        if done == len(self.jobs):
            failed = sum(1 for j in self.jobs if j.state == "failed")
            cancelled = sum(1 for j in self.jobs if j.state == "cancelled")
            if failed:
                self.overall_var.set(f"Finished — {failed} folder(s) had errors")
            elif cancelled:
                self.overall_var.set("Cancelled")
            else:
                self.overall_var.set("All folders done ✓")
        else:
            running = sum(1 for j in self.jobs if j.state == "running")
            queued = sum(1 for j in self.jobs if j.state == "queued")
            self.overall_var.set(
                f"{done}/{len(self.jobs)} folders done — "
                f"{running} running, {queued} queued")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")


def main() -> None:
    """Entry point installed by pyproject.toml (gui-scripts)."""
    root = tk.Tk()
    NoisemakerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Re-run the unit tests (imports now pull in tkinter — must still pass)**

Run: `cd "/Users/sasha/Documents/iPod Project/tvShowNoisemaker" && .venv/bin/python -m pytest tests/test_gui.py -v`
Expected: 7 passed. (If this errors with `No module named '_tkinter'`, Task 1 wasn't completed.)

- [ ] **Step 3: Manual smoke test with multiple folders**

Set up three fake "seasons" from the repo's test media, then launch:

```bash
cd "/Users/sasha/Documents/iPod Project/tvShowNoisemaker"
rm -rf /tmp/nm-gui-test && mkdir -p /tmp/nm-gui-test/out
cp -r test_media "/tmp/nm-gui-test/Season 1"
cp -r test_media "/tmp/nm-gui-test/Season 2"
cp -r test_media "/tmp/nm-gui-test/Season 3"
.venv/bin/python -m tvshow_noisemaker.gui
```

Verify by hand:
1. Window opens; source list and destination are pre-filled if `~/.tvshow_noisemaker_gui.json` exists.
2. Add Folder… three times (Season 1–3); Remove Selected works; adding a duplicate folder is a no-op.
3. Destination = `/tmp/nm-gui-test/out`. Start → three progress rows appear; **only 2 show activity at first, the third says "Queued"** and starts when a slot frees.
4. Each row's counters advance independently; log lines are prefixed `[Season 1]`, `[Season 2]`, …
5. On completion: each row "Done ✓" (or "Finished with errors" — acceptable if counters match the log), overall status "All folders done ✓", and MP3s are in `/tmp/nm-gui-test/out/Season 1/` etc. (one subfolder per season, each containing its own CLI log file).
6. Start again, click Cancel All mid-run — running rows become "Cancelled", queued rows never start, Start re-enables.
7. Two sources with the same folder name are rejected with an error dialog.
8. Close and reopen the GUI — source list and destination are remembered.

- [ ] **Step 4: Commit**

```bash
git add tvshow_noisemaker/gui.py
git commit -m "feat: add multi-folder tkinter GUI with capped-parallel job runner"
```

---

### Task 4: Launchers — entry point, Finder app, README

**Files:**
- Modify: `pyproject.toml`
- Create: `TV Show Noisemaker.app/Contents/Info.plist`
- Create: `TV Show Noisemaker.app/Contents/MacOS/tvshow-noisemaker-gui` (executable shell script)
- Modify: `README.md`

**Interfaces:**
- Consumes: `tvshow_noisemaker.gui:main` from Task 3.
- Produces: `tvshow-noisemaker-gui` console command in the venv, and a double-clickable `TV Show Noisemaker.app` in the repo root (draggable to Dock/Desktop as an alias).

- [ ] **Step 1: Add the gui-scripts entry point to `pyproject.toml`**

Insert after the existing `[project.scripts]` table:

```toml
[project.gui-scripts]
tvshow-noisemaker-gui = "tvshow_noisemaker.gui:main"
```

- [ ] **Step 2: Reinstall the package so the entry point is generated**

Run: `cd "/Users/sasha/Documents/iPod Project/tvShowNoisemaker" && .venv/bin/python -m pip install -e . -q && ls .venv/bin/tvshow-noisemaker-gui`
Expected: the path prints (script exists).

- [ ] **Step 3: Create the Finder app bundle**

Create `TV Show Noisemaker.app/Contents/Info.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>TV Show Noisemaker</string>
    <key>CFBundleExecutable</key>
    <string>tvshow-noisemaker-gui</string>
    <key>CFBundleIdentifier</key>
    <string>local.tvshow-noisemaker.gui</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
</dict>
</plist>
```

Create `TV Show Noisemaker.app/Contents/MacOS/tvshow-noisemaker-gui`:

```bash
#!/bin/zsh
# Launch the GUI with the project venv. The bundle lives in the repo root,
# so the repo is three levels up from this script.
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
PYTHONPATH="$REPO" exec "$REPO/.venv/bin/python3" -m tvshow_noisemaker.gui
```

Make it executable (git preserves the bit):

```bash
chmod +x "TV Show Noisemaker.app/Contents/MacOS/tvshow-noisemaker-gui"
```

- [ ] **Step 4: Verify the bundle launches from Finder**

Run: `open "/Users/sasha/Documents/iPod Project/tvShowNoisemaker/TV Show Noisemaker.app"`
Expected: the GUI window opens with no Terminal window. Close it. (If Gatekeeper complains — it shouldn't for a locally created bundle — right-click → Open once.)

- [ ] **Step 5: Add a GUI section to README.md**

Append near the top of the usage docs (after the CLI usage section):

```markdown
## GUI

A minimal GUI is included — add one or more source folders (one per season), pick a destination folder, hit Start, and watch per-folder progress. Up to 2 folders convert at a time; the rest queue automatically. Each season's MP3s land in `<destination>/<season folder name>/`.

- **Finder:** double-click `TV Show Noisemaker.app` in the repo root (drag it to the Dock to keep it handy).
- **Terminal:** `tvshow-noisemaker-gui`

The GUI remembers your last-used folders. It requires Tk for Homebrew Python: `brew install python-tk@3.14` (one time).
```

- [ ] **Step 6: Run the full test suite**

Run: `cd "/Users/sasha/Documents/iPod Project/tvShowNoisemaker" && .venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml "TV Show Noisemaker.app" README.md
git commit -m "feat: add GUI entry point, Finder app bundle, and README docs"
```

---

## Verification (end-to-end)

1. `.venv/bin/python -m pytest -q` — full suite green.
2. Double-click `TV Show Noisemaker.app` in Finder → window opens, no Terminal.
3. Add 3 season folders (copies of `test_media/`), one destination → Start → three progress rows; only 2 run at once, the third queues; counters advance per row; MP3s land in `<dest>/<season name>/`.
4. Cancel All mid-run → running jobs stop, queued jobs never start, Start re-enables.
5. Relaunch → source list and destination remembered.

## Explicitly out of scope (YAGNI)

- Individual file selection — the CLI (and fingerprinting) operate on folders; sources are folders only.
- No option fields (bitrate, fallback durations, workers…) — power users keep the CLI.
- Configurable concurrency — fixed at `MAX_CONCURRENT = 2`.
- No app icon, no code signing, no packaging (py2app/PyInstaller).
- No changes to any pipeline module.
