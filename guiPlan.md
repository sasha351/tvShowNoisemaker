# tvshow-noisemaker GUI — Spec & Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A bare-bones desktop GUI so the user can pick a source folder, pick a destination folder, hit Start, and watch progress — no terminal, no path-pasting, no formatting worries.

**Architecture:** One new file, `tvshow_noisemaker/gui.py`, built entirely on tkinter (Python stdlib — zero new dependencies). The GUI runs the existing CLI (`python -m tvshow_noisemaker.cli`) as a subprocess, streams its stdout line-by-line into a thread-safe queue, and the Tk main loop polls that queue every 100 ms to update a log pane and a progress bar. The CLI is never modified; progress is derived by parsing the lines it already prints.

**Tech Stack:** Python ≥3.9, tkinter + ttk (stdlib), `subprocess`, `threading`, `queue`. Nothing added to `pyproject.toml` dependencies.

## Global Constraints

- No new third-party dependencies — stdlib only (tkinter ships with python.org and Homebrew Python on macOS).
- The existing CLI (`tvshow_noisemaker/cli.py`) must not be modified.
- Python ≥3.9 (matches `pyproject.toml` `requires-python`).
- Single new source file for the GUI; one entry point `tvshow-noisemaker-gui` added to `[project.scripts]`.
- Function over looks: default ttk widgets, no theming, no icons.

---

## Part 1 — Spec

### What it is

A single small window (~700×500) with:

1. **Source row** — read-only text field + `Browse…` button (`tkinter.filedialog.askdirectory`). This is the folder of video episodes (`--input-dir`).
2. **Destination row** — same pattern, for the MP3 output folder (`--output-dir`).
3. **Start button** — disabled until both folders are chosen; becomes `Cancel` while a run is active.
4. **Progress bar** — `ttk.Progressbar`. Indeterminate (bouncing) during the detection/fingerprint phase; switches to determinate once the CLI prints its `Starting conversion  (N files, …)` line, then advances as each `✓ Done` / `✗ Skipped` line appears.
5. **Status line** — one label, e.g. `Fingerprinting…`, `Converting 3/12…`, `Done: 11 converted, 1 skipped`, or `Failed — see log`.
6. **Log pane** — read-only scrolling `tkinter.Text` showing the CLI's own output verbatim. This *is* the detail view; no custom formatting.

### How it works

- Start spawns `[sys.executable, "-u", "-m", "tvshow_noisemaker.cli", "-i", src, "-o", dst]` with `stdout=PIPE, stderr=STDOUT, text=True`.
- A daemon thread reads the pipe line-by-line and puts lines on a `queue.Queue`. A sentinel (`None`) is queued when the process exits, along with the return code.
- The Tk thread polls the queue via `root.after(100, poll)` — no cross-thread widget access ever, which is the classic tkinter crash source. This is the "bulletproof" part.
- Progress parsing (regex on lines the CLI already emits):
  - `Starting conversion  (N files` → total known, switch bar to determinate, max = N.
  - lines containing `✓ Done` or `✗ Skipped` → increment completed count.
- Cancel calls `process.terminate()`; the reader thread's sentinel unwinds the UI back to idle.
- Exit code 0 → status shows the summary; nonzero → status shows failure, log pane has the reason.

### Failure handling (kept minimal but real)

- Both paths required before Start enables — no invalid-input state possible.
- Source folder with no video files: the CLI already errors and exits nonzero; the GUI just shows "Failed — see log". No duplicate validation in the GUI.
- ffmpeg missing: same — the CLI already handles/reports it.
- Window closed mid-run: `WM_DELETE_WINDOW` handler terminates the subprocess before destroying the window (prevents orphaned ffmpeg processes).

### Explicitly out of scope (YAGNI)

- All CLI tuning options (bitrate, silence thresholds, workers, config file) — CLI defaults are sane; power users have the CLI.
- Drag-and-drop, remembering last-used folders, dark mode, packaging as a .app. Add only if asked.
- A separate progress channel in the CLI (JSON progress events, callbacks). Parsing existing log lines is enough; if the log format ever changes, the bar degrades to indeterminate but the run still works and the log still streams — graceful degradation by design.

### Design decisions & rationale

| Decision | Why |
|---|---|
| tkinter over PyQt/PySimpleGUI/web UI | Stdlib = zero install risk, zero new deps, tiny footprint. Ugly-but-functional was explicitly acceptable. |
| Subprocess over importing `cli.run()` in-process | Total isolation: a crash, hang, or `sys.exit` in the pipeline can't take down the GUI; Cancel is a clean `terminate()`; logging handlers don't fight over the root logger. |
| Parse existing log lines for progress | No CLI changes; worst case the bar is indeterminate but everything still works. |
| Queue + `after()` polling | The only thread-safe tkinter pattern; direct widget calls from threads segfault intermittently. |
| `-u` flag on the subprocess | Unbuffered stdout so lines arrive live, not in one burst at exit. |

---

## Part 2 — Implementation Plan (ponytail: full)

One file, three tasks, ~150 lines total. The parsing logic gets a real test; the widget layout does not (testing tkinter layout is effort spent where nothing breaks).

### Task 1: Progress-line parser

The only non-trivial logic, so it's the only unit-tested piece. Pure functions, no tkinter imports.

**Files:**
- Create: `tvshow_noisemaker/gui.py` (parser functions only, GUI added in Task 2)
- Test: `tests/test_gui_parser.py`

**Interfaces:**
- Produces: `parse_total(line: str) -> Optional[int]` — returns N from a `Starting conversion  (N files, …)` line, else None.
- Produces: `is_file_result(line: str) -> bool` — True if the line marks one file finished (`✓ Done` or `✗ Skipped`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gui_parser.py
from tvshow_noisemaker.gui import parse_total, is_file_result


def test_parse_total_matches_conversion_line():
    line = "2026-07-07 10:00:00 [INFO] tvshow_noisemaker.console: Starting conversion  (12 files, 11 to encode, 4 workers, output → /tmp/out)"
    assert parse_total(line) == 12


def test_parse_total_ignores_other_lines():
    assert parse_total("Processing episode01.mkv") is None
    assert parse_total("") is None


def test_is_file_result_done_and_skipped():
    assert is_file_result("  ✓ Done in 3.2s → /tmp/out/ep01.mp3 (4.1 MB)")
    assert is_file_result("  ✗ Skipped – no boundaries resolved")
    assert not is_file_result("Processing episode01.mkv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gui_parser.py -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError` (no `gui` module yet).

- [ ] **Step 3: Write minimal implementation**

```python
# tvshow_noisemaker/gui.py
"""Bare-bones tkinter GUI wrapping the tvshow-noisemaker CLI."""
from __future__ import annotations

import re
from typing import Optional

_TOTAL_RE = re.compile(r"Starting conversion\s+\((\d+) files")


def parse_total(line: str) -> Optional[int]:
    """Extract total file count from the CLI's 'Starting conversion' line."""
    m = _TOTAL_RE.search(line)
    return int(m.group(1)) if m else None


def is_file_result(line: str) -> bool:
    """True if the line marks one file as finished (converted or skipped)."""
    return "✓ Done" in line or "✗ Skipped" in line
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gui_parser.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tvshow_noisemaker/gui.py tests/test_gui_parser.py
git commit -m "feat: add GUI progress-line parser"
```

### Task 2: The window

All the tkinter code. No unit tests — verified by launching it (Step 3). Manual smoke test is the honest test here.

**Files:**
- Modify: `tvshow_noisemaker/gui.py` (append below the parser functions)

**Interfaces:**
- Consumes: `parse_total`, `is_file_result` from Task 1.
- Produces: `main() -> None` — builds the window and enters `mainloop()`. This is what Task 3's entry point calls.

- [ ] **Step 1: Append the GUI code to `tvshow_noisemaker/gui.py`**

```python
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk

_POLL_MS = 100


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("TV Show Noisemaker")
        root.geometry("700x500")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.proc: Optional[subprocess.Popen] = None
        self.q: "queue.Queue[Optional[str]]" = queue.Queue()
        self.total = 0
        self.done = 0

        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)

        self.src = tk.StringVar()
        self.dst = tk.StringVar()
        self._path_row(frm, "Source folder (videos):", self.src)
        self._path_row(frm, "Destination folder (MP3s):", self.dst)

        self.start_btn = ttk.Button(frm, text="Start", command=self._start_or_cancel)
        self.start_btn.pack(pady=6)

        self.bar = ttk.Progressbar(frm, mode="indeterminate")
        self.bar.pack(fill="x", pady=4)

        self.status = ttk.Label(frm, text="Pick a source and destination folder.")
        self.status.pack(anchor="w")

        log_frame = ttk.Frame(frm)
        log_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.log = tk.Text(log_frame, state="disabled", wrap="none", height=15)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        self.src.trace_add("write", self._update_start_state)
        self.dst.trace_add("write", self._update_start_state)
        self._update_start_state()

    def _path_row(self, parent: ttk.Frame, label: str, var: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=24).pack(side="left")
        ttk.Entry(row, textvariable=var, state="readonly").pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(
            row, text="Browse…",
            command=lambda: var.set(filedialog.askdirectory() or var.get()),
        ).pack(side="left")

    def _update_start_state(self, *_args) -> None:
        if self.proc is None:
            ok = bool(self.src.get()) and bool(self.dst.get())
            self.start_btn.config(state="normal" if ok else "disabled")

    def _start_or_cancel(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self.status.config(text="Cancelling…")
            return
        self.total = 0
        self.done = 0
        self._log_clear()
        cmd = [
            sys.executable, "-u", "-m", "tvshow_noisemaker.cli",
            "-i", self.src.get(), "-o", self.dst.get(),
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace",
            )
        except OSError as exc:
            self.status.config(text=f"Failed to launch: {exc}")
            return
        threading.Thread(target=self._reader, daemon=True).start()
        self.start_btn.config(text="Cancel")
        self.bar.config(mode="indeterminate")
        self.bar.start(80)
        self.status.config(text="Detecting intro/outro…")
        self.root.after(_POLL_MS, self._poll)

    def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self.q.put(line.rstrip("\n"))
        self.proc.wait()
        self.q.put(None)  # sentinel: process finished

    def _poll(self) -> None:
        try:
            while True:
                line = self.q.get_nowait()
                if line is None:
                    self._finish()
                    return
                self._handle_line(line)
        except queue.Empty:
            pass
        self.root.after(_POLL_MS, self._poll)

    def _handle_line(self, line: str) -> None:
        self._log_append(line)
        total = parse_total(line)
        if total is not None:
            self.total = total
            self.bar.stop()
            self.bar.config(mode="determinate", maximum=total, value=0)
            self.status.config(text=f"Converting 0/{total}…")
        elif self.total and is_file_result(line):
            self.done += 1
            self.bar.config(value=self.done)
            self.status.config(text=f"Converting {self.done}/{self.total}…")

    def _finish(self) -> None:
        code = self.proc.returncode if self.proc else -1
        self.proc = None
        self.bar.stop()
        if self.total:
            self.bar.config(mode="determinate", value=self.done)
        else:
            self.bar.config(mode="determinate", value=0)
        self.status.config(
            text=f"Done — {self.done}/{self.total} files processed."
            if code == 0 else f"Failed (exit code {code}) — see log above."
        )
        self.start_btn.config(text="Start")
        self._update_start_state()

    def _log_append(self, line: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _log_clear(self) -> None:
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _on_close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the parser tests still pass**

Run: `python -m pytest tests/test_gui_parser.py -v`
Expected: 3 passed (tkinter import at module level must not break headless test collection — it won't, since importing `tkinter` without calling `tk.Tk()` is safe; if CI ever lacks tkinter entirely, move the tkinter imports inside `main()`).

- [ ] **Step 3: Manual smoke test**

Run: `python -m tvshow_noisemaker.gui`
Expected: window opens; Start is disabled until both folders are picked. Pick `test_media` as source and a scratch folder as destination, hit Start: log lines stream, bar goes indeterminate then determinate, status ends with "Done" or "Failed" and Start re-enables. Test Cancel mid-run: button flips back, no orphaned ffmpeg (`pgrep ffmpeg` empty after a few seconds).

- [ ] **Step 4: Commit**

```bash
git add tvshow_noisemaker/gui.py
git commit -m "feat: add bare-bones tkinter GUI"
```

### Task 3: Entry point

**Files:**
- Modify: `pyproject.toml` (the `[project.scripts]` table)

**Interfaces:**
- Consumes: `main()` from Task 2.
- Produces: `tvshow-noisemaker-gui` console command.

- [ ] **Step 1: Add the script entry**

In `pyproject.toml`, change:

```toml
[project.scripts]
tvshow-noisemaker = "tvshow_noisemaker.cli:main"
```

to:

```toml
[project.scripts]
tvshow-noisemaker = "tvshow_noisemaker.cli:main"
tvshow-noisemaker-gui = "tvshow_noisemaker.gui:main"
```

- [ ] **Step 2: Reinstall and verify the command exists**

Run: `pip install -e . && which tvshow-noisemaker-gui`
Expected: prints a path; running `tvshow-noisemaker-gui` opens the window.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest`
Expected: all tests pass (existing suite + the 3 new parser tests).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add tvshow-noisemaker-gui entry point"
```

---

*Skipped: CLI option controls, drag-and-drop, folder persistence, packaging — add any of these only when actually wanted. If the CLI's log wording ever changes, update the two regex/substring checks in `parse_total`/`is_file_result`; until then the bar just stays indeterminate and everything else keeps working.*
