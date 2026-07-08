"""Bare-bones tkinter GUI wrapping the tvshow-noisemaker CLI."""
from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Optional

_TOTAL_RE = re.compile(r"Starting conversion\s+\((\d+) files")
_POLL_MS = 100


def parse_total(line: str) -> Optional[int]:
    """Extract total file count from the CLI's 'Starting conversion' line."""
    m = _TOTAL_RE.search(line)
    return int(m.group(1)) if m else None


def is_file_result(line: str) -> bool:
    """True if the line marks one file as finished (converted, skipped, or failed)."""
    return "✓ Done" in line or "✗ Skipped" in line or "✗ Failed" in line


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
            # ✗ Skipped lines can arrive before this one — carry them over.
            self.total = total
            self.bar.stop()
            self.bar.config(mode="determinate", maximum=total, value=self.done)
            self.status.config(text=f"Converting {self.done}/{total}…")
        elif is_file_result(line):
            self.done += 1
            if self.total:
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
