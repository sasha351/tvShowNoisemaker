"""Bare-bones tkinter GUI wrapping the tvshow-noisemaker CLI."""
from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

_TOTAL_RE = re.compile(r"Starting conversion\s+\((\d+) files")
_POLL_MS = 100
_FORMATS = ("mp3", "m4a", "flac", "wav")
_LOSSLESS_FORMATS = ("flac", "wav")
_BITRATES = ("128", "192", "256", "320")


def parse_total(line: str) -> Optional[int]:
    """Extract total file count from the CLI's 'Starting conversion' line."""
    m = _TOTAL_RE.search(line)
    return int(m.group(1)) if m else None


def is_file_result(line: str) -> bool:
    """True if the line marks one file as finished (converted, skipped, or failed)."""
    return "✓ Done" in line or "✗ Skipped" in line or "✗ Failed" in line


def output_subdir_for(source_folder: str, dest_root: str) -> str:
    """Per-source-folder output path: <dest_root>/<basename(source_folder)>."""
    return str(Path(dest_root) / Path(source_folder).name)


def basename_collisions(folders: list[str]) -> list[str]:
    """Basenames that repeat among *folders* — these collide into the same output subfolder."""
    counts = Counter(Path(f).name for f in folders)
    return sorted(name for name, c in counts.items() if c > 1)


def parse_chosen_folders(output: str) -> list[str]:
    """Parse newline-separated POSIX paths from the macOS folder picker, dropping trailing slashes."""
    return [p.rstrip("/") for p in output.strip().split("\n") if p.strip()]


def format_folder_progress(
    index: int, total_folders: int, folder_name: str, done: int, total_files: int
) -> str:
    """Status text for batch mode, e.g. 'Folder 2/3 (ShowName): Converting 4/10…'."""
    prefix = f"Folder {index}/{total_folders} ({folder_name})"
    if total_files:
        return f"{prefix}: Converting {done}/{total_files}…"
    return f"{prefix}: Detecting intro/outro…"


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

        self.src_folders: list[str] = []
        self.pending_folders: list[str] = []
        self.current_folder: Optional[str] = None
        self.folder_index = 0
        self.total_folders = 0
        self.batch_cancelled = False
        self.folder_failures: list[str] = []

        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)

        self.dst = tk.StringVar(value=str(Path.home() / "Downloads"))
        self._source_folders_row(frm)
        self._path_row(frm, "Destination folder (audio):", self.dst)
        self._options_row(frm)

        self.start_btn = ttk.Button(frm, text="Start", command=self._start_or_cancel)
        self.start_btn.pack(pady=6)

        self.bar = ttk.Progressbar(frm, mode="indeterminate")
        self.bar.pack(fill="x", pady=4)

        self.status = ttk.Label(frm, text="Pick source and destination folders.")
        self.status.pack(anchor="w")

        log_frame = ttk.Frame(frm)
        log_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.log = tk.Text(log_frame, state="disabled", wrap="none", height=15)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

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

    def _options_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)

        self.output_format = tk.StringVar(value="mp3")
        ttk.Label(row, text="Format:").pack(side="left")
        fmt_box = ttk.Combobox(
            row, textvariable=self.output_format, values=_FORMATS,
            state="readonly", width=6,
        )
        fmt_box.pack(side="left", padx=(4, 16))
        fmt_box.bind("<<ComboboxSelected>>", self._on_format_change)

        self.bitrate = tk.StringVar(value="256")
        ttk.Label(row, text="Quality:").pack(side="left")
        self.quality_box = ttk.Combobox(
            row, textvariable=self.bitrate, values=_BITRATES,
            state="readonly", width=6,
        )
        self.quality_box.pack(side="left", padx=4)

    def _on_format_change(self, *_args) -> None:
        lossless = self.output_format.get() in _LOSSLESS_FORMATS
        self.quality_box.config(state="disabled" if lossless else "readonly")

    def _source_folders_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Source folders (videos):", width=24).pack(side="left", anchor="n")

        list_frame = ttk.Frame(row)
        list_frame.pack(side="left", fill="both", expand=True, padx=4)
        self.src_listbox = tk.Listbox(
            list_frame, height=4, selectmode="extended", exportselection=False
        )
        self.src_listbox.pack(side="left", fill="both", expand=True)
        lb_scroll = ttk.Scrollbar(list_frame, command=self.src_listbox.yview)
        self.src_listbox.configure(yscrollcommand=lb_scroll.set)
        lb_scroll.pack(side="left", fill="y")

        btn_col = ttk.Frame(row)
        btn_col.pack(side="left")
        self.add_folder_btn = ttk.Button(
            btn_col, text="Add Folder…", command=self._add_source_folder
        )
        self.add_folder_btn.pack(fill="x")
        self.remove_folder_btn = ttk.Button(
            btn_col, text="Remove Selected", command=self._remove_selected_folders
        )
        self.remove_folder_btn.pack(fill="x", pady=(4, 0))

    def _choose_folders(self) -> list[str]:
        """Prompt for one or more source folders (cmd+click multi-select on macOS)."""
        if sys.platform == "darwin":
            script = (
                "set theFolders to choose folder with multiple selections allowed\n"
                "set thePaths to {}\n"
                "repeat with f in theFolders\n"
                "    set end of thePaths to POSIX path of f\n"
                "end repeat\n"
                "set AppleScript's text item delimiters to linefeed\n"
                "return thePaths as text\n"
            )
            try:
                result = subprocess.run(
                    ["osascript"], input=script, capture_output=True, text=True,
                )
            except OSError:
                return []
            if result.returncode != 0:
                return []  # user cancelled the picker
            return parse_chosen_folders(result.stdout)
        folder = filedialog.askdirectory()
        return [folder] if folder else []

    def _add_source_folder(self) -> None:
        for folder in self._choose_folders():
            if folder and folder not in self.src_folders:
                self.src_folders.append(folder)
                self.src_listbox.insert("end", folder)
        dupes = basename_collisions(self.src_folders)
        if dupes:
            messagebox.showwarning(
                "Duplicate folder name",
                "These source folders share the same name and will write into "
                "the same output subfolder, possibly overwriting each other:\n\n"
                + ", ".join(dupes),
            )
        self._update_start_state()

    def _remove_selected_folders(self) -> None:
        for idx in reversed(self.src_listbox.curselection()):
            self.src_listbox.delete(idx)
            del self.src_folders[idx]
        self._update_start_state()

    def _update_start_state(self, *_args) -> None:
        if self.proc is None:
            ok = bool(self.src_folders) and bool(self.dst.get())
            self.start_btn.config(state="normal" if ok else "disabled")

    def _start_or_cancel(self) -> None:
        if self.proc is not None:
            self.batch_cancelled = True
            self.pending_folders.clear()
            self.proc.terminate()
            self.status.config(text="Cancelling…")
            return
        self.batch_cancelled = False
        self.folder_failures = []
        self.pending_folders = list(self.src_folders)
        self.total_folders = len(self.pending_folders)
        self.folder_index = 0
        self._log_clear()
        self.add_folder_btn.config(state="disabled")
        self.remove_folder_btn.config(state="disabled")
        self._advance_to_next_folder()

    def _advance_to_next_folder(self) -> None:
        if not self.pending_folders:
            self._batch_finished()
            return
        self.current_folder = self.pending_folders.pop(0)
        self.folder_index += 1
        self.total = 0
        self.done = 0
        name = Path(self.current_folder).name
        out_dir = output_subdir_for(self.current_folder, self.dst.get())
        self._log_append(f"== {name} ({self.folder_index}/{self.total_folders}) ==")
        cmd = [
            sys.executable, "-u", "-m", "tvshow_noisemaker.cli",
            "-i", self.current_folder, "-o", out_dir,
            "--format", self.output_format.get(),
        ]
        if self.output_format.get() not in _LOSSLESS_FORMATS:
            cmd.extend(["--bitrate", self.bitrate.get()])
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace",
            )
        except OSError as exc:
            self._log_append(f"Failed to launch: {exc}")
            self.folder_failures.append(name)
            self._advance_to_next_folder()
            return
        threading.Thread(target=self._reader, daemon=True).start()
        self.start_btn.config(text="Cancel")
        self.bar.config(mode="indeterminate")
        self.bar.start(80)
        self.status.config(
            text=format_folder_progress(self.folder_index, self.total_folders, name, 0, 0)
        )
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
        name = Path(self.current_folder).name if self.current_folder else ""
        total = parse_total(line)
        if total is not None:
            # ✗ Skipped lines can arrive before this one — carry them over.
            self.total = total
            self.bar.stop()
            self.bar.config(mode="determinate", maximum=total, value=self.done)
            self.status.config(
                text=format_folder_progress(
                    self.folder_index, self.total_folders, name, self.done, total
                )
            )
        elif is_file_result(line):
            self.done += 1
            if self.total:
                self.bar.config(value=self.done)
                self.status.config(
                    text=format_folder_progress(
                        self.folder_index, self.total_folders, name, self.done, self.total
                    )
                )

    def _finish(self) -> None:
        code = self.proc.returncode if self.proc else -1
        self.proc = None
        self.bar.stop()
        if code != 0 and self.current_folder:
            self.folder_failures.append(Path(self.current_folder).name)
        if self.batch_cancelled:
            self.pending_folders.clear()
            self._batch_finished(cancelled=True)
            return
        self._advance_to_next_folder()

    def _batch_finished(self, cancelled: bool = False) -> None:
        self.current_folder = None
        self.bar.stop()
        self.bar.config(mode="determinate", value=0)
        if cancelled:
            self.status.config(text="Cancelled.")
        elif self.folder_failures:
            self.status.config(
                text=f"Done — {self.folder_index}/{self.total_folders} folder(s), "
                f"{len(self.folder_failures)} failed: {', '.join(self.folder_failures)}"
            )
        else:
            self.status.config(text=f"Done — {self.total_folders} folder(s) processed.")
        self.start_btn.config(text="Start")
        self.add_folder_btn.config(state="normal")
        self.remove_folder_btn.config(state="normal")
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
        self.batch_cancelled = True
        self.pending_folders.clear()
        if self.proc is not None:
            self.proc.terminate()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
