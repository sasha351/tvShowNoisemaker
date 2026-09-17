"""CustomTkinter GUI wrapping the tvshow-noisemaker CLI."""
from __future__ import annotations

import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import customtkinter as ctk

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
    """Per-source-folder output path: <dest_root>/<basename(source_folder)> (Audio).

    The " (Audio)" suffix keeps converts out of the source folder when the
    destination is the source's parent (e.g. shows sitting in ~/Downloads).
    """
    return str(Path(dest_root) / f"{Path(source_folder).name} (Audio)")


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


def friendly_result_line(line: str) -> str:
    """Condense a raw CLI result line into a short activity-feed line, e.g. '✓ ep01.mp3'."""
    stripped = line.strip()
    if "✓ Done" in stripped:
        m = re.search(r"→\s*(\S+)", stripped)
        name = Path(m.group(1)).name if m else stripped
        return f"✓ {name}"
    return stripped


class App:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        root.title("TV Show Noisemaker")
        root.geometry("640x680")
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
        self._folder_row_buttons: list[ctk.CTkButton] = []
        self._details_shown = False

        frm = ctk.CTkFrame(root, fg_color="transparent")
        frm.pack(fill="both", expand=True, padx=20, pady=20)

        self.dst = tk.StringVar(value=str(Path.home() / "Downloads"))
        self._shows_section(frm)
        self._destination_row(frm)
        self._options_row(frm)
        self._status_card(frm)
        self._activity_feed(frm)

        self.start_btn = ctk.CTkButton(
            frm, text="Start", height=40, command=self._start_or_cancel
        )
        self.start_btn.pack(fill="x", pady=(12, 6))

        self._details_toggle(frm)

        self.dst.trace_add("write", self._update_start_state)
        self._update_start_state()

    # -- Shows -----------------------------------------------------------

    def _shows_section(self, parent: ctk.CTkFrame) -> None:
        ctk.CTkLabel(
            parent, text="Shows", font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray40", "gray60"), anchor="w",
        ).pack(fill="x")
        self.src_frame = ctk.CTkScrollableFrame(parent, height=110)
        self.src_frame.pack(fill="x", pady=(4, 4))
        self.add_folder_btn = ctk.CTkButton(
            parent, text="＋ Add Folders…", fg_color="transparent", border_width=1,
            text_color=("gray10", "gray90"), command=self._add_source_folder,
        )
        self.add_folder_btn.pack(fill="x", pady=(0, 12))
        self._render_source_rows()

    def _render_source_rows(self) -> None:
        for child in self.src_frame.winfo_children():
            child.destroy()
        self._folder_row_buttons = []
        for folder in self.src_folders:
            row = ctk.CTkFrame(self.src_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)
            info = ctk.CTkFrame(row, fg_color="transparent")
            info.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(
                info, text=Path(folder).name, font=ctk.CTkFont(weight="bold"), anchor="w",
            ).pack(fill="x")
            ctk.CTkLabel(
                info, text=folder, font=ctk.CTkFont(size=11),
                text_color=("gray40", "gray60"), anchor="w",
            ).pack(fill="x")
            remove_btn = ctk.CTkButton(
                row, text="✕", width=28, height=28, fg_color="transparent",
                text_color=("gray40", "gray60"), hover_color=("gray80", "gray30"),
                command=lambda f=folder: self._remove_folder(f),
            )
            remove_btn.pack(side="right")
            self._folder_row_buttons.append(remove_btn)

    def _add_source_folder(self) -> None:
        for folder in self._choose_folders():
            if folder and folder not in self.src_folders:
                self.src_folders.append(folder)
        self._render_source_rows()
        dupes = basename_collisions(self.src_folders)
        if dupes:
            messagebox.showwarning(
                "Duplicate folder name",
                "These source folders share the same name and will write into "
                "the same output subfolder, possibly overwriting each other:\n\n"
                + ", ".join(dupes),
            )
        self._update_start_state()

    def _remove_folder(self, folder: str) -> None:
        self.src_folders.remove(folder)
        self._render_source_rows()
        self._update_start_state()

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

    # -- Destination & options --------------------------------------------

    def _destination_row(self, parent: ctk.CTkFrame) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text="Destination folder (audio):", width=180, anchor="w").pack(
            side="left"
        )
        ctk.CTkEntry(row, textvariable=self.dst, state="readonly").pack(
            side="left", fill="x", expand=True, padx=4
        )
        ctk.CTkButton(
            row, text="Choose…", width=90,
            command=lambda: self.dst.set(filedialog.askdirectory() or self.dst.get()),
        ).pack(side="left")

    def _options_row(self, parent: ctk.CTkFrame) -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(6, 2))

        self.output_format = tk.StringVar(value="mp3")
        ctk.CTkLabel(row, text="Format:").pack(side="left")
        ctk.CTkOptionMenu(
            row, values=list(_FORMATS), variable=self.output_format, width=90,
            command=self._on_format_change,
        ).pack(side="left", padx=(4, 16))

        self.bitrate = tk.StringVar(value="256")
        ctk.CTkLabel(row, text="Quality:").pack(side="left")
        self.quality_menu = ctk.CTkOptionMenu(
            row, values=list(_BITRATES), variable=self.bitrate, width=90,
        )
        self.quality_menu.pack(side="left", padx=4)

    def _on_format_change(self, *_args) -> None:
        lossless = self.output_format.get() in _LOSSLESS_FORMATS
        self.quality_menu.configure(state="disabled" if lossless else "normal")

    # -- Status card & activity feed --------------------------------------

    def _status_card(self, parent: ctk.CTkFrame) -> None:
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.pack(fill="x", pady=(12, 6))
        self.status = ctk.CTkLabel(
            card, text="Pick source and destination folders.",
            font=ctk.CTkFont(size=15), anchor="w",
        )
        self.status.pack(fill="x", padx=16, pady=(12, 8))
        self.bar = ctk.CTkProgressBar(card)
        self.bar.pack(fill="x", padx=16, pady=(0, 16))
        self.bar.set(0)

    def _activity_feed(self, parent: ctk.CTkFrame) -> None:
        self.activity = ctk.CTkTextbox(parent, state="disabled", wrap="word", height=140)
        self.activity.pack(fill="both", expand=True)

    def _details_toggle(self, parent: ctk.CTkFrame) -> None:
        self.details_btn = ctk.CTkButton(
            parent, text="Show details ▸", fg_color="transparent", anchor="w", height=24,
            text_color=("gray30", "gray70"), hover_color=("gray90", "gray20"),
            command=self._toggle_details,
        )
        self.details_btn.pack(fill="x")
        self.details_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.log = ctk.CTkTextbox(
            self.details_frame, state="disabled", wrap="none", height=140,
            font=("Menlo", 11),
        )
        self.log.pack(fill="both", expand=True)

    def _toggle_details(self) -> None:
        self._details_shown = not self._details_shown
        if self._details_shown:
            self.details_frame.pack(fill="both", expand=True, pady=(4, 0))
            self.details_btn.configure(text="Hide details ▾")
        else:
            self.details_frame.pack_forget()
            self.details_btn.configure(text="Show details ▸")

    # -- Run state ---------------------------------------------------------

    def _update_start_state(self, *_args) -> None:
        if self.proc is None:
            ok = bool(self.src_folders) and bool(self.dst.get())
            self.start_btn.configure(state="normal" if ok else "disabled")

    def _start_or_cancel(self) -> None:
        if self.proc is not None:
            self.batch_cancelled = True
            self.pending_folders.clear()
            self.proc.terminate()
            self.status.configure(text="Cancelling…")
            return
        self.batch_cancelled = False
        self.folder_failures = []
        self.pending_folders = list(self.src_folders)
        self.total_folders = len(self.pending_folders)
        self.folder_index = 0
        self._log_clear()
        self._activity_clear()
        self.add_folder_btn.configure(state="disabled")
        for btn in self._folder_row_buttons:
            btn.configure(state="disabled")
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
        if self.folder_index > 1:
            self._activity_append("")
        self._activity_append(f"{name} ({self.folder_index}/{self.total_folders})")
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
        self.start_btn.configure(text="Cancel")
        self.bar.configure(mode="indeterminate")
        self.bar.start()
        self.status.configure(
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
            self.bar.configure(mode="determinate")
            self.bar.set(self.done / total if total else 0)
            self.status.configure(
                text=format_folder_progress(
                    self.folder_index, self.total_folders, name, self.done, total
                )
            )
        elif is_file_result(line):
            self.done += 1
            self._activity_append(friendly_result_line(line))
            if self.total:
                self.bar.set(self.done / self.total)
                self.status.configure(
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
        self.bar.configure(mode="determinate")
        self.bar.set(0)
        if cancelled:
            self.status.configure(text="Cancelled.")
        elif self.folder_failures:
            self.status.configure(
                text=f"Done — {self.folder_index}/{self.total_folders} folder(s), "
                f"{len(self.folder_failures)} failed: {', '.join(self.folder_failures)}"
            )
        else:
            self.status.configure(text=f"Done — {self.total_folders} folder(s) processed.")
        self.start_btn.configure(text="Start")
        self.add_folder_btn.configure(state="normal")
        for btn in self._folder_row_buttons:
            btn.configure(state="normal")
        self._update_start_state()

    def _log_append(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _log_clear(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _activity_append(self, line: str) -> None:
        self.activity.configure(state="normal")
        self.activity.insert("end", line + "\n")
        self.activity.see("end")
        self.activity.configure(state="disabled")

    def _activity_clear(self) -> None:
        self.activity.configure(state="normal")
        self.activity.delete("1.0", "end")
        self.activity.configure(state="disabled")

    def _on_close(self) -> None:
        self.batch_cancelled = True
        self.pending_folders.clear()
        if self.proc is not None:
            self.proc.terminate()
        self.root.destroy()


def main() -> None:
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
