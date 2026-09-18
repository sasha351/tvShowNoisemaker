"""CustomTkinter GUI driving the tvshow-noisemaker pipeline in-process."""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional

import customtkinter as ctk

from tvshow_noisemaker.config import Config
from tvshow_noisemaker.pipeline import EpisodePlan, Plan, Summary, detect, encode
from tvshow_noisemaker.processor import terminate_all
from tvshow_noisemaker.review import (
    QueueHandler,
    ReviewRow,
    SnippetPlayer,
    format_time,
    parse_time,
    split_plan,
)

_POLL_MS = 100
_FORMATS = ("mp3", "m4a", "flac", "wav")
_LOSSLESS_FORMATS = ("flac", "wav")
_BITRATES = ("128", "192", "256", "320")
_FLUSH = ("flush",)  # encode-queue sentinel: detection finished (or the batch was cancelled)
_FIELDS = (("intro_start", "Intro start"), ("intro_end", "Intro end"), ("outro_start", "Credits start"))


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
    index: int, total_folders: int, folder_name: str, done: int, total_files: int,
    stage: str = "encode",
) -> str:
    """Status text for batch mode, e.g. 'Folder 2/3 (ShowName): Converting 4/10…'."""
    prefix = f"Folder {index}/{total_folders} ({folder_name})"
    if not total_files:
        return f"{prefix}: Detecting intro/outro…"
    verb = "Converting" if stage == "encode" else "Detecting"
    return f"{prefix}: {verb} {done}/{total_files}…"


def format_eta(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"~{minutes}m {secs}s left" if minutes else f"~{secs}s left"


@dataclass
class _Row:
    """One review-panel row: the episode plus the widgets holding the user's edits."""

    folder: str
    episode: EpisodePlan
    frame: "ctk.CTkFrame"
    skip_var: tk.BooleanVar
    entries: Dict[str, "ctk.CTkEntry"] = field(default_factory=dict)


class App:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        root.title("TV Show Noisemaker")
        root.geometry("640x760")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.encode_jobs: "queue.Queue" = queue.Queue()
        self.cancel = threading.Event()
        self.player = SnippetPlayer()

        self.src_folders: list[str] = []
        self.review_rows: List[_Row] = []
        self.approved: Dict[str, List[EpisodePlan]] = {}
        self.running = False
        self.cancelling = False
        self._detect_done = True
        self._folder_index: Dict[str, int] = {}
        self._total_folders = 0
        self._last_folder: Optional[str] = None
        self._enc_done = 0
        self._enc_total = 0
        self._enc_t0: Optional[float] = None
        self._handler: Optional[QueueHandler] = None
        self._folder_row_buttons: list[ctk.CTkButton] = []
        self._details_shown = False
        self._review_shown = False
        self._fmt, self._bitrate, self._dst, self._review_all = "mp3", "256", "", False

        frm = ctk.CTkFrame(root, fg_color="transparent")
        frm.pack(fill="both", expand=True, padx=20, pady=20)

        self.dst = tk.StringVar(value=str(Path.home() / "Downloads"))
        self._shows_section(frm)
        self._destination_row(frm)
        self._options_row(frm)
        self._status_card(frm)
        self._activity_feed(frm)
        self._review_panel(frm)

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
        self.quality_menu.pack(side="left", padx=(4, 16))

        self.review_all = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(row, text="Review every episode", variable=self.review_all).pack(side="left")

    def _on_format_change(self, *_args) -> None:
        lossless = self.output_format.get() in _LOSSLESS_FORMATS
        self.quality_menu.configure(state="disabled" if lossless else "normal")

    # -- Status card, review panel & activity feed -------------------------

    def _status_card(self, parent: ctk.CTkFrame) -> None:
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.pack(fill="x", pady=(12, 6))
        self.status = ctk.CTkLabel(
            card, text="Pick source and destination folders.",
            font=ctk.CTkFont(size=15), anchor="w",
        )
        self.status.pack(fill="x", padx=16, pady=(12, 8))
        self.bar = ctk.CTkProgressBar(card)
        self.bar.pack(fill="x", padx=16, pady=(0, 4))
        self.bar.set(0)
        self.eta = ctk.CTkLabel(
            card, text="", font=ctk.CTkFont(size=11), text_color=("gray40", "gray60"), anchor="w",
        )
        self.eta.pack(fill="x", padx=16, pady=(0, 12))

    def _review_panel(self, parent: ctk.CTkFrame) -> None:
        self.review_frame = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(
            self.review_frame, text="Needs review", font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray40", "gray60"), anchor="w",
        ).pack(fill="x")
        self.review_list = ctk.CTkScrollableFrame(self.review_frame, height=180)
        self.review_list.pack(fill="x", pady=(4, 4))
        bar = ctk.CTkFrame(self.review_frame, fg_color="transparent")
        bar.pack(fill="x")
        ctk.CTkButton(bar, text="Approve all", width=100, command=self._approve_all).pack(side="left")
        ctk.CTkButton(
            bar, text="Export approved cuts…", width=170, command=self._export_approved,
        ).pack(side="left", padx=6)
        ctk.CTkButton(bar, text="■", width=28, command=self.player.stop).pack(side="left")

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

    # -- Review rows -------------------------------------------------------

    def _add_review_row(self, folder: str, ep: EpisodePlan) -> None:
        frame = ctk.CTkFrame(self.review_list)
        frame.pack(fill="x", pady=2)
        row = _Row(folder=folder, episode=ep, frame=frame, skip_var=tk.BooleanVar(value=False))
        ctk.CTkLabel(
            frame, text=ep.basename, font=ctk.CTkFont(weight="bold"), anchor="w",
        ).pack(fill="x", padx=8, pady=(6, 0))
        ctk.CTkLabel(
            frame, text="; ".join(ep.flags), font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60"), anchor="w", justify="left", wraplength=420,
        ).pack(fill="x", padx=8)
        controls = ctk.CTkFrame(frame, fg_color="transparent")
        controls.pack(fill="x", padx=8, pady=(2, 6))
        for name, label in _FIELDS:
            ctk.CTkButton(
                controls, text="▶", width=28, command=lambda n=name: self._preview(row, n),
            ).pack(side="left")
            entry = ctk.CTkEntry(controls, width=70, placeholder_text=label)
            entry.insert(0, format_time(getattr(ep, name)))
            entry.pack(side="left", padx=(2, 8))
            row.entries[name] = entry
        ctk.CTkCheckBox(controls, text="Skip", variable=row.skip_var, width=60).pack(side="left")
        ctk.CTkButton(controls, text="Approve", width=80, command=lambda: self._approve(row)).pack(side="right")
        self.review_rows.append(row)
        if not self._review_shown:
            self.review_frame.pack(fill="x", pady=(6, 6), before=self.activity)
            self._review_shown = True

    def _preview(self, row: _Row, name: str) -> None:
        try:
            self.player.play(row.episode.path, parse_time(row.entries[name].get()))
        except ValueError:
            pass  # an unparseable time just means there is nothing to preview

    def _approve(self, row: _Row) -> bool:
        """Queue one reviewed episode for encoding.  False if the row still needs work."""
        review_row = ReviewRow(
            row.episode,
            row.entries["intro_start"].get(),
            row.entries["intro_end"].get(),
            row.entries["outro_start"].get(),
            skip=bool(row.skip_var.get()),
        )
        try:
            ep = review_row.to_episode()
        except ValueError as exc:
            messagebox.showwarning("Bad time", str(exc))
            return False
        if not ep.encodable and not review_row.skip:
            messagebox.showwarning(
                "Incomplete cut", "Fill in intro end and credits start (or tick Skip).",
            )
            return False
        self._drop_row(row)
        self.approved.setdefault(row.folder, []).append(ep)
        self._last_folder = row.folder
        self._enc_total += 1
        self.encode_jobs.put((
            row.folder,
            Plan(input_dir=row.folder, episodes=[ep]),
            output_subdir_for(row.folder, self._dst),
        ))
        return True

    def _approve_all(self) -> None:
        for row in list(self.review_rows):
            if not self._approve(row):
                return

    def _drop_row(self, row: _Row) -> None:
        row.frame.destroy()
        self.review_rows.remove(row)
        if not self.review_rows and self._review_shown:
            self.review_frame.pack_forget()
            self._review_shown = False

    def _clear_review(self) -> None:
        for row in list(self.review_rows):
            self._drop_row(row)

    def _export_approved(self) -> None:
        folder = self.review_rows[0].folder if self.review_rows else self._last_folder
        if folder is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json")
        if not path:
            return
        Plan(input_dir=folder, episodes=self.approved.get(folder, [])).save(path)

    # -- Run state ---------------------------------------------------------

    def _update_start_state(self, *_args) -> None:
        if not self.running:
            ok = bool(self.src_folders) and bool(self.dst.get())
            self.start_btn.configure(state="normal" if ok else "disabled")

    def _cfg(self, workers: Optional[int] = None) -> Config:
        return Config(output_format=self._fmt, mp3_bitrate=int(self._bitrate), workers=workers)

    def _encode_cfg(self) -> Config:
        # Leave cores for detection while it is still running; take them all afterwards.
        cpu = os.cpu_count() or 2
        return self._cfg(None if self._detect_done else max(1, cpu - cpu // 2))

    def _progress_for(self, folder: str):
        def _progress(stage: str, done: int, total: int, message: str) -> None:
            self.q.put(("progress", folder, stage, done, total, message))
        return _progress

    def _start_or_cancel(self) -> None:
        if self.running:
            self._cancel()
            return
        self._begin_batch()

    def _begin_batch(self, start_threads: bool = True) -> None:
        """Reset state, install the log relay and (optionally) start the workers."""
        folders = list(self.src_folders)
        self.cancel = threading.Event()
        self.encode_jobs = queue.Queue()
        self.q = queue.Queue()
        self.review_rows = []
        self.approved = {}
        self.running = True
        self.cancelling = False
        self._detect_done = False
        self._folder_index = {f: i + 1 for i, f in enumerate(folders)}
        self._total_folders = len(folders)
        self._last_folder = None
        self._enc_done = self._enc_total = 0
        self._enc_t0 = None
        self._fmt = self.output_format.get()
        self._bitrate = self.bitrate.get()
        self._dst = self.dst.get()
        self._review_all = bool(self.review_all.get())
        self._clear_review()
        self._log_clear()
        self._activity_clear()
        self.eta.configure(text="")
        self.bar.set(0)
        self.status.configure(text="Detecting intro/outro…")
        self.start_btn.configure(text="Cancel", state="normal")
        self.add_folder_btn.configure(state="disabled")
        for btn in self._folder_row_buttons:
            btn.configure(state="disabled")

        self._handler = QueueHandler(self.q)
        self._handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        self._handler.setLevel(logging.INFO)
        lib_log = logging.getLogger("tvshow_noisemaker")
        lib_log.setLevel(logging.INFO)
        lib_log.addHandler(self._handler)

        if start_threads:
            threading.Thread(target=self._detect_worker, args=(folders,), daemon=True).start()
            threading.Thread(target=self._encode_worker, daemon=True).start()
        self.root.after(_POLL_MS, self._poll)

    def _cancel(self) -> None:
        if self.cancelling:
            return
        self.cancelling = True
        self.cancel.set()
        terminate_all()
        self.player.stop()
        self._clear_review()
        self.encode_jobs.put(_FLUSH)  # wake the encode worker if it is waiting for approvals
        self.status.configure(text="Cancelling…")
        self.eta.configure(text="")

    # -- Workers (no Tk access: they only put tuples on self.q) -------------

    def _detect_worker(self, folders: List[str]) -> None:
        for folder in folders:
            if self.cancel.is_set():
                break
            try:
                plan = detect(folder, self._cfg(), progress=self._progress_for(folder), cancel=self.cancel)
            except Exception as exc:  # pylint: disable=broad-except
                self.q.put(("detect_failed", folder, str(exc)))
                continue
            if self.cancel.is_set():
                break
            self.q.put(("plan", folder, plan))
        self._detect_done = True
        # The sentinel goes through the Tk queue, not straight onto encode_jobs: the last
        # ("plan", …) may still be unhandled, and the encode worker would see an empty queue
        # with no review rows and call the batch finished before those jobs were ever queued.
        self.q.put(("detect_done",))

    def _encode_worker(self) -> None:
        flushed = False
        while True:
            job = self.encode_jobs.get()
            if self.cancel.is_set():
                break
            if job is _FLUSH:
                flushed = True
            else:
                folder, plan, out_dir = job
                try:
                    summary = encode(
                        plan, self._encode_cfg(), out_dir,
                        progress=self._progress_for(folder), cancel=self.cancel,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    summary = Summary(failed=[(Path(folder).name, str(exc))])
                self.q.put(("encoded", folder, summary))
            # ponytail: review_rows is read across threads without a lock — list length under
            # the GIL is enough here; the worker only ever waits longer than needed, never less.
            if flushed and not self.review_rows and self.encode_jobs.empty():
                break
        self.q.put(("idle",))

    # -- Tk-thread message pump --------------------------------------------

    def _poll(self) -> None:
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
        if self.running:
            self.root.after(_POLL_MS, self._poll)

    def _handle(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "log":
            self._log_append(msg[1])
        elif kind == "progress":
            self._on_progress(*msg[1:])
        elif kind == "plan":
            self._on_plan(msg[1], msg[2])
        elif kind == "detect_failed":
            folder, reason = msg[1], msg[2]
            self._activity_append(f"✗ {Path(folder).name}: detection failed – {reason}")
        elif kind == "encoded":
            folder, summary = msg[1], msg[2]
            if summary.failed:
                self._activity_append(
                    f"{Path(folder).name}: {len(summary.converted)} converted, {len(summary.failed)} failed"
                )
        elif kind == "detect_done":
            self.encode_jobs.put(_FLUSH)
        elif kind == "idle":
            self._on_idle()

    def _on_progress(self, folder: str, stage: str, done: int, total: int, text: str) -> None:
        name = Path(folder).name
        self.status.configure(
            text=format_folder_progress(
                self._folder_index.get(folder, 1), self._total_folders, name, done, total, stage,
            )
        )
        self.bar.set(done / total if total else 0)
        if stage != "encode":
            return
        if self._enc_t0 is None:
            self._enc_t0 = time.monotonic()
        if text.startswith(("✓", "✗", "Cancelled")):
            self._activity_append(text)
            self._enc_done += 1
            self._update_eta()

    def _update_eta(self) -> None:
        left = self._enc_total - self._enc_done
        if self._enc_done >= 2 and left > 0 and self._enc_t0 is not None:
            elapsed = time.monotonic() - self._enc_t0
            self.eta.configure(text=format_eta(elapsed / self._enc_done * left))
        else:
            self.eta.configure(text="")

    def _on_plan(self, folder: str, plan: Plan) -> None:
        name = Path(folder).name
        now, review = split_plan(plan, self._review_all)
        self._activity_append(f"{name}: {len(now)} to encode, {len(review)} to review")
        for ep in review:
            self._add_review_row(folder, ep)
        if now:
            self._enc_total += len(now)
            self.encode_jobs.put((
                folder, Plan(input_dir=folder, episodes=now), output_subdir_for(folder, self._dst),
            ))

    def _on_idle(self) -> None:
        self.running = False
        if self._handler is not None:
            logging.getLogger("tvshow_noisemaker").removeHandler(self._handler)
            self._handler = None
        self.status.configure(
            text="Cancelled." if self.cancelling else f"Done — {self._total_folders} folder(s) processed."
        )
        self.eta.configure(text="")
        self.bar.set(0)
        self.cancelling = False
        self.start_btn.configure(text="Start")
        self.add_folder_btn.configure(state="normal")
        for btn in self._folder_row_buttons:
            btn.configure(state="normal")
        self._update_start_state()

    # -- Text boxes --------------------------------------------------------

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
        if self.running:
            self._cancel()
        self.player.stop()
        self.root.destroy()


def main() -> None:
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
