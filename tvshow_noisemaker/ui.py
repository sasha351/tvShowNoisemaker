"""Look of the GUI: colour tokens, the CustomTkinter theme built from them, and two canvas widgets.

Direction (from an unslop pass, https://github.com/heydamir/unslop, CC BY 4.0, by Damir Sokolovsky):
the app is a small editing bay.  Chrome stays on one graphite ramp; the only colour is a
tape-deck amber that marks the primary action, progress, and the part of an episode you keep.
Timecodes are always set in mono.

Every colour is a ``(light, dark)`` pair, the form CustomTkinter resolves per appearance mode.
"""
from __future__ import annotations

import copy
import tkinter as tk
import tkinter.font as tkfont
from typing import Callable, Optional, Sequence, Tuple

import customtkinter as ctk

Color = Tuple[str, str]

# -- Tokens: graphite ramp, one accent, toned semantics ------------------------------

BG: Color = ("#fafafa", "#141415")
CARD: Color = ("#ffffff", "#19191b")
RAISED: Color = ("#f4f4f5", "#232325")
RAISED_HOVER: Color = ("#ebebed", "#2c2c2f")
HAIRLINE: Color = ("#e5e5e5", "#2a2a2c")
TEXT: Color = ("#171717", "#ededed")
TEXT_2: Color = ("#525252", "#a1a1a6")
MUTED: Color = ("#737373", "#85858b")
FAINT: Color = ("#c4c4c4", "#4a4a4e")

ACCENT: Color = ("#e09a2f", "#e8a33d")
ACCENT_HOVER: Color = ("#cf8a20", "#f0b357")
ON_ACCENT: Color = ("#1c1407", "#1c1407")
NEGATIVE: Color = ("#dc2626", "#e5484d")

# Spacing steps used by the layout (4px grid).
PAGE = 24
GROUP = 20
STACK = 8

_MONO_CANDIDATES = ("SF Mono", "Menlo", "JetBrains Mono", "IBM Plex Mono", "DejaVu Sans Mono", "Consolas")


def theme() -> dict:
    """A full CustomTkinter theme dict: the built-in one, re-tokenised."""
    ctk.ThemeManager.load_theme("blue")  # every key a widget may look up, then overridden
    t = copy.deepcopy(ctk.ThemeManager.theme)
    secondary = {"fg_color": RAISED, "hover_color": RAISED_HOVER, "text_color": TEXT}
    t["CTk"]["fg_color"] = BG
    t["CTkToplevel"]["fg_color"] = BG
    t["CTkFrame"].update(corner_radius=12, border_width=0, fg_color=CARD, top_fg_color=RAISED, border_color=HAIRLINE)
    t["CTkButton"].update(
        corner_radius=8, border_width=1, border_color=HAIRLINE, text_color_disabled=FAINT, **secondary,
    )
    t["CTkLabel"].update(text_color=TEXT)
    t["CTkEntry"].update(
        corner_radius=8, border_width=1, fg_color=RAISED, border_color=HAIRLINE,
        text_color=TEXT, placeholder_text_color=MUTED,
    )
    t["CTkCheckBox"].update(
        corner_radius=4, border_width=1, fg_color=ACCENT, hover_color=ACCENT_HOVER, border_color=MUTED,
        checkmark_color=ON_ACCENT, text_color=TEXT, text_color_disabled=FAINT,
    )
    t["CTkProgressBar"].update(corner_radius=2, border_width=0, fg_color=RAISED, progress_color=ACCENT)
    t["CTkOptionMenu"].update(
        corner_radius=8, fg_color=RAISED, button_color=RAISED, button_hover_color=RAISED_HOVER,
        text_color=TEXT, text_color_disabled=FAINT,
    )
    t["CTkScrollbar"].update(button_color=HAIRLINE, button_hover_color=FAINT)
    t["CTkTextbox"].update(
        corner_radius=12, border_width=1, fg_color=CARD, border_color=HAIRLINE, text_color=TEXT_2,
        scrollbar_button_color=HAIRLINE, scrollbar_button_hover_color=FAINT,
    )
    t["CTkScrollableFrame"].update(label_fg_color=CARD)
    t["DropdownMenu"].update(fg_color=CARD, hover_color=RAISED, text_color=TEXT)
    return t


def apply_theme() -> None:
    ctk.ThemeManager.theme = theme()


def mono_family(root: tk.Misc) -> str:
    families = set(tkfont.families(root))
    return next((f for f in _MONO_CANDIDATES if f in families), "TkFixedFont")


# -- Canvas widgets -----------------------------------------------------------------


def _round_rect(canvas: tk.Canvas, x0, y0, x1, y1, r, **kw) -> int:
    r = max(0.0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = (
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
        x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
    )
    return canvas.create_polygon(pts, smooth=True, **kw)


class _Ink(tk.Canvas):
    """A canvas painted from tokens that repaints when the appearance mode flips."""

    def __init__(self, master, width: int, height: int, surface: Color, **kw) -> None:
        self._scale = ctk.ScalingTracker.get_widget_scaling(master)
        super().__init__(
            master, width=int(width * self._scale), height=int(height * self._scale),
            highlightthickness=0, bd=0, **kw,
        )
        self._surface = surface
        self._dark = ctk.get_appearance_mode() == "Dark"
        ctk.AppearanceModeTracker.add(self._on_mode, self)
        self.bind("<Configure>", lambda _e: self.redraw(), add="+")
        self.bind("<Destroy>", self._on_destroy, add="+")

    def c(self, pair: Color) -> str:
        return pair[1] if self._dark else pair[0]

    def _on_mode(self, mode: str) -> None:
        self._dark = mode == "Dark"
        self.redraw()

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            ctk.AppearanceModeTracker.remove(self._on_mode)

    def redraw(self) -> None:
        super().configure(bg=self.c(self._surface))


class IconButton(_Ink):
    """A 28px square button with a 16px line glyph drawn at a 1.5 stroke (lucide geometry)."""

    SIZE = 28

    def __init__(self, master, kind: str, command: Callable[[], None], surface: Color = BG) -> None:
        super().__init__(master, self.SIZE, self.SIZE, surface, takefocus=1, cursor="hand2")
        self._kind, self._command = kind, command
        self._state, self._hover, self._focus = "normal", False, False
        for seq, fn in (
            ("<Enter>", lambda _e: self._set(hover=True)), ("<Leave>", lambda _e: self._set(hover=False)),
            ("<FocusIn>", lambda _e: self._set(focus=True)), ("<FocusOut>", lambda _e: self._set(focus=False)),
            ("<ButtonRelease-1>", self._invoke), ("<KeyPress-space>", self._invoke), ("<Return>", self._invoke),
        ):
            self.bind(seq, fn, add="+")
        self.redraw()

    def _set(self, **kw) -> None:
        self._hover = kw.get("hover", self._hover)
        self._focus = kw.get("focus", self._focus)
        self.redraw()

    def _invoke(self, _event=None) -> None:
        if self._state != "disabled":
            self._command()

    def configure(self, cnf=None, **kw):  # type: ignore[override]
        if "state" in kw:
            self._state = kw.pop("state")
            super().configure(cursor="" if self._state == "disabled" else "hand2")
            self.redraw()
        return super().configure(cnf, **kw) if (cnf or kw) else None

    config = configure

    def redraw(self) -> None:
        super().redraw()
        self.delete("all")
        s, size = self._scale, self.SIZE * self._scale
        enabled = self._state != "disabled"
        if enabled and self._hover:
            _round_rect(self, 0, 0, size, size, 6 * s, fill=self.c(RAISED_HOVER), outline="")
        if self._focus:
            _round_rect(self, 1, 1, size - 1, size - 1, 6 * s, fill="", outline=self.c(ACCENT), width=1)
        ink = self.c(FAINT if not enabled else TEXT if self._hover else MUTED)
        # Map lucide's 24-unit grid onto a 16px glyph centred in the button.
        k, o = 16 / 24 * s, 6 * s

        def p(*xy: float) -> Sequence[float]:
            return [o + v * k for v in xy]

        line = dict(fill=ink, width=1.5 * s, capstyle="round", joinstyle="round")
        if self._kind == "play":
            self.create_polygon(p(7, 4, 19, 12, 7, 20), outline=ink, fill="", width=1.5 * s, joinstyle="round")
        elif self._kind == "stop":
            _round_rect(self, *p(6, 6, 18, 18), 2 * s, outline=ink, fill="", width=1.5 * s)
        elif self._kind == "remove":
            self.create_line(p(18, 6, 6, 18), **line)
            self.create_line(p(6, 6, 18, 18), **line)


class CutStrip(_Ink):
    """The episode as a bar: the parts that are kept, the intro and credits that are cut.

    Kept spans (cold open and the episode body) are amber once the cut is complete,
    secondary grey while a time is still missing, and faint when the episode is skipped.
    """

    HEIGHT = 16
    _BAR = 6

    def __init__(self, master, surface: Color = CARD) -> None:
        super().__init__(master, 200, self.HEIGHT, surface)
        self._marks: Tuple[Optional[float], ...] = (None, None, None, None)
        self._skipped = False

    def show(
        self, duration: Optional[float], intro_start: Optional[float], intro_end: Optional[float],
        outro_start: Optional[float], skipped: bool = False,
    ) -> None:
        self._marks = (duration, intro_start, intro_end, outro_start)
        self._skipped = skipped
        self.redraw()

    def redraw(self) -> None:
        super().redraw()
        self.delete("all")
        w, h, s = self.winfo_width(), self.HEIGHT * self._scale, self._scale
        if w < 8:
            return
        duration, i_start, i_end, o_start = self._marks
        known = [v for v in (i_start, i_end, o_start) if v is not None]
        end = duration or (max(known) * 1.04 if known else 0.0)
        if end <= 0:
            return

        def x(t: float) -> float:
            return max(0.0, min(w, t / end * w))

        y0, y1, r = (h - self._BAR * s) / 2, (h + self._BAR * s) / 2, 3 * s
        _round_rect(self, 0, y0, w, y1, r, fill=self.c(RAISED), outline=self.c(HAIRLINE))
        complete = i_end is not None and o_start is not None and o_start > i_end
        kept = self.c(FAINT if self._skipped else ACCENT if complete else TEXT_2)
        spans = []
        if i_start and i_end and i_start < i_end:
            spans.append((0.0, i_start))  # cold open before the intro
        spans.append((i_end if i_end is not None else 0.0, o_start if o_start is not None else end))
        for a, b in spans:
            if b > a:
                _round_rect(self, x(a), y0, x(b), y1, r, fill=kept, outline="")
        for t in known:
            self.create_line(x(t), 0, x(t), h, fill=self.c(TEXT_2), width=1)
