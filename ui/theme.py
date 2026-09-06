"""Modern dark trading-terminal theme for the Sentinel GUI.

Linear / Bloomberg-lite: deep slate surfaces, crisp typography, flat accents.
Not neon matrix-green.
"""
from __future__ import annotations

from tkinter import ttk

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
APP_BG = "#0b0f14"
PANEL_BG = "#12181f"
ELEVATED_BG = "#1a2330"
BORDER = "#243041"
BORDER_SOFT = "#1e2836"

TEXT_PRIMARY = "#e8eef7"
TEXT_MUTED = "#8b9bb4"
TEXT_DIM = "#5c6b82"

ACCENT = "#3dd6c6"
ACCENT_HOVER = "#2bb8aa"
ACCENT_MUTED = "#2a9d8f"

WARNING = "#f0a05a"
WARNING_HOVER = "#d8893f"
CTA = WARNING

BULL = "#3ecf8e"
BEAR = "#f07178"

ENTRY_BG = ELEVATED_BG
ENTRY_BORDER = BORDER
SELECT_BG = "#243d4a"
SELECT_FG = TEXT_PRIMARY

LOG_BG = PANEL_BG
LOG_FG = "#a8b8cc"
LOG_ACCENT = ACCENT

CHART_BG = APP_BG
CHART_FACE = "#0e141c"
CHART_GRID = "#1c2633"
CHART_SPINE = BORDER
CHART_TICK = TEXT_MUTED
CHART_TITLE = TEXT_PRIMARY
CHART_PRICE = ACCENT
CHART_CONE = "#4db8c4"

TREE_BG = ELEVATED_BG
TREE_ALT = "#151d28"
TREE_HEADING = PANEL_BG

# Backward-compatible aliases (older callers / windows)
DARK_BG = APP_BG
DARK_FG = TEXT_PRIMARY

# Font stacks — Tk picks the first available family on the host OS
FONT_UI = ("Segoe UI", "Helvetica Neue", "DejaVu Sans", "Arial", "sans-serif")
FONT_MONO = ("Cascadia Code", "Consolas", "DejaVu Sans Mono", "Courier New", "monospace")

# Readable size ladder (pt). Body 12 / headers 13–14; dark UI was previously 8–10.
FONT_BODY = 12
FONT_SMALL = 11
FONT_TINY = 10
FONT_HEADER = 13
FONT_TITLE = 14
FONT_PRICE = 32
FONT_LOG = 11
FONT_TOOLTIP = 11
TREE_ROWHEIGHT = 30


def _font(size: int, weight: str = "normal", mono: bool = False) -> tuple:
    """Build a Tk font tuple with a sensible primary family."""
    family = FONT_MONO[0] if mono else FONT_UI[0]
    if weight == "bold":
        return (family, size, "bold")
    return (family, size)


def setup_dark_theme(root):
    """Apply the dark clam theme to ``root`` and return the Style object."""
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Base
    style.configure(
        ".",
        background=APP_BG,
        foreground=TEXT_PRIMARY,
        borderwidth=0,
        focusthickness=1,
        focuscolor=ACCENT,
        font=_font(FONT_BODY),
    )
    style.configure("TFrame", background=APP_BG)
    style.configure("Panel.TFrame", background=PANEL_BG)
    style.configure("Elevated.TFrame", background=ELEVATED_BG)

    style.configure(
        "TLabel",
        background=APP_BG,
        foreground=TEXT_PRIMARY,
        font=_font(FONT_BODY),
    )
    style.configure(
        "Muted.TLabel",
        background=APP_BG,
        foreground=TEXT_MUTED,
        font=_font(FONT_SMALL),
    )
    style.configure(
        "Metric.TLabel",
        background=PANEL_BG,
        foreground=TEXT_MUTED,
        font=_font(FONT_SMALL),
    )
    style.configure(
        "MetricValue.TLabel",
        background=PANEL_BG,
        foreground=TEXT_PRIMARY,
        font=_font(FONT_BODY),
    )
    style.configure(
        "Price.TLabel",
        background=APP_BG,
        foreground=TEXT_PRIMARY,
        font=_font(FONT_PRICE, "bold"),
    )
    style.configure(
        "Status.TLabel",
        background=APP_BG,
        foreground=TEXT_MUTED,
        font=_font(FONT_TINY),
    )
    style.configure(
        "Hint.TLabel",
        background=PANEL_BG,
        foreground=WARNING,
        font=_font(FONT_SMALL),
        cursor="question_arrow",
    )

    # Buttons — flat, padded
    style.configure(
        "TButton",
        background=ELEVATED_BG,
        foreground=TEXT_PRIMARY,
        borderwidth=0,
        relief="flat",
        padding=(14, 8),
        font=_font(FONT_BODY),
    )
    style.map(
        "TButton",
        background=[("active", BORDER), ("disabled", PANEL_BG)],
        foreground=[("disabled", TEXT_DIM)],
    )

    style.configure(
        "Accent.TButton",
        background=ACCENT,
        foreground=APP_BG,
        borderwidth=0,
        relief="flat",
        padding=(16, 8),
        font=_font(FONT_BODY, "bold"),
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_HOVER), ("disabled", BORDER_SOFT)],
        foreground=[("disabled", TEXT_DIM)],
    )

    style.configure(
        "Ghost.TButton",
        background=PANEL_BG,
        foreground=TEXT_MUTED,
        borderwidth=1,
        relief="flat",
        padding=(14, 8),
        font=_font(FONT_BODY),
    )
    style.map(
        "Ghost.TButton",
        background=[("active", ELEVATED_BG)],
        foreground=[("active", TEXT_PRIMARY), ("disabled", TEXT_DIM)],
        bordercolor=[("!disabled", BORDER), ("active", ACCENT_MUTED)],
    )

    style.configure(
        "Period.TButton",
        background=PANEL_BG,
        foreground=TEXT_MUTED,
        borderwidth=0,
        relief="flat",
        padding=(10, 5),
        font=_font(FONT_SMALL),
    )
    style.map(
        "Period.TButton",
        background=[("active", ELEVATED_BG), ("pressed", BORDER)],
        foreground=[("active", TEXT_PRIMARY), ("pressed", ACCENT)],
    )

    style.configure(
        "CTA.TButton",
        background=CTA,
        foreground=APP_BG,
        borderwidth=0,
        relief="flat",
        padding=(16, 9),
        font=_font(FONT_BODY, "bold"),
    )
    style.map(
        "CTA.TButton",
        background=[("active", WARNING_HOVER), ("disabled", BORDER_SOFT)],
        foreground=[("disabled", TEXT_DIM)],
    )

    # Entry
    style.configure(
        "TEntry",
        fieldbackground=ENTRY_BG,
        foreground=TEXT_PRIMARY,
        insertcolor=TEXT_PRIMARY,
        borderwidth=1,
        relief="flat",
        padding=6,
    )
    style.map(
        "TEntry",
        fieldbackground=[("focus", ELEVATED_BG)],
        bordercolor=[("focus", ACCENT)],
        lightcolor=[("focus", ACCENT)],
        darkcolor=[("focus", ACCENT)],
    )

    # Checkbutton — quieter chrome
    style.configure(
        "TCheckbutton",
        background=APP_BG,
        foreground=TEXT_MUTED,
        font=_font(FONT_SMALL),
        padding=3,
    )
    style.map(
        "TCheckbutton",
        background=[("active", APP_BG)],
        foreground=[("active", TEXT_PRIMARY)],
        indicatorcolor=[("selected", ACCENT), ("!selected", ELEVATED_BG)],
    )

    # Card-like LabelFrame
    style.configure(
        "TLabelframe",
        background=PANEL_BG,
        foreground=TEXT_MUTED,
        borderwidth=1,
        relief="flat",
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
    )
    style.configure(
        "TLabelframe.Label",
        background=PANEL_BG,
        foreground=TEXT_MUTED,
        font=_font(FONT_SMALL, "bold"),
    )
    style.configure(
        "Card.TLabelframe",
        background=PANEL_BG,
        foreground=TEXT_MUTED,
        borderwidth=1,
        relief="flat",
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        padding=12,
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=PANEL_BG,
        foreground=ACCENT,
        font=_font(FONT_HEADER, "bold"),
    )

    # Treeview
    style.configure(
        "Treeview",
        background=TREE_BG,
        foreground=TEXT_PRIMARY,
        fieldbackground=TREE_BG,
        borderwidth=0,
        rowheight=TREE_ROWHEIGHT,
        font=_font(FONT_SMALL),
    )
    style.map(
        "Treeview",
        background=[("selected", SELECT_BG)],
        foreground=[("selected", SELECT_FG)],
    )
    style.configure(
        "Treeview.Heading",
        background=TREE_HEADING,
        foreground=TEXT_MUTED,
        relief="flat",
        borderwidth=0,
        font=_font(FONT_SMALL, "bold"),
        padding=6,
    )
    style.map(
        "Treeview.Heading",
        background=[("active", ELEVATED_BG)],
        foreground=[("active", TEXT_PRIMARY)],
    )

    # PanedWindow / Scrollbar
    style.configure("TPanedwindow", background=APP_BG)
    style.configure(
        "TScrollbar",
        background=ELEVATED_BG,
        troughcolor=APP_BG,
        borderwidth=0,
        arrowsize=12,
    )
    style.map(
        "TScrollbar",
        background=[("active", BORDER)],
    )

    if root is not None:
        root.configure(bg=APP_BG)
        try:
            root.option_add("*Font", _font(FONT_BODY))
            root.option_add("*Text.Font", _font(FONT_LOG, mono=True))
            root.option_add("*Listbox.Background", TREE_BG)
            root.option_add("*Listbox.Foreground", TEXT_PRIMARY)
            root.option_add("*Listbox.SelectBackground", SELECT_BG)
            root.option_add("*Listbox.SelectForeground", SELECT_FG)
            root.option_add("*Listbox.BorderWidth", 0)
            root.option_add("*Listbox.HighlightThickness", 0)
        except Exception:
            pass

    return style



def log_colors() -> dict:
    """Colors for the system log Text widget."""
    return {
        "bg": LOG_BG,
        "fg": LOG_FG,
        "insertbackground": TEXT_PRIMARY,
        "selectbackground": SELECT_BG,
        "selectforeground": SELECT_FG,
        "highlightthickness": 0,
        "relief": "flat",
        "borderwidth": 0,
    }


def chart_colors() -> dict:
    """Matplotlib chrome colors matching the terminal palette."""
    return {
        "figure": CHART_BG,
        "face": CHART_FACE,
        "grid": CHART_GRID,
        "spine": CHART_SPINE,
        "tick": CHART_TICK,
        "title": CHART_TITLE,
        "price": CHART_PRICE,
        "cone": CHART_CONE,
        "muted": TEXT_MUTED,
    }
