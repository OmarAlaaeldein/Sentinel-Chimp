"""Options Explorer window chrome (extracted from MarketApp.open_options_window)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
from typing import Callable, Dict, Any

from ui.theme import (
    APP_BG,
    TEXT_PRIMARY,
    TEXT_MUTED,
    BULL,
    BEAR,
    TREE_BG,
    WARNING,
    FONT_BODY,
    FONT_SMALL,
    FONT_TINY,
    _font,
)
from ui.tooltip import Tooltip


# Display headers — units in the name so Mid / Fair / EV@Ask are not ambiguous.
OPTION_COLS = (
    "Date", "Type", "Strike $", "Vol", "OI", "Mid $", "Spread%",
    "BE $", "Imp Vol", "Fair $", "EV@Ask $", "Delta", "Gamma", "Theta", "Vega", "POP", "Verdict",
)

# Short legend shown under the table (pricing model unchanged — presentation only).
ANALYZER_LEGEND = (
    "Color guide:  green = Under (Fair $ beats Ask enough to buy) · "
    "red = Over (Bid beats Fair enough to sell) · no tint = Fair (no tradeable edge).\n"
    "EV@Ask $ = Fair − Ask (buy-side edge). Mid $ = (bid+ask)/2. "
    "Fair $ uses forecast vol only (EWMA ± optional GARCH), not contract IV. "
    "Imp Vol is market IV (display / Greeks)."
)

COLUMN_HELP = {
    "Mid $": "Quote midpoint (bid+ask)/2 — the displayed market price.",
    "Fair $": "Model fair value (American BS2002) at forecast vol (EWMA ± GARCH). Not circular with Imp Vol.",
    "EV@Ask $": "Fair $ − Ask. Positive = buy-side tradeable edge before hurdles.",
    "Verdict": "Under / Over / Fair after dollar + % hurdles vs the tradeable side of the quote. See docs/LOGIC_REVIEW.md.",
    "Spread%": "(Ask − Bid) / Mid. Liquidity filter rejects spreads > 20%.",
    "BE $": "Breakeven underlying price at Mid $ (call: K+mid, put: K−mid).",
    "Imp Vol": "Listed / smile-smoothed IV for display and Greeks — not used for Fair $.",
    "POP": "Rough risk-neutral probability of finishing beyond breakeven (market IV).",
}


def build_options_explorer(
    parent,
    ticker: str,
    *,
    on_filter_expirations: Callable,
    on_scan_all: Callable,
    on_viz_calls: Callable,
    on_viz_puts: Callable,
    on_export_csv: Callable,
    on_exp_select: Callable,
    on_sort_column: Callable,
) -> Dict[str, Any]:
    """Create the Options Explorer Toplevel and return widget refs.

    Returns dict with keys: win, entry_date, exp_list, tree, legend.
    """
    win = tk.Toplevel(parent)
    win.title(f"Options Explorer: {ticker}")
    win.geometry("1280x860")
    win.configure(bg=APP_BG)

    left_panel = ttk.Frame(win, width=220)
    left_panel.pack(side="left", fill="y", padx=10, pady=10)
    right_panel = ttk.Frame(win)
    right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    ttk.Label(left_panel, text="Target Date (YYYY-MM-DD)", style="Muted.TLabel").pack(fill="x")
    entry_date = ttk.Entry(left_panel)
    entry_date.pack(fill="x", pady=(4, 8))
    entry_date.insert(0, (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d"))

    ttk.Button(
        left_panel, text="Select Prev 7 Expirations", command=on_filter_expirations,
        style="Ghost.TButton",
    ).pack(fill="x", pady=4)
    ttk.Button(
        left_panel, text="Scan ALL Undervalued", command=on_scan_all,
        style="Accent.TButton",
    ).pack(fill="x", pady=(12, 8))

    viz_frame = ttk.LabelFrame(left_panel, text="3D Visualizer", padding=10, style="Card.TLabelframe")
    viz_frame.pack(fill="x", pady=12)
    ttk.Label(
        viz_frame,
        text="Z & color = EV@Ask ($) = Fair − Ask",
        style="Muted.TLabel",
        wraplength=190,
    ).pack(fill="x", pady=(0, 6))
    ttk.Button(viz_frame, text="3D Plot (CALLS)", command=on_viz_calls, style="Period.TButton").pack(fill="x", pady=3)
    ttk.Button(viz_frame, text="3D Plot (PUTS)", command=on_viz_puts, style="Period.TButton").pack(fill="x", pady=3)

    ttk.Button(
        left_panel, text="Export Results to CSV", command=on_export_csv,
        style="Ghost.TButton",
    ).pack(fill="x", pady=6)

    ttk.Label(left_panel, text="Expirations", style="Muted.TLabel").pack(anchor="w", pady=(10, 2))
    exp_list = tk.Listbox(
        left_panel, selectmode="extended", height=22,
        bg=TREE_BG, fg=TEXT_PRIMARY, highlightthickness=0,
        borderwidth=0, relief="flat",
        font=_font(FONT_BODY),
        selectbackground="#243d4a", selectforeground=TEXT_PRIMARY,
    )
    exp_list.pack(fill="both", expand=True, pady=(0, 0))
    exp_list.bind("<<ListboxSelect>>", on_exp_select)

    table_frame = ttk.Frame(right_panel)
    table_frame.pack(side="top", fill="both", expand=True)

    tree = ttk.Treeview(table_frame, columns=OPTION_COLS, show="headings")
    col_widths = {
        "Date": 96, "Type": 58, "Strike $": 78, "Vol": 52, "OI": 52,
        "Mid $": 68, "Spread%": 68, "BE $": 72, "Imp Vol": 72,
        "Fair $": 72, "EV@Ask $": 86, "Verdict": 110,
        "Delta": 60, "Gamma": 62, "Theta": 60, "Vega": 58, "POP": 52,
    }
    for c in OPTION_COLS:
        tree.heading(
            c, text=c,
            command=lambda _c=c: on_sort_column(tree, _c, False),
        )
        tree.column(c, width=col_widths.get(c, 64), anchor="center")
        help_txt = COLUMN_HELP.get(c)
        if help_txt:
            # Heading widgets are not first-class; bind identity via column id on motion.
            pass

    scr = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scr.set)
    tree.pack(side="left", fill="both", expand=True)
    scr.pack(side="right", fill="y")

    # Soft bull/bear row tints (readable on dark surfaces)
    tree.tag_configure("green", background="#1a3a2c", foreground=BULL)
    tree.tag_configure("red", background="#3a1e24", foreground=BEAR)

    legend_frame = ttk.LabelFrame(
        right_panel, text="How to read this scan", padding=10, style="Card.TLabelframe",
    )
    legend_frame.pack(side="bottom", fill="x", pady=(10, 0))
    legend = ttk.Label(
        legend_frame,
        text=ANALYZER_LEGEND,
        style="Muted.TLabel",
        wraplength=980,
        justify="left",
    )
    legend.pack(fill="x")
    # Explicit color chips so green/red meaning is not buried
    chips = ttk.Frame(legend_frame, style="Panel.TFrame")
    chips.pack(fill="x", pady=(8, 0))
    ttk.Label(chips, text="● Under", foreground=BULL, background=APP_BG, font=_font(FONT_SMALL, "bold")).pack(side="left", padx=(0, 14))
    ttk.Label(chips, text="● Over", foreground=BEAR, background=APP_BG, font=_font(FONT_SMALL, "bold")).pack(side="left", padx=(0, 14))
    ttk.Label(chips, text="○ Fair (no edge)", foreground=TEXT_MUTED, background=APP_BG, font=_font(FONT_SMALL)).pack(side="left", padx=(0, 14))
    hint = ttk.Label(chips, text="?", style="Hint.TLabel")
    hint.pack(side="left")
    Tooltip(
        hint,
        "Under: fair−ask beats max($0.10, ½spread+$0.05) and ≥8% of mid.\n"
        "Over: same structure for bid−fair.\n"
        "Earnings rows add +$0.05 to the dollar hurdle.\n"
        "Pricing model unchanged — see docs/LOGIC_REVIEW.md.",
    )

    # Column-header help via status-like tip on the legend (hover tree → show col help)
    tip_lbl = ttk.Label(
        legend_frame,
        text="Tip: Mid $ is not Fair $. EV@Ask $ > 0 alone is not enough — Verdict applies hurdles.",
        foreground=WARNING,
        background=APP_BG,
        font=_font(FONT_TINY),
        wraplength=980,
        justify="left",
    )
    tip_lbl.pack(fill="x", pady=(6, 0))

    return {
        "win": win,
        "entry_date": entry_date,
        "exp_list": exp_list,
        "tree": tree,
        "legend": legend,
    }
