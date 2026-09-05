"""Options Explorer window chrome (extracted from MarketApp.open_options_window)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
from typing import Callable, Dict, Any

from ui.theme import DARK_BG


OPTION_COLS = (
    "Date", "Type", "Strike", "Vol", "OI", "Price", "Spread%",
    "Breakeven", "Imp Vol", "Fair", "EV", "Delta", "Gamma", "Theta", "Vega", "POP", "Verdict",
)


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

    Returns dict with keys: win, entry_date, exp_list, tree.
    """
    win = tk.Toplevel(parent)
    win.title(f"Options Explorer: {ticker}")
    win.geometry("1200x800")
    win.configure(bg=DARK_BG)

    left_panel = ttk.Frame(win, width=200)
    left_panel.pack(side="left", fill="y", padx=5, pady=5)
    right_panel = ttk.Frame(win)
    right_panel.pack(side="right", fill="both", expand=True, padx=5, pady=5)

    ttk.Label(left_panel, text="Target Date (YYYY-MM-DD):").pack(fill="x")
    entry_date = ttk.Entry(left_panel)
    entry_date.pack(fill="x", pady=2)
    entry_date.insert(0, (datetime.now() + timedelta(days=180)).strftime("%Y-%m-%d"))

    ttk.Button(
        left_panel, text="Select Prev 7 Expirations", command=on_filter_expirations,
    ).pack(fill="x", pady=5)
    ttk.Button(
        left_panel, text="⚡ Scan ALL Undervalued", command=on_scan_all,
    ).pack(fill="x", pady=20)

    viz_frame = ttk.LabelFrame(left_panel, text="3D Visualizer", padding=5)
    viz_frame.pack(fill="x", pady=20)
    ttk.Button(viz_frame, text="3D Plot (CALLS)", command=on_viz_calls).pack(fill="x", pady=2)
    ttk.Button(viz_frame, text="3D Plot (PUTS)", command=on_viz_puts).pack(fill="x", pady=2)

    ttk.Button(
        left_panel, text="💾 Export Results to CSV", command=on_export_csv,
    ).pack(fill="x", pady=5)

    exp_list = tk.Listbox(
        left_panel, selectmode="extended", height=25,
        bg="#252526", fg="white", highlightthickness=0,
    )
    exp_list.pack(fill="both", expand=True)
    exp_list.bind("<<ListboxSelect>>", on_exp_select)

    tree = ttk.Treeview(right_panel, columns=OPTION_COLS, show="headings")
    col_widths = {
        "Date": 90, "Breakeven": 75, "Verdict": 65,
        "Delta": 55, "Gamma": 55, "Theta": 55, "Vega": 55, "POP": 50,
        "OI": 55, "Spread%": 55,
    }
    for c in OPTION_COLS:
        tree.heading(
            c, text=c,
            command=lambda _c=c: on_sort_column(tree, _c, False),
        )
        tree.column(c, width=col_widths.get(c, 60), anchor="center")

    scr = ttk.Scrollbar(right_panel, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scr.set)
    tree.pack(side="left", fill="both", expand=True)
    scr.pack(side="right", fill="y")

    tree.tag_configure("green", background="#8fbc8f", foreground="black")
    tree.tag_configure("red", background="#e57373", foreground="black")

    return {"win": win, "entry_date": entry_date, "exp_list": exp_list, "tree": tree}
