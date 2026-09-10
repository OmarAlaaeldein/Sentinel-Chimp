"""Stock Relationship Graph / Peers window for the Tk GUI.

Display-light helpers live in ``core.graph_service`` so unit tests do not need
a display. This module only builds Tk chrome.
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Optional

from core.graph_service import (
    categorize_peers,
    export_graph_temp_html,
    format_divergence_summary,
    load_graph,
    open_html_in_browser,
)
from core.stock_graph import StockGraph
from ui.theme import (
    APP_BG,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TREE_ALT,
    TREE_BG,
    _font,
)


def resolve_graph_ticker(entry_value: str, current_ticker: Optional[str]) -> str:
    """Prefer the entry box, then the loaded ticker."""
    sym = (entry_value or "").upper().strip()
    if sym:
        return sym
    return (current_ticker or "").upper().strip()


def open_stock_graph_window(
    parent,
    ticker: str,
    *,
    data_provider: Any = None,
    graph: Optional[StockGraph] = None,
    on_select_ticker: Optional[Callable[[str], None]] = None,
) -> tk.Toplevel:
    """Open peers / divergence / HTML export window for ``ticker``."""
    sym = ticker.upper().strip()
    if not sym:
        raise ValueError("Ticker is required")

    g = graph or load_graph()
    try:
        peers = categorize_peers(g, sym)
    except ValueError as exc:
        messagebox.showinfo("Stock Graph", str(exc), parent=parent)
        raise

    win = tk.Toplevel(parent)
    win.title(f"Stock Graph — {sym}")
    win.geometry("920x620")
    win.configure(bg=APP_BG)

    node = peers["node"]
    header = ttk.Frame(win)
    header.pack(fill="x", padx=14, pady=12)
    ttk.Label(
        header,
        text=f"{node['ticker']} — {node['name']}",
        font=_font(16, "bold"),
    ).pack(side="left")
    ttk.Label(
        header,
        text=f"{node['sector']} / {node['sub_industry']}  ·  {peers['peer_count']} peers",
        style="Muted.TLabel",
    ).pack(side="left", padx=12, pady=(4, 0))

    btn_row = ttk.Frame(win)
    btn_row.pack(fill="x", padx=14, pady=(0, 8))

    def _export(dim: str):
        try:
            path = export_graph_temp_html(g, center_ticker=sym, depth=1, dim=dim)
            open_html_in_browser(path)
        except Exception as exc:  # pragma: no cover - UI path
            messagebox.showerror("Export Graph", str(exc), parent=win)

    ttk.Button(
        btn_row, text="Open 2D HTML", style="Accent.TButton",
        command=lambda: _export("2d"),
    ).pack(side="left", padx=(0, 8))
    ttk.Button(
        btn_row, text="Open 3D HTML", style="Ghost.TButton",
        command=lambda: _export("3d"),
    ).pack(side="left", padx=(0, 8))

    div_frame = ttk.LabelFrame(win, text="Peer divergence (1mo)", padding=10, style="Card.TLabelframe")
    div_frame.pack(fill="x", padx=14, pady=(0, 8))
    div_var = tk.StringVar(value="Loading divergence summary…")
    div_lbl = ttk.Label(div_frame, textvariable=div_var, style="Muted.TLabel", wraplength=860, justify="left")
    div_lbl.pack(anchor="w")

    def _load_divergence():
        if data_provider is None:
            win.after(0, lambda: div_var.set("Divergence requires a live data provider (load a ticker first)."))
            return
        try:
            divs = g.analyze_divergence(data_provider, sym, period="1mo")
            lines = format_divergence_summary(divs, max_items=5)
            text = "\n".join(f"• {ln}" for ln in lines)
        except Exception as exc:
            text = f"Divergence unavailable: {exc}"
        win.after(0, lambda: div_var.set(text))

    threading.Thread(target=_load_divergence, daemon=True).start()

    columns = ("Relation", "Ticker", "Name", "Sector", "Notes")
    tree = ttk.Treeview(win, columns=columns, show="headings", height=18)
    tree.heading("Relation", text="Relation")
    tree.heading("Ticker", text="Ticker")
    tree.heading("Name", text="Name")
    tree.heading("Sector", text="Sector")
    tree.heading("Notes", text="Notes")
    tree.column("Relation", width=150, anchor="w")
    tree.column("Ticker", width=70, anchor="center")
    tree.column("Name", width=180, anchor="w")
    tree.column("Sector", width=140, anchor="w")
    tree.column("Notes", width=360, anchor="w")

    scr = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scr.set)
    tree.pack(side="left", fill="both", expand=True, padx=(14, 0), pady=(0, 14))
    scr.pack(side="right", fill="y", padx=(0, 14), pady=(0, 14))

    tree.tag_configure("odd", background=TREE_BG, foreground=TEXT_PRIMARY)
    tree.tag_configure("even", background=TREE_ALT, foreground=TEXT_PRIMARY)

    row_i = 0
    # Stable category order for readability
    preferred = [
        "COMPETITOR",
        "SUPPLIER_TO",
        "CUSTOMER_OF",
        "INFRASTRUCTURE_PARTNER",
        "POWER_PARTNER",
        "INVESTED_IN",
        "CORRELATED_PEER",
    ]
    cats = peers["peer_categories"]
    ordered = [k for k in preferred if k in cats] + [k for k in cats if k not in preferred]
    for rel in ordered:
        for p in cats[rel]:
            tag = "even" if row_i % 2 == 0 else "odd"
            tree.insert(
                "",
                "end",
                values=(
                    rel.replace("_", " "),
                    p["ticker"],
                    p["name"],
                    p["sector"],
                    (p.get("description") or "")[:120],
                ),
                tags=(tag,),
            )
            row_i += 1

    def _on_activate(_event=None):
        sel = tree.selection()
        if not sel or on_select_ticker is None:
            return
        vals = tree.item(sel[0], "values")
        if vals and len(vals) > 1:
            on_select_ticker(str(vals[1]))

    tree.bind("<Double-1>", _on_activate)

    foot = ttk.Label(
        win,
        text="Curated + Sectivia supply-chain · Supply-chain data: Sectivia (https://sectivia.com), CC BY 4.0 · docs/GRAPH_DATA.md",
        style="Muted.TLabel",
    )
    foot.pack(anchor="w", padx=14, pady=(0, 10))
    # Mute unused import warning for TEXT_MUTED in some linters
    _ = TEXT_MUTED
    return win
