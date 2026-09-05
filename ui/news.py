"""News feed / reader windows (extracted from MarketApp)."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import webbrowser

from ui.theme import DARK_BG


def open_news_feed(parent, ticker: str, news_items, on_view_content) -> tk.Toplevel:
    """Build the news list window. ``on_view_content(news_item)`` on double-click."""
    win = tk.Toplevel(parent)
    win.title(f"News Feed: {ticker}")
    win.geometry("900x600")
    win.configure(bg=DARK_BG)

    header = ttk.Frame(win)
    header.pack(fill="x", padx=10, pady=10)
    ttk.Label(
        header, text=f"Latest News ({len(news_items)})",
        font=("Arial", 16, "bold"), background="#1e1e1e", foreground="white",
    ).pack(side="left")
    ttk.Label(
        header, text="(Double-click to read)",
        font=("Arial", 10), background="#1e1e1e", foreground="gray",
    ).pack(side="left", padx=10, pady=(5, 0))

    columns = ("Date", "Source", "Headline")
    tree = ttk.Treeview(win, columns=columns, show="headings", height=20)
    tree.heading("Date", text="Date")
    tree.heading("Source", text="Source")
    tree.heading("Headline", text="Headline")
    tree.column("Date", width=120, anchor="center")
    tree.column("Source", width=100, anchor="center")
    tree.column("Headline", width=600, anchor="w")

    scr = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
    tree.configure(yscroll=scr.set)
    tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    scr.pack(side="right", fill="y", pady=10)

    tree.tag_configure("odd", background="#252526", foreground="white")
    tree.tag_configure("even", background="#333333", foreground="white")

    for i, item in enumerate(news_items):
        tag = "even" if i % 2 == 0 else "odd"
        date_str = item["published"].strftime("%Y-%m-%d %H:%M")
        tree.insert(
            "", "end", iid=i,
            values=(date_str, item["source"], item["title"]), tags=(tag,),
        )

    def on_double_click(_event):
        sel = tree.selection()
        if not sel:
            return
        on_view_content(news_items[int(sel[0])])

    tree.bind("<Double-1>", on_double_click)
    return win


def open_news_reader(parent, news_item) -> tk.Toplevel:
    """Build the single-article reader pane (behavior-preserving)."""
    reader = tk.Toplevel(parent)
    reader.title("News Reader")
    reader.geometry("600x450")
    reader.configure(bg="#1e1e1e")

    tk.Label(
        reader, text=news_item["title"], font=("Arial", 14, "bold"),
        bg="#1e1e1e", fg="white", wraplength=550, justify="left",
    ).pack(pady=15, padx=15, anchor="w")

    meta = tk.Frame(reader, bg="#1e1e1e")
    meta.pack(fill="x", padx=15)
    tk.Label(
        meta, text=f"{news_item['source']}  •  {news_item['published']}",
        bg="#1e1e1e", fg="#00e6ff", font=("Arial", 9),
    ).pack(side="left")

    tk.Label(reader, text="Snippet:", bg="#1e1e1e", fg="gray", anchor="w").pack(
        fill="x", padx=15, pady=(20, 5),
    )

    text_box = tk.Text(
        reader, height=10, bg="#252526", fg="#dddddd",
        font=("Segoe UI", 11), wrap="word", relief="flat", padx=10, pady=10,
    )
    display_text = news_item.get("summary", "")
    if len(display_text) < 10 or display_text == news_item["title"]:
        display_text = "No detailed summary available. Please read the full article below."
    text_box.insert("1.0", display_text)
    text_box.config(state="disabled")
    text_box.pack(fill="both", expand=True, padx=15, pady=5)

    btn_frame = tk.Frame(reader, bg="#1e1e1e")
    btn_frame.pack(fill="x", pady=20, padx=15)

    def open_link():
        if news_item.get("link"):
            webbrowser.open(news_item["link"])
        else:
            messagebox.showerror("Error", "No link found.")

    btn = tk.Button(
        btn_frame, text="🌐  Open Full Article in Browser", command=open_link,
        bg="#007acc", fg="white", font=("Arial", 11, "bold"),
        relief="flat", pady=8, cursor="hand2",
    )
    btn.pack(fill="x")
    return reader
