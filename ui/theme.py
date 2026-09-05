"""Dark theme constants and setup for the Sentinel GUI."""
from tkinter import ttk

DARK_BG = "#1e1e1e"
DARK_FG = "#ffffff"
ENTRY_BG = "#2d2d2d"


def setup_dark_theme(root):
    """Apply the dark clam theme to ``root`` and return the Style object."""
    style = ttk.Style()
    style.theme_use("clam")

    style.configure(".", background=DARK_BG, foreground=DARK_FG)
    style.configure("TLabel", background=DARK_BG, foreground=DARK_FG)
    style.configure("TButton", background="#333333", foreground=DARK_FG, borderwidth=1)
    style.map("TButton", background=[("active", "#ff8c00")])

    style.configure("TEntry", fieldbackground=ENTRY_BG, foreground=DARK_FG)
    style.configure("TFrame", background=DARK_BG)
    style.configure("TLabelframe", background=DARK_BG, foreground=DARK_FG)
    style.configure("TLabelframe.Label", background=DARK_BG, foreground=DARK_FG)

    style.configure(
        "Treeview",
        background="#252526",
        foreground=DARK_FG,
        fieldbackground="#252526",
        rowheight=25,
    )
    style.map("Treeview", background=[("selected", "#007acc")])

    style.configure(
        "Treeview.Heading",
        background="#333333",
        foreground=DARK_FG,
        relief="flat",
    )
    style.map("Treeview.Heading", background=[("active", "#4d4d4d")])

    root.configure(bg=DARK_BG)
    return style
