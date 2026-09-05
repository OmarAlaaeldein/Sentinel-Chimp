"""Hover tooltip helper for ttk/tk widgets."""
import tkinter as tk
from tkinter import ttk


class Tooltip:
    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._on_enter)
        widget.bind("<Leave>", self._on_leave)

    def _on_enter(self, _):
        self._after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, _):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        self._hide()

    def _show(self):
        if self._tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 10
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        lbl = ttk.Label(self._tip, text=self.text, justify="left", relief="solid", borderwidth=1,
                        background="#333333", foreground="#ffffff", padding=2, wraplength=300)
        lbl.pack(ipadx=1)

    def _hide(self):
        if self._tip:
            self._tip.destroy()
            self._tip = None

    def set_text(self, text):
        """Update tooltip body (e.g. dynamic 'why this vol' hint)."""
        self.text = text
        self._hide()

