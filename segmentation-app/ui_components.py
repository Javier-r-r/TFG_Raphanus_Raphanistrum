"""
Custom UI components for the segmentation app.
"""
import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget for handling content that exceeds window size."""
    
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vscroll.pack(side="right", fill="y")

        self.container = ttk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.container, anchor="nw")

        # Expand inner frame to canvas width
        def on_canvas_configure(event):
            self.canvas.itemconfigure(self.window_id, width=event.width)
        self.canvas.bind("<Configure>", on_canvas_configure)

        # Update scroll region on frame resize
        def on_frame_configure(_event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.container.bind("<Configure>", on_frame_configure)

        # Mouse wheel (Windows and Linux)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)  # Linux
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)  # Linux

    def _on_mousewheel(self, event):
        if event.num == 4:  # Linux scroll up
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:  # Linux scroll down
            self.canvas.yview_scroll(3, "units")
        else:
            # Windows/Mac: event.delta positive (up) or negative (down)
            self.canvas.yview_scroll(-int(event.delta / 120) * 3, "units")
