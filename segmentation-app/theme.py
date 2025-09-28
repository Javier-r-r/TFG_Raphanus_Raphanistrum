"""
Theme and styling configuration for the segmentation app.
"""
from tkinter import ttk


def create_theme(style: ttk.Style):
    """Create and apply custom theme inspired by modern web UI."""
    ACCENT = "#22c55e"        
    ACCENT_DARK = "#16a34a"   
    SURFACE = "#ffffff"
    SURFACE_MUTED = "#f4f5f7"
    BORDER = "#e5e7eb"
    TEXT = "#111827"
    TEXT_MUTED = "#6b7280"
    HEADER_BG = "#111827"
    HEADER_FG = "#ffffff"

    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Global
    style.configure(".", font=("Segoe UI", 10))

    # Header
    style.configure("Header.TFrame", background=HEADER_BG)
    style.configure("Header.TLabel", background=HEADER_BG, foreground=HEADER_FG, font=("Segoe UI Semibold", 12))

    # Card-like frames
    style.configure("Card.TFrame", background=SURFACE, relief="solid", borderwidth=1)
    style.map("Card.TFrame", background=[("active", SURFACE)])
    style.configure("CardTitle.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 11))
    style.configure("CardDesc.TLabel", background=SURFACE, foreground=TEXT_MUTED, font=("Segoe UI", 9))

    # Buttons
    style.configure("Accent.TButton", foreground="#ffffff", background=ACCENT)
    style.map("Accent.TButton",
              background=[("active", ACCENT_DARK), ("pressed", ACCENT_DARK)],
              foreground=[("disabled", "#e5e7eb")])
    style.configure("Secondary.TButton", foreground=TEXT, background=SURFACE_MUTED)
    style.map("Secondary.TButton",
              background=[("active", "#e5e7eb"), ("pressed", "#e5e7eb")])

    # Labels
    style.configure("Muted.TLabel", background=SURFACE, foreground=TEXT_MUTED)
    style.configure("Badge.TLabel", background="#eef2f7", foreground=TEXT, padding=(6, 2))
    style.configure("Section.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 10))

    # Notebook (tabs)
    style.configure("Preview.TNotebook", background=SURFACE, borderwidth=0)
    style.configure("Preview.TNotebook.Tab", padding=(12, 6))
    style.map("Preview.TNotebook.Tab",
              background=[("selected", SURFACE), ("!selected", SURFACE_MUTED)])

    # Treeview (metrics table)
    style.configure("Metrics.Treeview", background=SURFACE, fieldbackground=SURFACE, bordercolor=BORDER)
    style.configure("Metrics.Treeview.Heading", font=("Segoe UI Semibold", 10))
