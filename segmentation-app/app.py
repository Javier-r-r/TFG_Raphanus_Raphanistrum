"""
Entry point for the segmentation inference application.
"""
import tkinter as tk
from tkinter import ttk

from main import SegTkApp


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = SegTkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
