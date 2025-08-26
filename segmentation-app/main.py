"""
Main application class for the segmentation inference GUI.
"""
import os
import json
import csv
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk
import torch

from models import CamVidModel, load_config_from_dir, find_stats_in_dir
from models import preprocess_image_pil, postprocess_mask, color_overlay
from ui_components import ScrollableFrame
from theme import create_theme
from metrics import compute_normalized_metrics
from metrics import generate_petal_mask_from_rgb


class SegTkApp:
    """Main segmentation inference application."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Segmentation Mask Inference")
        root.geometry("1100x820")
        root.minsize(980, 680)

        # Apply theme
        style = ttk.Style()
        create_theme(style)

        # Tk control variables must exist before building widgets that bind to them
        self.force_imagenet = tk.BooleanVar(value=False)
        self.default_resize_var = tk.StringVar(value="640")
        self.arch_var = tk.StringVar(value="Unet")
        self.encoder_var = tk.StringVar(value="resnet34")
        self.weights_path_var = tk.StringVar(value="")
        self.threshold = tk.DoubleVar(value=0.5)
        self.alpha = tk.DoubleVar(value=0.5)

        # Scrollable wrapper (page-level)
        self.scroll = ScrollableFrame(root)
        self.scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(self.scroll.container, style="Header.TFrame")
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text="Segmentation Mask Inference", style="Header.TLabel").pack(side=tk.LEFT, padx=16, pady=10)

        # Content grid (two columns like the web app)
        content = ttk.Frame(self.scroll.container)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=14)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        # Left column: Input & Controls (Card)
        self._build_input_controls(content)

        # Right column: Preview (Card with tabs)
        self._build_preview_card(content)

        # Bottom row: Batch, Metrics, and Debug (three cards)
        self._build_batch_card(self.scroll.container)
        self._build_metrics_card(self.scroll.container)
        self._build_debug_card(self.scroll.container)

        # Model/meta state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[torch.nn.Module] = None
        self.debug_meta: Dict[str, Any] = {}

        # Inference state
        self.pil_input: Optional[Image.Image] = None
        self.np_input_rgb: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_overlay: Optional[np.ndarray] = None
        self.current_metrics: Optional[Dict[str, Any]] = None

        # Instructions for the debug console
        self._clear_debug()
        self._write_debug(
            "Welcome to the Segmentation Mask Inference App!\n"
            "\n"
            "Instructions:\n"
            "1. Load model weights (.pth) using the 'Browse' and 'Load Model' buttons.\n"
            "2. Open an image for segmentation.\n"
            "3. Adjust threshold and overlay alpha as needed.\n"
            "4. Click 'Segment' to generate the mask and overlay.\n"
            "5. Save the mask or overlay if desired.\n"
            "6. Use 'Batch Processing' to process all images in a folder.\n"
            "7. Use the 'Metrics' section to compute and save mask metrics.\n"
            "\n"
            "All actions, errors, and progress will be shown here."
        )

    # -------- UI building blocks --------

    def _build_input_controls(self, parent: ttk.Frame):
        """Build the input controls card."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 12))

        # Title + description
        ttk.Label(card, text="Input", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=6, sticky="w")
        ttk.Label(card, text="Select model weights and an image, set threshold and alpha, then run segmentation.",
                  style="CardDesc.TLabel").grid(row=1, column=0, columnspan=6, sticky="w", pady=(2, 10))

        # Weights row
        ttk.Label(card, text="Weights (.pth):", style="Section.TLabel").grid(row=2, column=0, sticky="w", pady=2)
        self.e_weights = ttk.Entry(card, textvariable=self.weights_path_var, width=56)
        self.e_weights.grid(row=2, column=1, columnspan=3, sticky="we", padx=6)
        ttk.Button(card, text="Browse", command=self.on_browse_weights, style="Secondary.TButton").grid(row=2, column=4, padx=4)
        ttk.Button(card, text="Load Model", command=self.on_load_model, style="Accent.TButton").grid(row=2, column=5, padx=2)

        # Model status indicator row
        status_frame = ttk.Frame(card)
        status_frame.grid(row=3, column=0, columnspan=6, sticky="we", pady=(4, 2))
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="Model Status:", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.model_status_label = ttk.Label(status_frame, text="Not loaded", style="Muted.TLabel")
        self.model_status_label.grid(row=0, column=1, sticky="w", padx=(8, 0))
        
        # Model info display (initially hidden)
        self.model_info_frame = ttk.Frame(card)
        self.model_info_frame.grid(row=4, column=0, columnspan=6, sticky="we", pady=(2, 6))
        self.model_info_label = ttk.Label(self.model_info_frame, text="", style="Muted.TLabel", wraplength=600)
        self.model_info_label.pack(anchor="w")
        self.model_info_frame.grid_remove()  # Hide initially

        # Arch/Encoder row
        ttk.Label(card, text="Arch:", style="Muted.TLabel").grid(row=5, column=0, sticky="w", pady=(6, 2))
        ttk.Entry(card, textvariable=self.arch_var, width=18).grid(row=5, column=1, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(card, text="Encoder:", style="Muted.TLabel").grid(row=5, column=2, sticky="e", pady=(6, 2))
        ttk.Entry(card, textvariable=self.encoder_var, width=18).grid(row=5, column=3, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(card, text="Default Resize (px):", style="Muted.TLabel").grid(row=5, column=4, sticky="e", pady=(6, 2))
        ttk.Entry(card, textvariable=self.default_resize_var, width=10).grid(row=5, column=5, sticky="w", padx=4, pady=(6, 2))

        # Actions row
        row = 6
        actions = ttk.Frame(card, style="Card.TFrame")
        actions.grid(row=row, column=0, columnspan=6, sticky="we", pady=(8, 2))
        ttk.Button(actions, text="Open Image", command=self.on_open_image, style="Secondary.TButton").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="Segment", command=self.on_segment, style="Accent.TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Mask", command=self.on_save_mask, style="Secondary.TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Overlay", command=self.on_save_overlay, style="Secondary.TButton").pack(side=tk.LEFT, padx=6)

        # Remove threshold control from here
        # (delete the following block)
        # thr_frame = ttk.Frame(card)
        # thr_frame.grid(row=7, column=0, columnspan=6, sticky="we", pady=4)
        # thr_frame.columnconfigure(1, weight=1)
        # ttk.Label(thr_frame, text="Threshold").grid(row=0, column=0, sticky="w")
        # self.thr_entry = ttk.Entry(thr_frame, width=5, textvariable=self.threshold)
        # self.thr_entry.grid(row=0, column=2, padx=(0, 10), sticky="e")
        # ttk.Scale(thr_frame, from_=0.0, to=1.0, variable=self.threshold,
        #         orient=tk.HORIZONTAL, command=self._on_slider_change).grid(row=0, column=1, sticky="we", padx=10)
        # self.thr_entry.bind('<FocusOut>', self._validate_threshold)
        # self.thr_entry.bind('<Return>', self._validate_threshold)

        for c in range(6):
            card.columnconfigure(c, weight=1)

    def _build_preview_card(self, parent: ttk.Frame):
        """Build the preview card with tabbed interface."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=(0, 12))
        ttk.Label(card, text="Preview", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text="Input, mask, and overlay", style="CardDesc.TLabel").pack(anchor="w", pady=(2, 8))

        # Notebook with three tabs
        self.preview_nb = ttk.Notebook(card, style="Preview.TNotebook")
        self.preview_nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Input Tab
        tab_input = ttk.Frame(self.preview_nb)
        self.preview_nb.add(tab_input, text="Input")
        self.label_input = ttk.Label(tab_input, background="#222")
        self.label_input.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Mask Tab
        tab_mask = ttk.Frame(self.preview_nb)
        self.preview_nb.add(tab_mask, text="Mask")
        self.label_mask = ttk.Label(tab_mask, background="#222")
        self.label_mask.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Overlay Tab
        tab_overlay = ttk.Frame(self.preview_nb)
        self.preview_nb.add(tab_overlay, text="Overlay")
        self.label_overlay = ttk.Label(tab_overlay, background="#222")
        self.label_overlay.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        
        # Insert threshold slider/entry here, below preview
        thr_frame = ttk.Frame(card)
        thr_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        thr_frame.columnconfigure(1, weight=1)
        ttk.Label(thr_frame, text="Threshold").grid(row=0, column=0, sticky="w")
        self.thr_entry = ttk.Entry(thr_frame, width=5, textvariable=self.threshold)
        self.thr_entry.grid(row=0, column=2, padx=(0, 10), sticky="e")
        ttk.Scale(thr_frame, from_=0.0, to=1.0, variable=self.threshold,
                  orient=tk.HORIZONTAL, command=self._on_slider_change).grid(row=0, column=1, sticky="we", padx=10)
        self.thr_entry.bind('<FocusOut>', self._validate_threshold)
        self.thr_entry.bind('<Return>', self._validate_threshold)

        # Alpha control (now below threshold)
        alpha_frame = ttk.Frame(card)
        alpha_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        alpha_frame.columnconfigure(1, weight=1)
        ttk.Label(alpha_frame, text="Overlay alpha").grid(row=0, column=0, sticky="w")
        self.alpha_entry = ttk.Entry(alpha_frame, width=5, textvariable=self.alpha)
        self.alpha_entry.grid(row=0, column=2, padx=(0, 10), sticky="e")
        ttk.Scale(alpha_frame, from_=0.0, to=1.0, variable=self.alpha,
                  orient=tk.HORIZONTAL, command=self._on_slider_change).grid(row=0, column=1, sticky="we", padx=10)
        self.alpha_entry.bind('<FocusOut>', self._validate_alpha)
        self.alpha_entry.bind('<Return>', self._validate_alpha)

    def _build_metrics_card(self, parent: ttk.Widget):
        """Build the metrics computation card."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(side=tk.TOP, fill=tk.X, padx=14, pady=(0, 12))
        ttk.Label(card, text="Metrics (from mask)", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text="Compute personalized metrics directly from a binary mask (0/255).",
                  style="CardDesc.TLabel").pack(anchor="w", pady=(2, 8))

        # Buttons row
        btns = ttk.Frame(card)
        btns.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        ttk.Button(btns, text="Metrics from predicted mask", command=self.on_metrics_from_pred, style="Accent.TButton").pack(side=tk.LEFT)
        ttk.Button(btns, text="Metrics from mask file...", command=self.on_metrics_from_file, style="Secondary.TButton").pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save Metrics to CSV", command=self.on_save_metrics, style="Secondary.TButton").pack(side=tk.LEFT, padx=8)

        # Body: left preview, right table
        body = ttk.Frame(card)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: preview
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 8))

        ttk.Label(left, text="Metrics image", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        self.metrics_preview_title = ttk.Label(left, text="No image selected", style="Muted.TLabel")
        self.metrics_preview_title.pack(anchor="w", pady=(0, 6))

        self.metrics_preview = ttk.Label(left, background="#222", relief="solid")
        self.metrics_preview.pack(fill=tk.BOTH, expand=False)

        # Right: table
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.metrics_tree = ttk.Treeview(right, columns=("metric", "value"), show="headings", height=12, style="Metrics.Treeview")
        self.metrics_tree.heading("metric", text="Metric")
        self.metrics_tree.heading("value", text="Value")
        self.metrics_tree.column("metric", width=420, anchor="w")
        self.metrics_tree.column("value", width=140, anchor="e")
        self.metrics_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_batch_card(self, parent: ttk.Widget):
        """Build the batch processing card."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(side=tk.TOP, fill=tk.X, padx=14, pady=(0, 12))
        ttk.Label(card, text="Batch Processing", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text="Processes all images in a folder and saves masks + metrics to 'results' subfolder",
                  style="CardDesc.TLabel").pack(anchor="w", pady=(2, 8))

        batch_frame = ttk.Frame(card, style="Card.TFrame")
        batch_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        ttk.Button(batch_frame, text="Select Folder & Process All Images", command=self.on_batch_process, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 6))

        # Progress bar (initially hidden)
        self.progress_frame = ttk.Frame(card)
        self.progress_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        self.progress_label = ttk.Label(self.progress_frame, text="", style="Muted.TLabel")
        self.progress_label.pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        self.progress_frame.pack_forget()  # Hide initially

    def _build_debug_card(self, parent: ttk.Widget):
        """Build the debug console card."""
        card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        card.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        header = ttk.Frame(card, style="Card.TFrame")
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text="Debug", style="CardTitle.TLabel").pack(side=tk.LEFT)
        ttk.Button(header, text="Clear", command=self._clear_debug, style="Secondary.TButton").pack(side=tk.RIGHT)

        # Debug console with scrollbar
        dbg_container = ttk.Frame(card)
        dbg_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))
        self.debug_text = tk.Text(dbg_container, height=10, wrap="word")
        dbg_scroll = ttk.Scrollbar(dbg_container, orient="vertical", command=self.debug_text.yview)
        self.debug_text.configure(yscrollcommand=dbg_scroll.set)
        self.debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dbg_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # -------- UI helpers --------

    def _clear_debug(self):
        """Clear the debug console."""
        self.debug_text.delete("1.0", tk.END)

    def _write_debug(self, msg: str):
        """Write message to debug console."""
        self.debug_text.insert(tk.END, msg + "\n")
        self.debug_text.see(tk.END)

    def _validate_threshold(self, event=None):
        """Validate threshold entry input."""
        try:
            value_str = self.thr_entry.get()
            if not value_str:  # If empty, set to default
                value = 0.5
            else:
                value = float(value_str)
                
            value = max(0.0, min(1.0, value))  # Ensure between 0 and 1
            self.threshold.set(round(value, 2))
            self._on_slider_change()
        except ValueError:
            # If not numeric, restore previous value
            self.thr_entry.delete(0, tk.END)
            self.thr_entry.insert(0, f"{self.threshold.get():.2f}")

    def _validate_alpha(self, event=None):
        """Validate alpha entry input."""
        try:
            value_str = self.alpha_entry.get()
            if not value_str:  # If empty, set to default
                value = 0.5
            else:
                value = float(value_str)
                
            value = max(0.0, min(1.0, value))  # Ensure between 0 and 1
            self.alpha.set(round(value, 2))
            self._on_slider_change()
        except ValueError:
            # If not numeric, restore previous value
            self.alpha_entry.delete(0, tk.END)
            self.alpha_entry.insert(0, f"{self.alpha.get():.2f}")

    def _update_badges(self):
        """Update UI badges and entry fields."""
        # Update badges if they exist
        if hasattr(self, "badge_thr"):
            self.badge_thr.config(text=f"{self.threshold.get():.2f}")
        if hasattr(self, "badge_alpha"):
            self.badge_alpha.config(text=f"{self.alpha.get():.2f}")
        
        # Update entry fields if they exist
        if hasattr(self, "thr_entry") and self.thr_entry.winfo_exists():
            current_thr = self.thr_entry.get()
            new_thr = f"{self.threshold.get():.2f}"
            if current_thr != new_thr:
                self.thr_entry.delete(0, tk.END)
                self.thr_entry.insert(0, new_thr)
        
        if hasattr(self, "alpha_entry") and self.alpha_entry.winfo_exists():
            current_alpha = self.alpha_entry.get()
            new_alpha = f"{self.alpha.get():.2f}"
            if current_alpha != new_alpha:
                self.alpha_entry.delete(0, tk.END)
                self.alpha_entry.insert(0, new_alpha)

    def _set_metrics_preview(self, pil_img: Image.Image, title: str):
        """Set the metrics preview image."""
        # Keep a reasonable thumbnail size for the sidebar preview
        img = pil_img.copy()
        # If it's not grayscale, convert for a consistent look; masks are usually L
        if img.mode != "L":
            img = img.convert("L")
        img.thumbnail((360, 240), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.metrics_preview.config(image=photo)
        self.metrics_preview.image = photo  # prevent GC
        self.metrics_preview_title.config(text=title)

    def _on_slider_change(self, _evt=None):
        """Handle slider value changes."""
        self._update_badges()
        if self.pil_input is not None and self.model is not None and hasattr(self, 'label_mask'):
            try:
                self.on_segment(redraw_only=True)
            except Exception as e:
                self._write_debug(f"Error updating preview: {str(e)}")

    def _show_notification(self, title, message):
        """Unified notification method."""
        messagebox.showinfo(title, message)

    # -------- App actions --------

    def on_browse_weights(self):
        """Browse for model weights file."""
        path = filedialog.askopenfilename(
            title="Select best_model.pth",
            filetypes=[("PyTorch Weights", "*.pth"), ("All files", "*.*")]
        )
        if path:
            self.weights_path_var.set(path)

    def on_load_model(self):
        """Load the segmentation model."""
        path = self.weights_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid weights file (.pth)")
            self._update_model_status("❌ Invalid path", "Muted.TLabel")
            return

        self._update_model_status("⏳ Loading...", "Muted.TLabel")
        self.root.update()  # Force UI update

        try:
            cfg = load_config_from_dir(path)
            arch = cfg.get("arch") or cfg.get("arquitectura") or self.arch_var.get().strip() or "Unet"
            enc = cfg.get("encoder_name") or self.encoder_var.get().strip() or "resnet34"
            self.arch_var.set(arch)
            self.encoder_var.set(enc)

            self._write_debug(f"Loading model: arch={arch}, encoder={enc}")

            model = CamVidModel(arch=arch, encoder_name=enc, in_channels=3, out_classes=1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            mean_src, std_src = "imagenet", "imagenet"
            # Optional: dataset stats if present
            mean, std = None, None
            if mean is not None or std is not None:
                mean_src, std_src = model.set_stats(mean, std, device)

            state = torch.load(path, map_location=device)
            load_res = model.load_state_dict(state, strict=False)
            model.eval()

            self.model = model
            
            device_name = "GPU (CUDA)" if device.type == "cuda" else "CPU"
            model_info = f"Architecture: {arch} | Encoder: {enc} | Device: {device_name} | Stats: {mean_src}/{std_src}"
            if load_res.missing_keys or load_res.unexpected_keys:
                model_info += f" | Missing keys: {len(load_res.missing_keys)} | Unexpected keys: {len(load_res.unexpected_keys)}"
            
            self._update_model_status("✅ Loaded successfully", "Muted.TLabel", model_info)
            
            dbg = json.dumps({
                "arch": arch,
                "encoder_name": enc,
                "used_mean_source": mean_src,
                "used_std_source": std_src,
                "missing_keys_count": len(load_res.missing_keys),
                "unexpected_keys_count": len(load_res.unexpected_keys),
            }, indent=2)
            self._write_debug(dbg)
            self._show_notification("Model", "Model loaded successfully.")
            
        except Exception as e:
            self._update_model_status("❌ Load failed", "Muted.TLabel", f"Error: {str(e)}")
            self._write_debug(f"Model loading failed: {e}")
            messagebox.showerror("Model Error", f"Failed to load model:\n{e}")

    def on_open_image(self):
        """Open an image for segmentation."""
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
            self.pil_input = pil
            self.np_input_rgb = np.array(pil)
            self._show_preview(pil, self.label_input, max_size=(1000, 600))
            self.last_mask = None
            self.last_overlay = None
            self.label_mask.config(image="")
            self.label_overlay.config(image="")
            self.photo_mask = None
            self.photo_overlay = None
            self._write_debug(f"Loaded image: {os.path.basename(path)} size={pil.size}")
            # Switch to Input tab
            self.preview_nb.select(0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def _get_default_resize(self) -> Optional[int]:
        """Get the default resize value."""
        s = self.default_resize_var.get().strip()
        if not s:
            return None
        try:
            v = int(s)
            return v if v > 0 else None
        except Exception:
            return None

    def on_segment(self, redraw_only: bool = False):
        """Run segmentation inference."""
        if self.model is None:
            messagebox.showwarning("Model", "Please load model weights first.")
            return
        if self.pil_input is None:
            messagebox.showwarning("Image", "Please open an image first.")
            return

        thr = float(self.threshold.get())
        alpha = float(self.alpha.get())
        target_size = self._get_default_resize()

        try:
            img_rgb, img_t, original_size = preprocess_image_pil(self.pil_input, target_size=target_size)
            with torch.no_grad():
                logits = self.model(img_t.to(next(self.model.parameters()).device))
            mask = postprocess_mask(logits, thr, out_size=original_size)
            overlay = color_overlay(img_rgb, mask, alpha=alpha, color=(0, 255, 0))

            self.last_mask = mask
            self.last_overlay = overlay

            pil_mask = Image.fromarray(mask)
            self._show_preview(pil_mask, self.label_mask, max_size=(1000, 600), is_mask=True)

            pil_overlay = Image.fromarray(overlay)
            self._show_preview(pil_overlay, self.label_overlay, max_size=(1000, 600))

            self._write_debug(f"Inference done. thr={thr:.2f} alpha={alpha:.2f}, out={original_size}")
            if not redraw_only:
                # Switch to Overlay tab after a full run
                self.preview_nb.select(2)
                self._show_notification("Segmentación completada", "El proceso de segmentación ha finalizado correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed:\n{e}")

    def _show_preview(self, pil_img: Image.Image, widget: ttk.Label, max_size=(1000, 600), is_mask=False):
        """Show image preview in widget."""
        img = pil_img.copy()
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        if is_mask and img.mode != "L":
            img = img.convert("L")
        elif not is_mask and img.mode != "RGB":
            img = img.convert("RGB")
        photo = ImageTk.PhotoImage(img)
        widget.config(image=photo)
        # Keep a reference on the widget to avoid GC
        widget.image = photo

    def on_save_mask(self):
        """Save the predicted mask."""
        if self.last_mask is None:
            messagebox.showwarning("Save Mask", "No mask to save. Run segmentation first.")
            return
        path = filedialog.asksaveasfilename(title="Save Mask PNG", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            cv2.imwrite(path, self.last_mask)
            self._show_notification("Save Mask", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Mask", f"Failed to save mask:\n{e}")

    def on_save_overlay(self):
        """Save the overlay image."""
        if self.last_overlay is None:
            messagebox.showwarning("Save Overlay", "No overlay to save. Run segmentation first.")
            return
        path = filedialog.asksaveasfilename(title="Save Overlay PNG", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            bgr = cv2.cvtColor(self.last_overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            self._show_notification("Save Overlay", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Overlay", f"Failed to save overlay:\n{e}")

    def on_metrics_from_pred(self):
        """Compute metrics from predicted mask."""
        if self.last_mask is None:
            messagebox.showwarning("Metrics", "No predicted mask. Run segmentation first.")
            return
        try:
            # Preview: predicted mask
            pil_mask = Image.fromarray(self.last_mask if self.last_mask.ndim == 2 else cv2.cvtColor(self.last_mask, cv2.COLOR_BGR2GRAY))
            self._set_metrics_preview(pil_mask, "Predicted mask")

            # Compute metrics
            mask_bin = (self.last_mask >= 128).astype(np.uint8)
            # Generate petal mask from input image if available, and add vein mask
            petal_mask = None
            img_rgb = self.np_input_rgb if self.np_input_rgb is not None else None
            res = compute_normalized_metrics(mask_bin, petal_mask=petal_mask, img_rgb=img_rgb)
            self.current_metrics = res
            self._show_metrics(res)
        except Exception as e:
            messagebox.showerror("Metrics", f"Failed to compute metrics:\n{e}")

    def on_metrics_from_file(self):
        """Compute metrics from mask file."""
        path = filedialog.askopenfilename(
            title="Open Mask Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("Unable to read mask file")

            # Preview: file mask
            pil = Image.fromarray(img)
            self._set_metrics_preview(pil, f"File: {os.path.basename(path)}")

            # Ask user if they want to provide a petal mask
            use_petal_mask = messagebox.askyesno("Petal Mask", "Provide the original image to calculate petal masks")
            petal_mask = None
            if use_petal_mask:
                petal_path = filedialog.askopenfilename(
                    title="Open Petal Mask Image",
                    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("All files", "*.*")]
                )
                if petal_path:
                    petal_img = cv2.imread(petal_path, cv2.IMREAD_GRAYSCALE)
                    if petal_img is not None:
                        petal_mask = (petal_img >= 128).astype(np.uint8)

            # Compute metrics
            mask_bin = (img >= 128).astype(np.uint8)
            res = compute_normalized_metrics(mask_bin, petal_mask=petal_mask)
            self.current_metrics = res
            self._show_metrics(res)
        except Exception as e:
            messagebox.showerror("Metrics", f"Failed to compute metrics:\n{e}")

    def on_save_metrics(self):
        """Save metrics to CSV file."""
        if not self.current_metrics:
            messagebox.showwarning("Save Metrics", "No metrics to save. Compute metrics first.")
            return
        
        path = filedialog.asksaveasfilename(
            title="Save Metrics to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        
        try:
            # Add timestamp and image info to metrics
            metrics_with_info = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_name": self.metrics_preview_title.cget("text"),
                **self.current_metrics
            }
            
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(path)
            
            with open(path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = list(metrics_with_info.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(metrics_with_info)
            
            self._show_notification("Save Metrics", f"Metrics saved to: {path}")
            self._write_debug(f"Metrics saved to CSV: {path}")
            
        except Exception as e:
            messagebox.showerror("Save Metrics", f"Failed to save metrics:\n{e}")
            self._write_debug(f"Error saving metrics: {e}")

    def _show_metrics(self, res: Dict[str, Any]):
        """Display metrics in the tree view."""
        for child in self.metrics_tree.get_children():
            self.metrics_tree.delete(child)
        for k, v in res.items():
            disp = f"{v:.2f}" if isinstance(v, float) else str(v)
            self.metrics_tree.insert("", tk.END, values=(k, disp))

    def on_batch_process(self):
        """Process all images in a selected folder."""
        if self.model is None:
            messagebox.showwarning("Model", "Please load model weights first.")
            return
        
        folder_path = filedialog.askdirectory(title="Select folder with images to process")
        if not folder_path:
            return
        
        # Find all image files
        folder = Path(folder_path)
        image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]
        
        if not image_files:
            messagebox.showwarning("No Images", "No supported image files found in the selected folder.")
            return
        
        # Create results folder
        results_folder = folder / "results"
        results_folder.mkdir(exist_ok=True)
        
        # Check for existing CSV to see what's already been processed
        csv_path = results_folder / "metrics_summary.csv"
        existing_processed = set()
        if csv_path.exists():
            try:
                with open(csv_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_processed = {row['image_name'] for row in reader}
                self._write_debug(f"Found existing metrics CSV with {len(existing_processed)} processed images")
            except Exception as e:
                self._write_debug(f"Could not read existing CSV: {e}")
        
        # Filter out already processed images
        images_to_process = []
        skipped_count = 0
        for image_path in image_files:
            mask_filename = f"{image_path.stem}_mask.png"
            mask_path = results_folder / mask_filename
            
            # Skip if both mask exists AND image is in CSV
            if mask_path.exists() and image_path.name in existing_processed:
                skipped_count += 1
                continue
            images_to_process.append(image_path)
        
        if skipped_count > 0:
            self._write_debug(f"Skipping {skipped_count} already processed images")
        
        if not images_to_process:
            self._show_notification("Batch Complete", "All images in this folder have already been processed!")
            return
        
        # Show progress
        self.progress_frame.pack()
        self.progress_bar['maximum'] = len(images_to_process)
        self.progress_bar['value'] = 0
        
        # Prepare CSV for metrics - load existing data if CSV exists
        metrics_data = []
        if csv_path.exists():
            try:
                with open(csv_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    metrics_data = list(reader)
                self._write_debug(f"Loaded {len(metrics_data)} existing metrics records")
            except Exception as e:
                self._write_debug(f"Could not load existing metrics: {e}")
                metrics_data = []
        
        thr = float(self.threshold.get())
        target_size = self._get_default_resize()
        
        self._write_debug(f"Starting batch processing of {len(images_to_process)} new images...")
        
        try:
            for i, image_path in enumerate(images_to_process):
                self.progress_label.config(text=f"Processing {image_path.name}...")
                self.root.update()  # Update UI
                
                try:
                    # Load and process image
                    pil_img = Image.open(image_path).convert("RGB")
                    img_rgb, img_t, original_size = preprocess_image_pil(pil_img, target_size=target_size)
                    
                    # Run inference
                    with torch.no_grad():
                        logits = self.model(img_t.to(next(self.model.parameters()).device))
                    mask = postprocess_mask(logits, thr, out_size=original_size)
                    
                    # Save mask
                    mask_filename = f"{image_path.stem}_mask.png"
                    mask_path = results_folder / mask_filename
                    cv2.imwrite(str(mask_path), mask)
                    
                    # Compute metrics
                    mask_bin = (mask >= 128).astype(np.uint8)
                    metrics = compute_normalized_metrics(mask_bin)
                    
                    # Add to CSV data
                    row_data = {
                        'image_name': image_path.name,
                        'mask_name': mask_filename,
                        'threshold': thr,
                        'image_size': f"{original_size[0]}x{original_size[1]}",
                        **metrics
                    }
                    metrics_data.append(row_data)
                    
                    self._write_debug(f"✓ Processed {image_path.name}")
                    
                except Exception as e:
                    self._write_debug(f"✗ Failed to process {image_path.name}: {e}")
                    continue
                
                # Update progress
                self.progress_bar['value'] = i + 1
                self.root.update()
        
            # Save metrics CSV
            if metrics_data:
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = metrics_data[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(metrics_data)
                
                self._write_debug(f"✓ Saved metrics summary to {csv_path}")
            
            # Hide progress and show completion
            self.progress_frame.pack_forget()

            self._show_notification("Batch Complete", 
                              f"Processed {len(images_to_process)} new images successfully.\n"
                              f"Total images in results: {len(metrics_data)}\n"
                              f"Results saved to: {results_folder}\n"
                              f"- Mask images: *_mask.png\n"
                              f"- Metrics summary: metrics_summary.csv")
            
            self._write_debug(f"Batch processing complete. {len(images_to_process)} new images processed. Total: {len(metrics_data)} images in results.")
            
        except Exception as e:
            self.progress_frame.pack_forget()
            messagebox.showerror("Batch Error", f"Batch processing failed: {e}")
            self._write_debug(f"Batch processing error: {e}")

    def _update_model_status(self, status: str, color_style: str = "Muted.TLabel", info: str = ""):
        """Update the model status indicator."""
        self.model_status_label.config(text=status, style=color_style)
        
        if info:
            self.model_info_label.config(text=info)
            self.model_info_frame.grid()
        else:
            self.model_info_frame.grid_remove()