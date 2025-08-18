"""
Tkinter desktop app for binary segmentation inference and mask-based metrics
Styled to resemble the web UI (cards, header, tabs, badges, accent buttons).

Features:
- Load weights (.pth) and auto-read config.json, dataset_mean.npy next to it
- Select image (PNG/JPG/TIFF), run segmentation, tune threshold & alpha
- Tabbed Preview: Input / Mask / Overlay
- Save mask/overlay to PNG
- Compute personalized metrics from predicted mask or any mask file
- Scroll to access all controls on smaller screens
- Scrollable Debug console with Clear
- Batch processing of images in a folder

Run (Windows):
  cd scripts
  .venv\Scripts\activate
  pip install -r requirements.txt
  python tk_inference_app.py
"""

import os
import json
from typing import Optional, Tuple, Any, Dict
import csv
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

import torch
import segmentation_models_pytorch as smp

# Local metrics helper
from metrics import compute_metrics_from_mask

# ==========================================
# 🔧 MODEL CONFIGURATION - EDIT THIS SECTION
# ==========================================

MODEL_CONFIG = {
    # Model file names (should be in the same directory as this script)
    "weights_file": "best_model.pth",           # Your trained model weights
    "config_file": "config.json",              # Optional: model config
    "mean_file": "dataset_mean.npy",           # Optional: dataset mean
    "std_file": "dataset_std.npy",             # Optional: dataset std
    
    # Default model architecture (used if config.json not found)
    "default_arch": "Unet",                    # Unet, FPN, DeepLabV3Plus, etc.
    "default_encoder": "resnet34",             # resnet34, efficientnet-b0, etc.
    
    # Inference settings
    "default_resize": 640,                     # Resize images to this size (None to disable)
    "force_imagenet_stats": False,             # True to ignore dataset_mean/std.npy
    "device": "auto",                          # "auto", "cpu", or "cuda"
    "auto_load": True,                         # Auto-load model on startup
}

# ==========================================

# ---------------------------
# Model helpers (same logic as server)
# ---------------------------

class CamVidModel(torch.nn.Module):
    def __init__(self, arch: str, encoder_name: str, in_channels: int = 3, out_classes: int = 1, **kwargs):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

    def set_stats(self, mean: Optional[np.ndarray], std: Optional[np.ndarray], device: torch.device) -> Tuple[str, str]:
        mean_src, std_src = "imagenet", "imagenet"
        if mean is not None:
            m = np.array(mean, dtype=np.float32)
            if m.max() > 1.5:
                m = m / 255.0
            self.mean = torch.tensor(m, dtype=torch.float32, device=device).view(1, 3, 1, 1)
            mean_src = "dataset_mean.npy"
        if std is not None:
            s = np.array(std, dtype=np.float32)
            if s.max() > 1.5:
                s = s / 255.0
            self.std = torch.tensor(s, dtype=torch.float32, device=device).view(1, 3, 1, 1)
            std_src = "dataset_std.npy"
        return mean_src, std_src

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        return self.model(image)


def load_config_from_dir(weights_path: str) -> Dict[str, Any]:
    cfg = {}
    base = os.path.dirname(os.path.abspath(weights_path))
    cfg_path = os.path.join(base, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            pass
    return cfg


def find_stats_in_dir(weights_path: str):
    base = os.path.dirname(os.path.abspath(weights_path))
    mean_path = os.path.join(base, "dataset_mean.npy")
    std_path = os.path.join(base, "dataset_std.npy")
    mean = np.load(mean_path) if os.path.exists(mean_path) else None
    std = np.load(std_path) if os.path.exists(std_path) else None
    return mean, std


def preprocess_image_pil(pil_img: Image.Image, target_size: Optional[int] = None):
    img_rgb = np.array(pil_img)  # RGB
    h, w = img_rgb.shape[:2]
    original_size = (w, h)
    if target_size and target_size > 0:
        img_rs = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    else:
        img_rs = img_rgb
    img_t = torch.from_numpy(img_rs.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return img_rgb, img_t, original_size


def postprocess_mask(logits: torch.Tensor, threshold: float, out_size: Tuple[int, int]) -> np.ndarray:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    mask = pred[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
    if out_size:
        mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
    return mask


def color_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color=(0, 255, 0)) -> np.ndarray:
    out = rgb.copy()
    m = mask > 0
    if m.any():
        overlay = np.zeros_like(out)
        overlay[m] = color
        out[m] = (out[m].astype(np.float32) * (1 - alpha) + overlay[m].astype(np.float32) * alpha).astype(np.uint8)
    return out


def load_integrated_model(script_dir: str, config: Dict[str, Any]) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load the integrated model from files in the script directory.
    """
    # Determine device
    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    
    # Check for weights file
    weights_path = os.path.join(script_dir, config["weights_file"])
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    
    # Load config if available
    model_config = {}
    config_path = os.path.join(script_dir, config["config_file"])
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                model_config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
    
    # Determine architecture
    arch = model_config.get("arch") or model_config.get("arquitectura") or config["default_arch"]
    encoder_name = model_config.get("encoder_name") or config["default_encoder"]
    
    # Create model
    model = CamVidModel(arch=arch, encoder_name=encoder_name, in_channels=3, out_classes=1)
    model.to(device)
    
    # Load normalization stats
    mean_src, std_src = "imagenet", "imagenet"
    if not config["force_imagenet_stats"]:
        mean_path = os.path.join(script_dir, config["mean_file"])
        std_path = os.path.join(script_dir, config["std_file"])
        
        mean = np.load(mean_path) if os.path.exists(mean_path) else None
        std = np.load(std_path) if os.path.exists(std_path) else None
        
        if mean is not None or std is not None:
            mean_src, std_src = model.set_stats(mean, std, device)
    
    # Load weights
    state = torch.load(weights_path, map_location=device)
    load_res = model.load_state_dict(state, strict=False)
    model.eval()
    
    # Create debug metadata
    debug_meta = {
        "arch": arch,
        "encoder_name": encoder_name,
        "weights_path": os.path.abspath(weights_path),
        "device": str(device),
        "used_mean_source": mean_src,
        "used_std_source": std_src,
        "load_missing_keys_count": len(load_res.missing_keys),
        "load_unexpected_keys_count": len(load_res.unexpected_keys),
        "load_missing_keys_sample": load_res.missing_keys[:5],
        "load_unexpected_keys_sample": load_res.unexpected_keys[:5],
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": torch.__version__,
        "smp_version": getattr(smp, "__version__", "unknown"),
        "opencv_version": cv2.__version__,
        "integrated_model": True,
        "auto_loaded": True,
    }
    
    return model, debug_meta


# ---------------------------
# Scrollable Frame (outer page scroll)
# ---------------------------

class ScrollableFrame(ttk.Frame):
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


# ---------------------------
# Theme
# ---------------------------

def create_theme(style: ttk.Style):
    # Accent and neutrals inspired by the web UI
    ACCENT = "#22c55e"        # green-500
    ACCENT_DARK = "#16a34a"   # green-600
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


# ---------------------------
# Tkinter App
# ---------------------------

class SegTkApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Segmentation Mask Inference (Tk) - v2.0")
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
        ttk.Label(header, text="v2.0", style="Header.TLabel").pack(side=tk.RIGHT, padx=16)

        # Content grid (two columns like the web app)
        content = ttk.Frame(self.scroll.container)
        content.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=14, pady=14)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        # Left column: Input & Controls (Card)
        self._build_input_controls(content)

        # Right column: Preview (Card with tabs)
        self._build_preview_card(content)

        # Bottom row: Metrics and Debug (two cards)
        self._build_metrics_card(self.scroll.container)
        self._build_debug_card(self.scroll.container)

        # Model/meta state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[torch.nn.Module] = None
        self.debug_meta: Dict[str, Any] = {}
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Inference state
        self.pil_input: Optional[Image.Image] = None
        self.np_input_rgb: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_overlay: Optional[np.ndarray] = None

        # UI state

        # Set initial labels for badges
        self._update_badges()

        self._write_debug("Ready. Load model, open an image, then Segment.")
        
        # Auto-load model if enabled
        if MODEL_CONFIG.get("auto_load", False):
            self.root.after(100, self._auto_load_model)

    # -------- UI building blocks --------

    def _build_input_controls(self, parent: ttk.Frame):
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

        # Arch/Encoder row
        ttk.Label(card, text="Arch:", style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(6, 2))
        ttk.Entry(card, textvariable=self.arch_var, width=18).grid(row=3, column=1, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(card, text="Encoder:", style="Muted.TLabel").grid(row=3, column=2, sticky="e", pady=(6, 2))
        ttk.Entry(card, textvariable=self.encoder_var, width=18).grid(row=3, column=3, sticky="w", padx=6, pady=(6, 2))
        ttk.Label(card, text="Default Resize (px):", style="Muted.TLabel").grid(row=3, column=4, sticky="e", pady=(6, 2))
        ttk.Entry(card, textvariable=self.default_resize_var, width=10).grid(row=3, column=5, sticky="w", padx=4, pady=(6, 2))

        # Actions row
        row = 4
        actions = ttk.Frame(card, style="Card.TFrame")
        actions.grid(row=row, column=0, columnspan=6, sticky="we", pady=(8, 2))
        ttk.Button(actions, text="Open Image", command=self.on_open_image, style="Secondary.TButton").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="Segment", command=self.on_segment, style="Accent.TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Mask", command=self.on_save_mask, style="Secondary.TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Overlay", command=self.on_save_overlay, style="Secondary.TButton").pack(side=tk.LEFT, padx=6)

        # Sliders section
        row += 1
        ttk.Label(card, text="Controls", style="Section.TLabel").grid(row=row, column=0, columnspan=6, sticky="w", pady=(10, 4))
        row += 1

        # Threshold slider + badge
        thr_frame = ttk.Frame(card)
        thr_frame.grid(row=row, column=0, columnspan=6, sticky="we", pady=4)
        thr_frame.columnconfigure(1, weight=1)
        ttk.Label(thr_frame, text="Threshold").grid(row=0, column=0, sticky="w")
        self.badge_thr = ttk.Label(thr_frame, text="0.50", style="Badge.TLabel")
        self.badge_thr.grid(row=0, column=2, sticky="e")
        ttk.Scale(thr_frame, from_=0.0, to=1.0, variable=self.threshold,
                  orient=tk.HORIZONTAL, command=self._on_slider_change).grid(row=0, column=1, sticky="we", padx=10)

        # Alpha slider + badge
        row += 1
        alpha_frame = ttk.Frame(card)
        alpha_frame.grid(row=row, column=0, columnspan=6, sticky="we", pady=4)
        alpha_frame.columnconfigure(1, weight=1)
        ttk.Label(alpha_frame, text="Overlay alpha").grid(row=0, column=0, sticky="w")
        self.badge_alpha = ttk.Label(alpha_frame, text="0.50", style="Badge.TLabel")
        self.badge_alpha.grid(row=0, column=2, sticky="e")
        ttk.Scale(alpha_frame, from_=0.0, to=1.0, variable=self.alpha,
                  orient=tk.HORIZONTAL, command=self._on_slider_change).grid(row=0, column=1, sticky="we", padx=10)

        # Batch processing section
        row += 1
        ttk.Label(card, text="Batch Processing", style="Section.TLabel").grid(row=row, column=0, columnspan=6, sticky="w", pady=(10, 4))
        row += 1

        batch_frame = ttk.Frame(card, style="Card.TFrame")
        batch_frame.grid(row=row, column=0, columnspan=6, sticky="we", pady=(4, 8))
        ttk.Button(batch_frame, text="Select Folder & Process All Images", command=self.on_batch_process, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(batch_frame, text="Processes all images in a folder and saves masks + metrics to 'results' subfolder", style="Muted.TLabel").pack(side=tk.LEFT)

        # Progress bar (initially hidden)
        self.progress_frame = ttk.Frame(card)
        self.progress_frame.grid(row=row+1, column=0, columnspan=6, sticky="we", pady=4)
        self.progress_label = ttk.Label(self.progress_frame, text="", style="Muted.TLabel")
        self.progress_label.pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        self.progress_frame.grid_remove()  # Hide initially

        for c in range(6):
            card.columnconfigure(c, weight=1)

    def _build_preview_card(self, parent: ttk.Frame):
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

    def _build_metrics_card(self, parent: ttk.Widget):
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

    def _build_debug_card(self, parent: ttk.Widget):
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
        self.debug_text.delete("1.0", tk.END)

    def _write_debug(self, msg: str):
        self.debug_text.insert(tk.END, msg + "\n")
        self.debug_text.see(tk.END)

    def _update_badges(self):
        if hasattr(self, "badge_thr"):
            self.badge_thr.config(text=f"{self.threshold.get():.2f}")
        if hasattr(self, "badge_alpha"):
            self.badge_alpha.config(text=f"{self.alpha.get():.2f}")

    def _set_metrics_preview(self, pil_img: Image.Image, title: str):
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
        self._update_badges()
        if self.pil_input is not None and self.model is not None:
            try:
                self.on_segment(redraw_only=True)
            except Exception:
                pass

    # -------- App actions --------

    def on_browse_weights(self):
        path = filedialog.askopenfilename(
            title="Select best_model.pth",
            filetypes=[("PyTorch Weights", "*.pth"), ("All files", "*.*")]
        )
        if path:
            self.weights_path_var.set(path)

    def on_load_model(self):
        path = self.weights_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid weights file (.pth)")
            return

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
        mean, std = find_stats_in_dir(path)
        if mean is not None or std is not None:
            mean_src, std_src = model.set_stats(mean, std, device)

        state = torch.load(path, map_location=device)
        load_res = model.load_state_dict(state, strict=False)
        model.eval()

        self.model = model
        dbg = json.dumps({
            "arch": arch,
            "encoder_name": enc,
            "used_mean_source": mean_src,
            "used_std_source": std_src,
            "missing_keys_count": len(load_res.missing_keys),
            "unexpected_keys_count": len(load_res.unexpected_keys),
        }, indent=2)
        self._write_debug(dbg)
        messagebox.showinfo("Model", "Model loaded successfully.")

    def on_open_image(self):
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
        s = self.default_resize_var.get().strip()
        if not s:
            # Use config default if field is empty
            return MODEL_CONFIG.get("default_resize")
        try:
            v = int(s)
            return v if v > 0 else None
        except Exception:
            return MODEL_CONFIG.get("default_resize")

    def on_segment(self, redraw_only: bool = False):
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
                # Switch to Overlay tab after a full run (like the web app)
                self.preview_nb.select(2)
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed:\n{e}")

    def _show_preview(self, pil_img: Image.Image, widget: ttk.Label, max_size=(1000, 600), is_mask=False):
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
        if self.last_mask is None:
            messagebox.showwarning("Save Mask", "No mask to save. Run segmentation first.")
            return
        path = filedialog.asksaveasfilename(title="Save Mask PNG", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            cv2.imwrite(path, self.last_mask)
            messagebox.showinfo("Save Mask", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Mask", f"Failed to save mask:\n{e}")

    def on_save_overlay(self):
        if self.last_overlay is None:
            messagebox.showwarning("Save Overlay", "No overlay to save. Run segmentation first.")
            return
        path = filedialog.asksaveasfilename(title="Save Overlay PNG", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            bgr = cv2.cvtColor(self.last_overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
            messagebox.showinfo("Save Overlay", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Overlay", f"Failed to save overlay:\n{e}")

    def on_metrics_from_pred(self):
        if self.last_mask is None:
            messagebox.showwarning("Metrics", "No predicted mask. Run segmentation first.")
            return
        try:
            # Preview: predicted mask
            pil_mask = Image.fromarray(self.last_mask if self.last_mask.ndim == 2 else cv2.cvtColor(self.last_mask, cv2.COLOR_BGR2GRAY))
            self._set_metrics_preview(pil_mask, "Predicted mask")

            # Compute metrics
            mask_bin = (self.last_mask >= 128).astype(np.uint8)
            res = compute_metrics_from_mask(mask_bin)
            self._show_metrics(res)
        except Exception as e:
            messagebox.showerror("Metrics", f"Failed to compute metrics:\n{e}")

    def on_metrics_from_file(self):
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

            # Compute metrics
            mask_bin = (img >= 128).astype(np.uint8)
            res = compute_metrics_from_mask(mask_bin)
            self._show_metrics(res)
        except Exception as e:
            messagebox.showerror("Metrics", f"Failed to compute metrics:\n{e}")

    def _show_metrics(self, res: Dict[str, Any]):
        for child in self.metrics_tree.get_children():
            self.metrics_tree.delete(child)
        for k, v in res.items():
            disp = f"{v:.2f}" if isinstance(v, float) else str(v)
            self.metrics_tree.insert("", tk.END, values=(k, disp))

    def on_batch_process(self):
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
            messagebox.showinfo("Batch Complete", "All images in this folder have already been processed!")
            return
        
        # Show progress
        self.progress_frame.grid()
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
                    metrics = compute_metrics_from_mask(mask_bin)
                    
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
            self.progress_frame.grid_remove()
            
            messagebox.showinfo("Batch Complete", 
                              f"Processed {len(images_to_process)} new images successfully.\n"
                              f"Total images in results: {len(metrics_data)}\n"
                              f"Results saved to: {results_folder}\n"
                              f"- Mask images: *_mask.png\n"
                              f"- Metrics summary: metrics_summary.csv")
            
            self._write_debug(f"Batch processing complete. {len(images_to_process)} new images processed. Total: {len(metrics_data)} images in results.")
            
        except Exception as e:
            self.progress_frame.grid_remove()
            messagebox.showerror("Batch Error", f"Batch processing failed: {e}")
            self._write_debug(f"Batch processing error: {e}")

    def _auto_load_model(self):
        """Auto-load the integrated model on startup"""
        try:
            self._write_debug("🚀 Auto-loading integrated model...")
            self._write_debug(f"📂 Script directory: {self.script_dir}")
            
            # Check if required files exist
            weights_path = os.path.join(self.script_dir, MODEL_CONFIG["weights_file"])
            if not os.path.exists(weights_path):
                self._write_debug(f"❌ Model weights not found: {MODEL_CONFIG['weights_file']}")
                self._write_debug("📋 To use auto-load, place these files in the scripts/ directory:")
                self._write_debug(f"   - {MODEL_CONFIG['weights_file']} (required)")
                self._write_debug(f"   - {MODEL_CONFIG['config_file']} (optional)")
                self._write_debug(f"   - {MODEL_CONFIG['mean_file']} (optional)")
                self._write_debug(f"   - {MODEL_CONFIG['std_file']} (optional)")
                self._write_debug("💡 Or disable auto_load in MODEL_CONFIG and use 'Load Model' button")
                return
            
            self._write_debug(f"✅ Found weights: {MODEL_CONFIG['weights_file']}")
            
            # Load the model
            self.model, self.debug_meta = load_integrated_model(self.script_dir, MODEL_CONFIG)
            
            # Update the UI fields with loaded values
            self.arch_var.set(self.debug_meta.get("arch", "Unet"))
            self.encoder_var.set(self.debug_meta.get("encoder_name", "resnet34"))
            self.weights_path_var.set(self.debug_meta.get("weights_path", ""))
            
            # Set default resize from config
            if MODEL_CONFIG.get("default_resize"):
                self.default_resize_var.set(str(MODEL_CONFIG["default_resize"]))
            
            # Log success
            arch = self.debug_meta.get("arch", "Unknown")
            encoder = self.debug_meta.get("encoder_name", "Unknown")
            device = self.debug_meta.get("device", "Unknown")
            
            self._write_debug("🎉 Integrated model loaded successfully!")
            self._write_debug(f"Architecture: {arch}")
            self._write_debug(f"Encoder: {encoder}")
            self._write_debug(f"Device: {device}")
            self._write_debug(f"Normalization: {self.debug_meta.get('used_mean_source')} mean, {self.debug_meta.get('used_std_source')} std")
            self._write_debug(f"Parameters: {self.debug_meta.get('num_parameters', 0):,}")
            
            if self.debug_meta.get('load_missing_keys_count', 0) > 0:
                self._write_debug(f"⚠️  Missing keys: {self.debug_meta.get('load_missing_keys_count')}")
            if self.debug_meta.get('load_unexpected_keys_count', 0) > 0:
                self._write_debug(f"⚠️  Unexpected keys: {self.debug_meta.get('load_unexpected_keys_count')}")
            
            self._write_debug("🖼️  Model ready! Open an image and click 'Segment' to start.")
            
        except Exception as e:
            self._write_debug(f"❌ Failed to auto-load model: {e}")
            self._write_debug("💡 You can still use 'Load Model' button to load manually")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = SegTkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
