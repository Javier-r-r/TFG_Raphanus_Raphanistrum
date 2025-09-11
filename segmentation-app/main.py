"""
Main application class for the segmentation inference GUI.
"""
import os
import csv
import unicodedata
import re
import queue
import numpy as np
import torch
from background_worker import run_callable_in_thread
import cv2
import tkinter as tk

from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from models import CamVidModel, load_config_from_dir
from models import preprocess_image_pil, postprocess_mask, color_overlay
from ui_components import ScrollableFrame
from theme import create_theme
from metrics import compute_normalized_metrics


def normalize_filename(filename: str) -> str:
    """Normaliza nombres de archivo/ruta para evitar problemas con caracteres especiales."""
    # Normaliza a NFKD y elimina diacríticos
    nfkd = unicodedata.normalize('NFKD', filename)
    ascii_str = nfkd.encode('ASCII', 'ignore').decode('ASCII')
    # Reemplaza caracteres no válidos por guion bajo
    ascii_str = re.sub(r'[^\w\-.]', '_', ascii_str)
    return ascii_str


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

        # Queue para comunicación entre worker y GUI
        self.task_queue = queue.Queue()
        # Estado para últimos resultados
        self.pil_input = None
        self.last_mask = None
        self.last_overlay = None
        # Start polling the queue
        self.root.after(150, self._poll_task_queue)

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
        ttk.Label(card, text="Select model weights and an image, set architecture, encoder, loss, threshold and alpha, then run segmentation.",
                  style="CardDesc.TLabel").grid(row=1, column=0, columnspan=6, sticky="w", pady=(2, 10))

        # Weights row
        ttk.Label(card, text="Weights (.pth):", style="Section.TLabel").grid(row=2, column=0, sticky="w", pady=2)
        self.e_weights = ttk.Entry(card, textvariable=self.weights_path_var, width=56)
        self.e_weights.grid(row=2, column=1, columnspan=3, sticky="we", padx=6)
        ttk.Button(card, text="Browse", command=self.on_browse_weights, style="Secondary.TButton").grid(row=2, column=4, padx=4)
        ttk.Button(card, text="Load Model", command=self.on_load_model, style="Accent.TButton").grid(row=2, column=5, padx=2)

        # Architecture, Encoder, Loss row
        ttk.Label(card, text="Architecture:", style="Section.TLabel").grid(row=3, column=0, sticky="w", pady=2)
        self.arch_var = tk.StringVar(value="Unet")
        arch_options = ["Unet", "FPN", "PSPNet", "DeepLabV3"]
        self.arch_combo = ttk.Combobox(card, textvariable=self.arch_var, values=arch_options, state="readonly", width=12)
        self.arch_combo.grid(row=3, column=1, sticky="w", padx=2)

        ttk.Label(card, text="Encoder:", style="Section.TLabel").grid(row=3, column=2, sticky="e", pady=2)
        self.encoder_var = tk.StringVar(value="resnet34")
        encoder_options = ["resnet34", "resnet50", "efficientnet-b0", "mobilenet_v2"]
        self.encoder_combo = ttk.Combobox(card, textvariable=self.encoder_var, values=encoder_options, state="readonly", width=16)
        self.encoder_combo.grid(row=3, column=3, sticky="w", padx=2)

        ttk.Label(card, text="Loss:", style="Section.TLabel").grid(row=3, column=4, sticky="e", pady=2)
        self.loss_var = tk.StringVar(value="dice")
        loss_options = ["dice", "bce", "focal", "bce_dice"]
        self.loss_combo = ttk.Combobox(card, textvariable=self.loss_var, values=loss_options, state="readonly", width=10)
        self.loss_combo.grid(row=3, column=5, sticky="w", padx=2)

        # Model status indicator row
        status_frame = ttk.Frame(card)
        status_frame.grid(row=4, column=0, columnspan=6, sticky="we", pady=(4, 2))
        status_frame.columnconfigure(1, weight=1)
        ttk.Label(status_frame, text="Model Status:", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.model_status_label = ttk.Label(status_frame, text="Not loaded", style="Muted.TLabel")
        self.model_status_label.grid(row=0, column=1, sticky="w", padx=(8, 0))

        # Model info display (initially hidden)
        self.model_info_frame = ttk.Frame(card)
        self.model_info_frame.grid(row=5, column=0, columnspan=6, sticky="we", pady=(2, 6))
        self.model_info_label = ttk.Label(self.model_info_frame, text="", style="Muted.TLabel", wraplength=600)
        self.model_info_label.pack(anchor="w")
        self.model_info_frame.grid_remove()  # Hide initially

        ttk.Label(card, text="Default Resize (px):", style="Muted.TLabel").grid(row=6, column=4, sticky="e", pady=(6, 2))
        resize_entry = ttk.Entry(card, textvariable=self.default_resize_var, width=10, state="readonly")
        resize_entry.grid(row=6, column=5, sticky="w", padx=4, pady=(6, 2))
        self.resize_entry = resize_entry

        # Actions row
        row = 7
        actions = ttk.Frame(card, style="Card.TFrame")
        actions.grid(row=row, column=0, columnspan=6, sticky="we", pady=(8, 2))
        ttk.Button(actions, text="Open Image", command=self.on_open_image, style="Secondary.TButton").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(actions, text="Segment", command=self.on_segment, style="Accent.TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Mask", command=self.on_save_mask, style="Secondary.TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Overlay", command=self.on_save_overlay, style="Secondary.TButton").pack(side=tk.LEFT, padx=6)

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

    def _show_busy_dialog(self, title="Segmenting", message="Segmenting image, please wait..."):
        """Muestra un diálogo modal mientras se segmenta."""
        self._hide_busy_dialog()  # Por si acaso
        self._busy_dialog = tk.Toplevel(self.root)
        self._busy_dialog.title(title)
        self._busy_dialog.geometry("320x100")
        self._busy_dialog.transient(self.root)
        self._busy_dialog.grab_set()
        ttk.Label(self._busy_dialog, text=message, anchor="center").pack(expand=True, fill="both", padx=20, pady=20)
        self._busy_dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # Evita cerrar

    def _hide_busy_dialog(self):
        if hasattr(self, "_busy_dialog") and self._busy_dialog.winfo_exists():
            self._busy_dialog.grab_release()
            self._busy_dialog.destroy()
            del self._busy_dialog

    def _poll_task_queue(self):
        """Leer mensajes desde la cola de worker y actualizar UI."""
        try:
            while True:
                msg_type, payload = self.task_queue.get_nowait()
                if msg_type == "log":
                    self._write_debug(payload)
                elif msg_type == "progress":
                    # payload expected as float 0..1
                    try:
                        pct = float(payload)
                        # if you have a progress widget update here
                        # self.progress_var.set(pct*100)
                        self._write_debug(f"Progress: {pct*100:.1f}%")
                    except Exception:
                        pass
                elif msg_type == "done":
                    self._write_debug("Task finished")
                    self._on_task_done(payload)
                    self._enable_controls()
                elif msg_type == "error":
                    self._write_debug(f"Error: {payload}")
                    messagebox.showerror("Error", str(payload))
                    self._enable_controls()
        except queue.Empty:
            pass
        # volver a programar polling
        self.root.after(150, self._poll_task_queue)

    def _disable_controls(self):
        """Deshabilitar controles principales para evitar ejecuciones simultáneas."""
        try:
            # intenta deshabilitar widgets comunes si existen
            for name in ("btn_segment", "btn_batch", "btn_load_model", "e_weights"):
                w = getattr(self, name, None)
                if w is not None:
                    try:
                        w.config(state="disabled")
                    except Exception:
                        pass
        except Exception:
            pass

    def _enable_controls(self):
        """Re-habilitar controles tras completar tarea."""
        try:
            for name in ("btn_segment", "btn_batch", "btn_load_model", "e_weights"):
                w = getattr(self, name, None)
                if w is not None:
                    try:
                        w.config(state="normal")
                    except Exception:
                        pass
        except Exception:
            pass

    def start_task_in_background(self, fn, args=(), kwargs=None):
        """Helper para lanzar una función en background y conectar su salida a task_queue."""
        self._disable_controls()
        run_callable_in_thread(fn, args=args, kwargs=kwargs or {}, out_q=self.task_queue)

    def _on_task_done(self, result):
        """Procesar resultado devuelto por el worker (ej.: actualizar previews)."""
        self._hide_busy_dialog()  # Oculta el aviso de segmentación
        if not result:
            return
        if isinstance(result, dict) and "error" in result:
            self._write_debug(f"Worker error: {result['error']}")
            return
        if isinstance(result, dict) and "mask" in result:
            self.last_mask = result.get("mask")
            self.last_overlay = result.get("overlay")
            input_img = result.get("input_img")
            try:
                if input_img is not None and hasattr(self, "label_input"):
                    self._show_preview(Image.fromarray(input_img), self.label_input)
                if self.last_mask is not None and hasattr(self, "label_mask"):
                    self._show_preview(Image.fromarray(self.last_mask), self.label_mask)
                if self.last_overlay is not None and hasattr(self, "label_overlay"):
                    self._show_preview(Image.fromarray(self.last_overlay), self.label_overlay)
                # Selecciona la pestaña Overlay
                if hasattr(self, "preview_nb"):
                    self.preview_nb.select(2)
            except Exception as e:
                self._write_debug(f"Error updating previews: {e}")

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
        """Carga el modelo y los pesos, comprobando compatibilidad de arquitectura, encoder y función de loss con config.json."""
        weights_path = self.weights_path_var.get()
        if not weights_path or not os.path.exists(weights_path):
            self._show_notification("Error", "Please select a valid weights (.pth) file.")
            return
        try:
            cfg = load_config_from_dir(weights_path)
            arch_cfg = cfg.get("arch")
            encoder_cfg = cfg.get("encoder") or cfg.get("encoder_name")
            loss_cfg = cfg.get("loss") or cfg.get("loss_fn")
            arch_ui = self.arch_var.get()
            encoder_ui = self.encoder_var.get()
            loss_ui = self.loss_var.get()
            compatible = True
            msg = ""
            if arch_cfg and arch_cfg != arch_ui:
                compatible = False
                msg += f"\n- Architecture mismatch: config={arch_cfg}, selected={arch_ui}"
            if encoder_cfg and encoder_cfg != encoder_ui:
                compatible = False
                msg += f"\n- Encoder mismatch: config={encoder_cfg}, selected={encoder_ui}"
            if loss_cfg and loss_cfg != loss_ui:
                compatible = False
                msg += f"\n- Loss mismatch: config={loss_cfg}, selected={loss_ui}"
            if not compatible:
                self._show_notification("Incompatible weights/config", f"The selected weights are not compatible with the current settings:{msg}")
                self._update_model_status("Incompatible", color_style="Error.TLabel", info=msg)
                return
            in_channels = cfg.get("in_channels", 3)
            out_classes = cfg.get("out_classes", 1)
            model = CamVidModel(arch_ui, encoder_ui, in_channels=in_channels, out_classes=out_classes)
            import torch
            state = torch.load(weights_path, map_location=self.device)
            try:
                result = model.load_state_dict(state, strict=False)
                missing = set(result.missing_keys)
                unexpected = set(result.unexpected_keys)
                ignorable = {"mean", "std", "mean", "std"}
                # Si solo faltan/son inesperadas mean y std, mostrar warning pero continuar
                if (missing - ignorable or unexpected - ignorable):
                    # Hay otras claves importantes faltantes o inesperadas
                    msg = f"Missing keys: {missing}\nUnexpected keys: {unexpected}"
                    self._update_model_status("Error: pesos incompatibles", color_style="Error.TLabel")
                    self._write_debug(f"[ERROR] Los pesos no son compatibles con la arquitectura/encoder/configuración actual.\n{msg}")
                    self._show_notification("Error al cargar pesos", "Los pesos seleccionados no son compatibles con la arquitectura, encoder o configuración del modelo.\n\nDetalle: " + msg)
                    return
                elif missing or unexpected:
                    # Solo faltan/son inesperadas mean y std
                    msg = f"Advertencia: faltan o sobran claves no críticas (mean/std). El modelo se ha cargado igualmente.\nMissing: {missing}\nUnexpected: {unexpected}"
                    self._write_debug(msg)
            except Exception as e:
                self._update_model_status("Error: pesos incompatibles", color_style="Error.TLabel")
                self._write_debug(f"[ERROR] Los pesos no son compatibles con la arquitectura/encoder/configuración actual.\n{str(e)}")
                self._show_notification("Error al cargar pesos", "Los pesos seleccionados no son compatibles con la arquitectura, encoder o configuración del modelo.\n\nDetalle: " + str(e))
                return
            self.model = model.to(self.device)
            self._update_model_status("Modelo cargado", color_style="Success.TLabel")
            self._write_debug(f"Modelo cargado correctamente: {arch_ui} / {encoder_ui} / {loss_ui}")
            # Mostrar info de modelo
            self.model_info_label.config(text=f"Arch: {arch_ui}, Encoder: {encoder_ui}, Loss: {loss_ui}, InCh: {in_channels}, OutCl: {out_classes}")
            self.model_info_frame.grid()  # Mostrar info
        except Exception as e:
            self._update_model_status("Error al cargar modelo", color_style="Error.TLabel")
            self._write_debug(f"[ERROR] No se pudo cargar el modelo: {str(e)}")
            self._show_notification("Error al cargar modelo", str(e))
        return

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
        """Lanza la inferencia en un thread para no bloquear la GUI."""
        if not redraw_only:
            self._show_busy_dialog("Segmenting", "Segmenting image, please wait...")

        def _segment_task(redraw_only):
            """Esta función se ejecuta en background — NUNCA tocar widgets aquí."""
            try:
                if getattr(self, "pil_input", None) is None:
                    return {"error": "No input image loaded"}
                if getattr(self, "model", None) is None:
                    return {"error": "Model not loaded"}

                # Preprocesado (usa helper desde models.py)
                pil = self.pil_input
                img_rgb, img_t, original_size = preprocess_image_pil(pil, target_size=self._get_default_resize())
                device = next(self.model.parameters()).device if hasattr(self, "model") else torch.device("cpu")
                img_t = img_t.to(device)
                with torch.no_grad():
                    logits = self.model(img_t)
                thresh = float(self.threshold.get()) if hasattr(self, "threshold") else 0.5
                mask = postprocess_mask(logits, threshold=thresh, out_size=(original_size[0], original_size[1]))
                alpha = float(self.alpha.get()) if hasattr(self, "alpha") else 0.5
                overlay = color_overlay(img_rgb, mask, alpha=alpha)
                return {"mask": mask, "overlay": overlay, "input_img": img_rgb}
            except Exception as e:
                return {"error": str(e)}

        # lanzar en background y retornar inmediatamente
        self.start_task_in_background(_segment_task, args=(redraw_only,))

    def on_save_mask(self):
        """Save the predicted mask."""
        if self.last_mask is None:
            messagebox.showwarning("Save Mask", "No mask to save. Run segmentation first.")
            return
        path = filedialog.asksaveasfilename(title="Save Mask PNG", defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            # Normaliza la ruta antes de guardar
            norm_path = normalize_filename(path)
            cv2.imwrite(norm_path, self.last_mask)
            self._show_notification("Save Mask", f"Saved: {norm_path}")
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
            norm_path = normalize_filename(path)
            bgr = cv2.cvtColor(self.last_overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(norm_path, bgr)
            self._show_notification("Save Overlay", f"Saved: {norm_path}")
        except Exception as e:
            messagebox.showerror("Save Overlay", f"Failed to save overlay:\n{e}")

    def on_metrics_from_pred(self):
        """Compute metrics from predicted mask."""
        if self.last_mask is None:
            messagebox.showwarning("Metrics", "No predicted mask. Run segmentation first.")
            return
        try:
            pil_mask = Image.fromarray(self.last_mask if self.last_mask.ndim == 2 else cv2.cvtColor(self.last_mask, cv2.COLOR_BGR2GRAY))
            self._set_metrics_preview(pil_mask, "Predicted mask")
            mask_bin = (self.last_mask >= 128).astype(np.uint8)
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
            pil = Image.fromarray(img)
            self._set_metrics_preview(pil, f"File: {os.path.basename(path)}")
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
            folder = os.path.dirname(path)
            base = os.path.basename(path)
            norm_base = normalize_filename(base)
            norm_path = os.path.join(folder, norm_base)
            # Si las métricas son una lista (por pétalo), guardar cada una
            metrics_list = self.current_metrics if isinstance(self.current_metrics, list) else [self.current_metrics]
            file_exists = os.path.exists(norm_path)
            with open(norm_path, 'a', newline='', encoding='utf-8') as csvfile:
                # Unir todos los campos posibles
                all_fields = set()
                for m in metrics_list:
                    all_fields.update(m.keys())
                all_fields.update(["timestamp", "image_name"])
                fieldnames = list(all_fields)
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for m in metrics_list:
                    row = {**m}
                    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    row["image_name"] = self.metrics_preview_title.cget("text")
                    writer.writerow(row)
            self._show_notification("Save Metrics", f"Metrics saved to: {norm_path}")
            self._write_debug(f"Metrics saved to CSV: {norm_path}")
        except Exception as e:
            messagebox.showerror("Save Metrics", f"Failed to save metrics:\n{e}")
            self._write_debug(f"Error saving metrics: {e}")

    def _show_metrics(self, res):
        """Display metrics in the tree view. Soporta lista de métricas por pétalo."""
        for child in self.metrics_tree.get_children():
            self.metrics_tree.delete(child)
        if isinstance(res, list):
            for m in res:
                label = m.get("petal_label", "?")
                self.metrics_tree.insert("", tk.END, values=(f"--- Petalo {label} ---", ""))
                for k, v in m.items():
                    if k == "petal_label":
                        continue
                    disp = f"{v:.2f}" if isinstance(v, float) else str(v)
                    self.metrics_tree.insert("", tk.END, values=(k, disp))
        else:
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

        # Usa la ruta seleccionada tal cual (no normalizar la ruta completa)
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
            # Normaliza el nombre del archivo de máscara
            mask_filename = normalize_filename(f"{image_path.stem}_mask.png")
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
                    mask_filename = normalize_filename(f"{image_path.stem}_mask.png")
                    mask_path = results_folder / mask_filename
                    cv2.imwrite(str(mask_path), mask)
                    
                    # Compute metrics
                    mask_bin = (mask >= 128).astype(np.uint8)
                    metrics = compute_normalized_metrics(mask_bin, img_rgb=img_rgb)
                    
                    # Add to CSV data
                    row_data = {
                        'image_name': normalize_filename(image_path.name),
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

    def _show_preview(self, pil_img: Image.Image, label_widget, max_size=(1000, 600)):
        """Muestra una imagen PIL en un widget Label de Tkinter."""
        img = pil_img.copy()
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label_widget.config(image=photo)
        label_widget.image = photo  # Previene garbage collection