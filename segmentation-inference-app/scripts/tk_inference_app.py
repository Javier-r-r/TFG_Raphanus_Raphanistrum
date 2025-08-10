"""
Tkinter desktop app for binary segmentation inference and mask-based metrics (scrollable).

Features:
- Load weights (.pth) and auto-read config.json, dataset_mean.npy next to it
- Select image (PNG/JPG/TIFF), run segmentation, tune threshold & alpha
- Preview Input, Mask, Overlay
- Save mask/overlay to PNG
- Compute personalized metrics from predicted mask or any mask file
- Scroll to access all controls on smaller screens

Run (Windows):
  cd scripts
  .venv\Scripts\activate
  pip install -r requirements.txt
  python tk_inference_app.py
"""

import os
import json
from typing import Optional, Tuple, Any, Dict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

import torch
import segmentation_models_pytorch as smp

# Local metrics helper
from metrics import compute_metrics_from_mask


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


# ---------------------------
# Scrollable Frame
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
# Tkinter App
# ---------------------------

class SegTkApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Segmentation Inference (Tk) - v1.0.1")
        root.geometry("1100x800")
        root.minsize(900, 650)

        # Scrollable wrapper
        self.scroll = ScrollableFrame(root)
        self.scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Model/meta state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.force_imagenet = tk.BooleanVar(value=False)
        self.default_resize_var = tk.StringVar(value="640")
        self.arch_var = tk.StringVar(value="Unet")
        self.encoder_var = tk.StringVar(value="resnet34")
        self.weights_path_var = tk.StringVar(value="")

        self.model: Optional[torch.nn.Module] = None
        self.debug_meta: Dict[str, Any] = {}

        # Inference state
        self.pil_input: Optional[Image.Image] = None
        self.np_input_rgb: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.last_overlay: Optional[np.ndarray] = None

        # UI state
        self.threshold = tk.DoubleVar(value=0.5)
        self.alpha = tk.DoubleVar(value=0.5)

        self.photo_input = None
        self.photo_mask = None
        self.photo_overlay = None

        self._build_ui(self.scroll.container)

    def _build_ui(self, parent: ttk.Frame):
        # Top controls
        top = ttk.Frame(parent, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Weights (.pth):").grid(row=0, column=0, sticky="w")
        e_weights = ttk.Entry(top, textvariable=self.weights_path_var, width=70)
        e_weights.grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(top, text="Browse", command=self.on_browse_weights).grid(row=0, column=2, padx=5)
        ttk.Button(top, text="Load Model", command=self.on_load_model).grid(row=0, column=3, padx=5)

        ttk.Label(top, text="Arch:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.arch_var, width=20).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(top, text="Encoder:").grid(row=1, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.encoder_var, width=20).grid(row=1, column=3, sticky="w", padx=5)

        ttk.Label(top, text="Default Resize (px):").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.default_resize_var, width=10).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Checkbutton(top, text="Force ImageNet stats", variable=self.force_imagenet).grid(row=2, column=2, sticky="w", padx=5)

        # Actions
        actions = ttk.Frame(parent, padding=(10, 0, 10, 10))
        actions.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(actions, text="Open Image", command=self.on_open_image).pack(side=tk.LEFT)
        ttk.Button(actions, text="Segment", command=self.on_segment).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Mask", command=self.on_save_mask).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Save Overlay", command=self.on_save_overlay).pack(side=tk.LEFT, padx=6)

        # Sliders
        sliders = ttk.Frame(parent, padding=(10, 0, 10, 10))
        sliders.pack(side=tk.TOP, fill=tk.X)

        thr_frame = ttk.Frame(sliders)
        thr_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(thr_frame, text="Threshold:").pack(side=tk.LEFT)
        ttk.Scale(thr_frame, from_=0.0, to=1.0, variable=self.threshold, orient=tk.HORIZONTAL, command=self._on_slider_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.label_thr = ttk.Label(thr_frame, text=f"{self.threshold.get():.2f}")
        self.label_thr.pack(side=tk.LEFT)

        alpha_frame = ttk.Frame(sliders)
        alpha_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(alpha_frame, text="Overlay alpha:").pack(side=tk.LEFT)
        ttk.Scale(alpha_frame, from_=0.0, to=1.0, variable=self.alpha, orient=tk.HORIZONTAL, command=self._on_slider_change).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.label_alpha = ttk.Label(alpha_frame, text=f"{self.alpha.get():.2f}")
        self.label_alpha.pack(side=tk.LEFT)

        # Previews grid
        previews = ttk.Frame(parent, padding=10)
        previews.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        col_left = ttk.Frame(previews)
        col_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        col_right = ttk.Frame(previews)
        col_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(col_left, text="Input").pack(anchor="w")
        self.label_input = ttk.Label(col_left, background="#222")
        self.label_input.pack(fill=tk.BOTH, expand=True)

        ttk.Label(col_right, text="Mask").pack(anchor="w")
        self.label_mask = ttk.Label(col_right, background="#222")
        self.label_mask.pack(fill=tk.BOTH, expand=True)

        # Overlay full width
        overlay_frame = ttk.Frame(parent, padding=10)
        overlay_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(overlay_frame, text="Overlay").pack(anchor="w")
        self.label_overlay = ttk.Label(overlay_frame, background="#222")
        self.label_overlay.pack(fill=tk.BOTH, expand=True)

        # Metrics section (the buttons you couldn't reach)
        metrics_section = ttk.Frame(parent, padding=10)
        metrics_section.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(metrics_section, text="Metrics (from mask)").pack(anchor="w")
        m_buttons = ttk.Frame(metrics_section)
        m_buttons.pack(side=tk.TOP, fill=tk.X, pady=(4, 8))
        ttk.Button(m_buttons, text="Metrics from predicted mask", command=self.on_metrics_from_pred).pack(side=tk.LEFT)
        ttk.Button(m_buttons, text="Metrics from mask file...", command=self.on_metrics_from_file).pack(side=tk.LEFT, padx=8)

        self.metrics_tree = ttk.Treeview(metrics_section, columns=("metric", "value"), show="headings", height=6)
        self.metrics_tree.heading("metric", text="Metric")
        self.metrics_tree.heading("value", text="Value")
        self.metrics_tree.column("metric", width=380, anchor="w")
        self.metrics_tree.column("value", width=120, anchor="e")
        self.metrics_tree.pack(side=tk.TOP, fill=tk.X, expand=False)

        # Debug info
        debug_frame = ttk.Frame(parent, padding=10)
        debug_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(debug_frame, text="Debug").pack(anchor="w")

        # Container for the debug Text with a vertical scrollbar
        debug_container = ttk.Frame(debug_frame)
        debug_container.pack(fill=tk.BOTH, expand=True)

        self.debug_text = tk.Text(debug_container, height=10, wrap="word")
        debug_scroll = ttk.Scrollbar(debug_container, orient="vertical", command=self.debug_text.yview)
        self.debug_text.configure(yscrollcommand=debug_scroll.set)

        # Layout: Text on the left, Scrollbar on the right
        self.debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        debug_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._write_debug("Ready. Load model, open an image, then Segment.")

    # ------------- UI callbacks -------------

    def _write_debug(self, msg: str):
        self.debug_text.insert(tk.END, msg + "\n")
        self.debug_text.see(tk.END)

    def _on_slider_change(self, _evt=None):
        self.label_thr.config(text=f"{self.threshold.get():.2f}")
        self.label_alpha.config(text=f"{self.alpha.get():.2f}")
        if self.pil_input is not None and self.model is not None:
            try:
                self.on_segment(redraw_only=True)
            except Exception:
                pass

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
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        mean_src, std_src = "imagenet", "imagenet"
        if not self.force_imagenet.get():
            mean, std = find_stats_in_dir(path)
            mean_src, std_src = model.set_stats(mean, std, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        state = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
            self._show_preview(pil, self.label_input, max_size=(520, 360))
            self.last_mask = None
            self.last_overlay = None
            self.label_mask.config(image="")
            self.label_overlay.config(image="")
            self.photo_mask = None
            self.photo_overlay = None
            self._write_debug(f"Loaded image: {os.path.basename(path)} size={pil.size}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def _get_default_resize(self) -> Optional[int]:
        s = self.default_resize_var.get().strip()
        if not s:
            return None
        try:
            v = int(s)
            return v if v > 0 else None
        except Exception:
            return None

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
            self._show_preview(pil_mask, self.label_mask, max_size=(520, 360), is_mask=True)

            pil_overlay = Image.fromarray(overlay)
            self._show_preview(pil_overlay, self.label_overlay, max_size=(1050, 360))

            self._write_debug(f"Inference done. thr={thr:.2f} alpha={alpha:.2f}, out={original_size}")
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed:\n{e}")

    def _show_preview(self, pil_img: Image.Image, widget: ttk.Label, max_size=(520, 360), is_mask=False):
        img = pil_img.copy()
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        if is_mask and img.mode != "L":
            img = img.convert("L")
        elif not is_mask and img.mode != "RGB":
            img = img.convert("RGB")
        photo = ImageTk.PhotoImage(img)
        widget.config(image=photo)
        if widget is self.label_input:
            self.photo_input = photo
        elif widget is self.label_mask:
            self.photo_mask = photo
        else:
            self.photo_overlay = photo

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
