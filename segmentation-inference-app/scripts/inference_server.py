import argparse
import json
import os
import sys
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import segmentation_models_pytorch as smp

from metrics import compute_metrics_from_mask


class CamVidModel(torch.nn.Module):
    """
    Wrapper around segmentation_models_pytorch with dataset mean/std normalization.
    Mirrors your training flow.
    """
    def __init__(self, arch: str, encoder_name: str, in_channels: int = 3, out_classes: int = 1, **kwargs):
        super().__init__()
        # Defaults to ImageNet stats; may be overridden via set_stats()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

    def set_stats(self, mean: Optional[np.ndarray], std: Optional[np.ndarray], device: torch.device) -> Tuple[str, str]:
        """
        Set mean/std if provided. Returns sources used ("imagenet" or filename).
        """
        mean_src = "imagenet"
        std_src = "imagenet"

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
    """
    Try to locate config.json in the same directory as weights to retrieve arch/encoder.
    """
    cfg: Dict[str, Any] = {}
    base_dir = os.path.dirname(os.path.abspath(weights_path))
    cfg_path = os.path.join(base_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            pass
    return cfg


def find_stats_in_dir(weights_path: str):
    """
    Find dataset_mean.npy and dataset_std.npy next to weights.
    """
    base_dir = os.path.dirname(os.path.abspath(weights_path))
    mean_path = os.path.join(base_dir, "dataset_mean.npy")
    std_path = os.path.join(base_dir, "dataset_std.npy")
    mean = np.load(mean_path) if os.path.exists(mean_path) else None
    std = np.load(std_path) if os.path.exists(std_path) else None
    return mean, std


def build_model(weights_path: str, arch: Optional[str], encoder_name: Optional[str], device: torch.device, force_imagenet_stats: bool = False):
    """
    Build model, load stats and weights, return (model, debug_meta).
    """
    cfg = load_config_from_dir(weights_path)
    arch = arch or cfg.get("arch") or cfg.get("arquitectura") or "Unet"
    encoder_name = encoder_name or cfg.get("encoder_name") or "resnet34"

    model = CamVidModel(arch=arch, encoder_name=encoder_name, in_channels=3, out_classes=1)
    model.to(device)

    # Stats
    if force_imagenet_stats:
        mean_src, std_src = "imagenet", "imagenet"
    else:
        mean, std = find_stats_in_dir(weights_path)
        mean_src, std_src = model.set_stats(mean, std, device)

    # Weights
    state = torch.load(weights_path, map_location=device)
    load_res = model.load_state_dict(state, strict=False)

    debug_meta = {
        "arch": arch,
        "encoder_name": encoder_name,
        "weights_path": os.path.abspath(weights_path),
        "device": str(device),
        "used_mean_source": mean_src,
        "used_std_source": std_src,
        "load_missing_keys_count": len(load_res.missing_keys),
        "load_unexpected_keys_count": len(load_res.unexpected_keys),
        "load_missing_keys_sample": load_res.missing_keys[:8],
        "load_unexpected_keys_sample": load_res.unexpected_keys[:8],
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": torch.__version__,
        "smp_version": getattr(smp, "__version__", "unknown"),
        "opencv_version": cv2.__version__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "server_version": "1.3.0",
    }

    model.eval()
    return model, debug_meta


def preprocess_image(file_bytes: bytes, target_size: Optional[int] = None):
    """
    Read image as RGB float tensor (1,3,H,W) in [0,1], optionally resize to square target_size.
    Returns (np_img_rgb, tensor_img).
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size and target_size > 0:
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return img, img_t


def postprocess_mask(logits: torch.Tensor, threshold: float, out_size: Tuple[int, int] | None = None) -> np.ndarray:
    """
    logits: (1,1,H,W) -> returns uint8 mask (H,W) with values {0,255}
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    mask = pred[0, 0].cpu().numpy().astype(np.uint8) * 255

    if out_size is not None:
        mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
    return mask


def probs_to_uint8(probs: torch.Tensor, out_size: Tuple[int, int] | None = None) -> np.ndarray:
    """
    probs: (1,1,H,W) -> uint8 [0..255] heatmap
    """
    p = probs[0, 0].clamp(0, 1).cpu().numpy()
    p8 = (p * 255.0).round().astype(np.uint8)
    if out_size is not None:
        p8 = cv2.resize(p8, out_size, interpolation=cv2.INTER_LINEAR)
    return p8


def apply_colormap(gray: np.ndarray, cmap: str) -> np.ndarray:
    """
    gray: uint8 (H,W). Returns BGR image if colored, or gray if 'none'.
    """
    cmap = (cmap or "none").lower()
    if cmap in ("none", "gray", "grayscale"):
        return gray
    colormaps = {
        "jet": cv2.COLORMAP_JET,
        "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
        "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
        "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
        "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
        "plasma": getattr(cv2, "COLORMAP_PLASMA", cv2.COLORMAP_JET),
    }
    code = colormaps.get(cmap, cv2.COLORMAP_JET)
    return cv2.applyColorMap(gray, code)


app = FastAPI(title="Segmentation Inference Server", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals
MODEL: Optional[torch.nn.Module] = None
DEVICE = torch.device("cpu")
DEFAULT_RESIZE: Optional[int] = None
DEBUG_META: Dict[str, Any] = {}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    threshold: float = Form(0.5),
    resize: Optional[int] = Form(None),
):
    try:
        bytes_ = await image.read()
        # Preprocess
        original_np, img_t = preprocess_image(bytes_, target_size=resize or DEFAULT_RESIZE)
        h, w, _ = original_np.shape

        with torch.no_grad():
            logits = MODEL(img_t.to(DEVICE))

        mask = postprocess_mask(logits, float(threshold), out_size=(w, h))

        ok, buf = cv2.imencode(".png", mask)
        if not ok:
            return JSONResponse(status_code=500, content={"error": "Failed to encode mask"})
        return Response(content=buf.tobytes(), media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ===== Metrics endpoints (mask-based) =====

@app.post("/metrics/from-mask")
async def metrics_from_mask(mask: UploadFile = File(...), bin_thresh: int = Form(128)):
    """
    Compute personalized metrics from an already binary (or grayscale) mask image.
    - bin_thresh: if the mask isn't strictly 0/255, we binarize with (pixel >= bin_thresh)
    """
    try:
        bytes_ = await mask.read()
        arr = np.frombuffer(bytes_, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Unable to decode mask image"})

        mask_bin = (img >= int(bin_thresh)).astype(np.uint8)
        metrics = compute_metrics_from_mask(mask_bin)
        return metrics
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ========== Debug endpoints ==========

@app.get("/debug/ping")
async def debug_ping():
    return {"ok": True}


@app.get("/debug/info")
async def debug_info():
    meta = dict(DEBUG_META) if DEBUG_META else {}
    meta.update({
        "device": str(DEVICE),
        "default_resize": DEFAULT_RESIZE,
        "has_model": MODEL is not None,
        "server_version": "1.3.0",
    })
    return meta


@app.post("/debug/probmap")
async def debug_probmap(
    image: UploadFile = File(...),
    resize: Optional[int] = Form(None),
    colormap: str = Form("none"),
):
    """
    Returns an 8-bit PNG heatmap of probabilities (0-255).
    Use colormap: none | jet | turbo | magma | inferno | viridis | plasma
    """
    try:
        bytes_ = await image.read()
        original_np, img_t = preprocess_image(bytes_, target_size=resize or DEFAULT_RESIZE)
        h, w, _ = original_np.shape

        with torch.no_grad():
            logits = MODEL(img_t.to(DEVICE))
            probs = torch.sigmoid(logits)

        p8 = probs_to_uint8(probs, out_size=(w, h))
        colored = apply_colormap(p8, colormap)

        ok, buf = cv2.imencode(".png", colored)
        if not ok:
            return JSONResponse(status_code=500, content={"error": "Failed to encode probmap"})
        return Response(content=buf.tobytes(), media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/debug/stats")
async def debug_stats(
    image: UploadFile = File(...),
    threshold: float = Form(0.5),
    resize: Optional[int] = Form(None),
):
    """
    Returns summary stats for logits and probabilities, plus fraction >= threshold.
    """
    try:
        bytes_ = await image.read()
        original_np, img_t = preprocess_image(bytes_, target_size=resize or DEFAULT_RESIZE)

        with torch.no_grad():
            logits = MODEL(img_t.to(DEVICE))
            probs = torch.sigmoid(logits)

        l = logits[0, 0].detach().cpu().numpy()
        p = probs[0, 0].detach().cpu().numpy()

        stats = {
            "logits": {
                "min": float(l.min()),
                "max": float(l.max()),
                "mean": float(l.mean()),
                "std": float(l.std()),
            },
            "probs": {
                "min": float(p.min()),
                "max": float(p.max()),
                "mean": float(p.mean()),
                "std": float(p.std()),
                "frac_ge_threshold": float((p >= float(threshold)).mean()),
            },
            "model_out_shape": [int(logits.shape[-2]), int(logits.shape[-1])],
            "server_version": "1.3.0",
        }
        return stats
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="FastAPI inference server for SMP binary segmentation (v1.3.0)")
    parser.add_argument("--weights", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--arch", type=str, default=None, help="Model architecture if not in config.json (e.g., Unet, FPN)")
    parser.add_argument("--encoder", type=str, default=None, help="Encoder name if not in config.json (e.g., resnet34)")
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | auto")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--default-resize", type=int, default=None, help="Resize input to NxN before inference (e.g., 640)")
    parser.add_argument("--force-imagenet-stats", action="store_true", help="Ignore dataset_mean.npy/std.npy and use ImageNet stats")
    args = parser.parse_args()

    global MODEL, DEVICE, DEFAULT_RESIZE, DEBUG_META

    if args.device == "auto":
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(args.device)

    DEFAULT_RESIZE = args.default_resize

    MODEL, DEBUG_META = build_model(args.weights, args.arch, args.encoder, DEVICE, args.force_imagenet_stats)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Loaded arch={DEBUG_META.get('arch')} encoder={DEBUG_META.get('encoder_name')}")
    print(f"[INFO] Mean source={DEBUG_META.get('used_mean_source')} Std source={DEBUG_META.get('used_std_source')}")
    print(f"[INFO] Missing keys: {DEBUG_META.get('load_missing_keys_count')} Unexpected: {DEBUG_META.get('load_unexpected_keys_count')}")
    print(f"[INFO] Server v1.3.0 ready.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
