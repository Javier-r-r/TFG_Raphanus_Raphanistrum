"""
Model classes and utilities for segmentation inference.
"""
import os
import json
from typing import Optional, Tuple, Any, Dict

import numpy as np
import cv2
from PIL import Image

import torch
import segmentation_models_pytorch as smp


class CamVidModel(torch.nn.Module):
    def __init__(self, arch: str, encoder_name: str, in_channels: int = 3, out_classes: int = 1, **kwargs):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

    def set_stats(self, mean: Optional[np.ndarray], std: Optional[np.ndarray], device: torch.device) -> Tuple[str, str]:
        # Always use default ImageNet stats
        return "imagenet", "imagenet"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        return self.model(image)


def load_config_from_dir(weights_path: str) -> Dict[str, Any]:
    """Load config.json from the same directory as weights file."""
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


def preprocess_image_pil(pil_img: Image.Image, target_size: Optional[int] = None):
    """Preprocess PIL image for model inference."""
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
    """Convert model logits to binary mask."""
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    mask = pred[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
    if out_size:
        mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
    return mask


def color_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color=(0, 255, 0)) -> np.ndarray:
    """Create colored overlay of mask on RGB image."""
    out = rgb.copy()
    m = mask > 0
    if m.any():
        overlay = np.zeros_like(out)
        overlay[m] = color
        out[m] = (out[m].astype(np.float32) * (1 - alpha) + overlay[m].astype(np.float32) * alpha).astype(np.uint8)
    return out
    return img_rgb, img_t, original_size


def postprocess_mask(logits: torch.Tensor, threshold: float, out_size: Tuple[int, int]) -> np.ndarray:
    """Convert model logits to binary mask."""
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    mask = pred[0, 0].detach().cpu().numpy().astype(np.uint8) * 255
    if out_size:
        mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
    return mask


def color_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5, color=(0, 255, 0)) -> np.ndarray:
    """Create colored overlay of mask on RGB image."""
    out = rgb.copy()
    m = mask > 0
    if m.any():
        overlay = np.zeros_like(out)
        overlay[m] = color
        out[m] = (out[m].astype(np.float32) * (1 - alpha) + overlay[m].astype(np.float32) * alpha).astype(np.uint8)
    return out
