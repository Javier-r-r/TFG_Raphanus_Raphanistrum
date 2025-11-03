"""
Model classes and utilities for segmentation inference.
"""
import os
import json
import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

from PIL import Image
from typing import Optional, Tuple, Any, Dict
# Advanced postprocessing removed — keep file minimal for raw predictions


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


def denoise_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Elimina ruido pequeño usando la misma rutina ligera de prueba (prueba_post.py).

    - structuring element elíptico de tamaño `kernel_size`
    - apertura seguida de cierre (open -> close)

    Esta implementación replica el comportamiento de `prueba_post.py`.
    """
    # Manejo defensivo
    if mask is None:
        return mask

    # Asegurar valores en {0,255} y tipo uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # Kernel elíptico similar al usado en el script de prueba
    k = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Operaciones: apertura (open) luego cierre (close)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # Normalizar a 0/255
    result = (closed > 0).astype(np.uint8) * 255
    return result


def preprocess_image_pil(pil_img: Image.Image, target_size: Optional[int] = None):
    """Preprocess PIL image for model inference."""
    img_rgb = np.array(pil_img)  # RGB
    h, w = img_rgb.shape[:2]
    original_size = (w, h)
    
    # Force 224x224 size to match training (models were trained on 224x224)
    model_size = 224
    if target_size and target_size > 0:
        model_size = target_size
    
    img_rs = cv2.resize(img_rgb, (model_size, model_size), interpolation=cv2.INTER_AREA)
    img_t = torch.from_numpy(img_rs.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    return img_rgb, img_t, original_size


def postprocess_mask(logits: torch.Tensor, threshold: float, out_size: Tuple[int, int], denoise: bool = False, kernel_size: int = 3) -> np.ndarray:
    """Convert model logits to binary mask (0/255).

    Opcional: elimina ruido pequeño usando abertura morfológica si denoise=True.

    - Aplica sigmoid -> threshold -> (opcional) denoise -> resize.
    """
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    mask = pred[0, 0].detach().cpu().numpy().astype(np.uint8) * 255

    # Eliminar ruido pequeño (apertura morfológica) opcional
    if denoise:
        mask = denoise_mask(mask, kernel_size=kernel_size)

    if out_size:
        # Choose interpolation to reduce artifacts: AREA for downscale, LINEAR for up
        h_orig, w_orig = out_size[1], out_size[0]
        h_mask, w_mask = mask.shape
        if h_orig * w_orig < h_mask * w_mask:
            mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_AREA)
        else:
            mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_LINEAR)
            mask = (mask > 127).astype(np.uint8) * 255

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
