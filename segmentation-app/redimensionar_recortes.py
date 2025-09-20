import cv2
import numpy as np
import os

from pathlib import Path


input_folder = "petalos_recortados"
output_folder = "petalos_iguales"
input_mask_folder = "petalos_recortados_mascara"
output_mask_folder = "petalos_iguales_mascara"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Encuentra el tamaño máximo
max_w, max_h = 0, 0
for img_path in Path(input_folder).glob("*.png"):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    h, w = img.shape[:2]
    max_w = max(max_w, w)
    max_h = max(max_h, h)

# Rellena cada imagen y su máscara hasta el tamaño máximo con fondo blanco/negro
for img_path in Path(input_folder).glob("*.png"):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    h, w = img.shape[:2]
    top = (max_h - h) // 2
    bottom = max_h - h - top
    left = (max_w - w) // 2
    right = max_w - w - left

    # Si la imagen tiene canal alfa, convertir fondo a blanco
    if img.shape[-1] == 4:
        bgr = img[..., :3]
        alpha = img[..., 3]
        bg = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
        bg[top:top+h, left:left+w][alpha > 0] = bgr[alpha > 0]
        img_padded = bg
    else:
        img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))

    out_name = os.path.join(output_folder, img_path.name)
    cv2.imwrite(out_name, img_padded)

    # Procesar la máscara correspondiente
    mask_path = Path(input_mask_folder) / img_path.name
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask_padded = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            out_mask_name = os.path.join(output_mask_folder, img_path.name)
            cv2.imwrite(out_mask_name, mask_padded)