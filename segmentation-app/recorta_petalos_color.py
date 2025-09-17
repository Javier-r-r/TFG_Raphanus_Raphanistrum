import cv2
import numpy as np
import os

from pathlib import Path


# Solo amarillo (ajusta si es necesario)
LOWER_YELLOW = np.array([20, 80, 80])   # HSV lower bound para amarillo
UPPER_YELLOW = np.array([40, 255, 255]) # HSV upper bound para amarillo

input_folder = "segmentation-app/petalos"
output_folder = "petalos_recortados"
mask_folder = "segmentation-app/masks"  # Cambia esta ruta si tus máscaras están en otra carpeta
output_mask_folder = "petalos_recortados_mascara"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

MARGIN = 5  # píxeles de margen extra

for img_path in Path(input_folder).glob("*.tif"):
    img = cv2.imread(str(img_path))
    # Cargar la máscara correspondiente (debe tener el mismo nombre base pero extensión .png)
    mask_path = Path(mask_folder) / (img_path.stem + ".png")
    if not mask_path.exists():
        print(f"No se encontró la máscara para {img_path.name}")
        continue
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    # Usar la máscara binaria para encontrar regiones conectadas (no solo amarillo)
    mask_bin = (mask_img > 0).astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(mask_bin)
    for label in range(1, num_labels):
        region_mask = (labels == label).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(region_mask)
        if w < 30 or h < 30:
            continue  # descarta regiones pequeñas
        # Añadir margen, asegurando que no se salga de la imagen
        x0 = max(x - MARGIN, 0)
        y0 = max(y - MARGIN, 0)
        x1 = min(x + w + MARGIN, img.shape[1])
        y1 = min(y + h + MARGIN, img.shape[0])
        region_crop = img[y0:y1, x0:x1]
        out_name = f"{img_path.stem}_region{label}.png"
        cv2.imwrite(os.path.join(output_folder, out_name), region_crop)
        # Recortar la misma región en la máscara y guardar
        region_mask_crop = mask_img[y0:y1, x0:x1]
        out_mask_name = f"{img_path.stem}_region{label}.png"
        cv2.imwrite(os.path.join(output_mask_folder, out_mask_name), region_mask_crop)
