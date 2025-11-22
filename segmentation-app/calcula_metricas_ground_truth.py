import os
import cv2
import numpy as np
import csv

from metrics import compute_normalized_metrics


# Cambia estos paths por los que necesites o pásalos por línea de comandos
IMAGES_DIR = "petalos_iguales_224"
MASKS_DIR = "petalos_iguales_mascara_224"
OUTPUT_CSV = "metrics_ground_truth.csv"

# Busca todos los archivos de la carpeta de máscaras
mask_files = [f for f in os.listdir(MASKS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]

results = []

for mask_name in mask_files:
    mask_path = os.path.join(MASKS_DIR, mask_name)
    # Buscar la imagen original con el mismo nombre
    base_name, _ = os.path.splitext(mask_name)
    
    # Buscar la imagen original con el mismo nombre base
    image_path = None
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        potential_path = os.path.join(IMAGES_DIR, base_name + ext)
        if os.path.exists(potential_path):
            image_path = potential_path
            break
    
    if image_path is None:
        print(f"No se encontró la imagen original para la máscara: {mask_name}")
        continue
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None or img_rgb is None:
        print(f"Error leyendo {mask_name}")
        continue
    mask_bin = (mask >= 128).astype(np.uint8)
    # Como todas las imágenes son del mismo tamaño, usar la diagonal actual
    # para desactivar la normalización (factor de escala = 1.0)
    h, w = mask_bin.shape
    current_diagonal = np.sqrt(h**2 + w**2)
    metrics = compute_normalized_metrics(mask_bin, img_rgb=img_rgb, reference_resolution=current_diagonal)
    metrics['image_name'] = mask_name
    results.append(metrics)

# Guardar a CSV con el mismo formato que metrics_summary.csv
csv_fields = [
    'image_name',
    'mask_name',
    'threshold',
    'image_size',
    'Vein Density (VD)',
    'Vein Thickness (VT)',
    'Areole Size (AS)',
    'Number of Areoles (NA)',
    'Branching Angle (BA)',
    'Vein-to-Vein Distance (VVD)',
    'Main Veins (MV)'
]

if results:
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in results:
            out_row = {k: row.get(k, '') for k in csv_fields}
            # Rellenar campos que no están en el cálculo ground truth
            out_row['mask_name'] = row['image_name'].replace('.tif', '_mask.png')
            out_row['threshold'] = 0.5
            if 'image_size' not in out_row or not out_row['image_size']:
                if 'img_rgb' in locals() and img_rgb is not None:
                    out_row['image_size'] = f"{img_rgb.shape[1]}x{img_rgb.shape[0]}"
                else:
                    out_row['image_size'] = ''
            writer.writerow(out_row)
    print(f"Métricas guardadas en {OUTPUT_CSV}")
else:
    print("No se calcularon métricas.")
