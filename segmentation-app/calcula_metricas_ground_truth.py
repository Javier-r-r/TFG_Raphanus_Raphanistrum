"""
script para calcular métricas de referencia (ground truth) a partir de
una carpeta de imágenes y su correspondiente carpeta de máscaras.

Este módulo busca en `MASKS_DIR` todas las imágenes de máscara, localiza
la imagen RGB original con el mismo nombre base en `IMAGES_DIR`, convierte
la máscara a binaria y calcula las métricas normalizadas usando
`compute_normalized_metrics` (definida en `metrics`). Los resultados se
guardan en un archivo CSV con formato similar a `metrics_summary.csv`.

Uso básico::

    python calcula_metricas_ground_truth.py

Los valores por defecto para `IMAGES_DIR`, `MASKS_DIR` y `OUTPUT_CSV`
se pueden modificar desde este archivo o adaptarlos para pasarlos por
línea de comandos en una futura versión.
"""

import os
import cv2
import numpy as np
import csv

from metrics import compute_normalized_metrics


IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']


def find_image_for_mask(base_name, images_dir):
    """Buscar la imagen RGB correspondiente a una máscara.

    Args:
        base_name (str): Nombre de archivo sin extensión (por ejemplo, "img001").
        images_dir (str): Carpeta donde buscar las imágenes RGB.

    Returns:
        str|None: Ruta completa de la imagen encontrada, o ``None`` si no existe.
    """
    for ext in IMAGE_EXTS:
        potential_path = os.path.join(images_dir, base_name + ext)
        if os.path.exists(potential_path):
            return potential_path
    return None


def process_mask(mask_path, images_dir):
    """Procesar una máscara: localizar la imagen, leer archivos y calcular métricas.

    Args:
        mask_path (str): Ruta al archivo de máscara (escala de grises esperada).
        images_dir (str): Carpeta donde buscar la imagen RGB correspondiente.

    Returns:
        dict|None: Diccionario con las métricas calculadas (incluye clave
        `'image_name'`) o ``None`` si hubo errores de lectura o no se
        encontró la imagen original.
    """
    mask_name = os.path.basename(mask_path)
    base_name, _ = os.path.splitext(mask_name)
    image_path = find_image_for_mask(base_name, images_dir)
    if image_path is None:
        print(f"No se encontró la imagen original para la máscara: {mask_name}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error leyendo imagen RGB: {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error leyendo máscara: {mask_path}")
        return None

    mask_bin = (mask >= 128).astype(np.uint8)
    h, w = mask_bin.shape
    current_diagonal = np.sqrt(h**2 + w**2)
    metrics = compute_normalized_metrics(mask_bin, img_rgb=img_rgb, reference_resolution=current_diagonal)
    metrics['image_name'] = mask_name
    return metrics


def process_all_masks(images_dir="petalos_iguales_224", masks_dir="petalos_iguales_mascara_224", output_csv="metrics_ground_truth.csv"):
    """Recorrer la carpeta de máscaras, calcular métricas y volcar a CSV.

    Args:
        images_dir (str): Carpeta con las imágenes RGB originales.
        masks_dir (str): Carpeta con las máscaras binarias o en escala de grises.
        output_csv (str): Ruta del CSV de salida.
    """
    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(tuple(IMAGE_EXTS))]
    results = []

    for mask_name in mask_files:
        mask_path = os.path.join(masks_dir, mask_name)
        metrics = process_mask(mask_path, images_dir)
        if metrics is not None:
            results.append(metrics)

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
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for row in results:
                out_row = {k: row.get(k, '') for k in csv_fields}
                out_row['mask_name'] = row.get('image_name', '').rsplit('.', 1)[0] + '_mask.png'
                out_row['threshold'] = 0.5
                if not out_row.get('image_size'):
                    out_row['image_size'] = ''
                writer.writerow(out_row)
        print(f"Métricas guardadas en {output_csv}")
    else:
        print("No se calcularon métricas.")


if __name__ == '__main__':
    process_all_masks()
