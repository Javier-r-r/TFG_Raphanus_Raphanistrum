import numpy as np
import os
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import argparse
import shutil

def cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras, tamaño=(128, 128)):
    """Carga imágenes y máscaras preservando la calidad original"""
    imagenes = sorted(glob.glob(os.path.join(ruta_imagenes, "*.tif")))
    mascaras = sorted(glob.glob(os.path.join(ruta_mascaras, "*.png")))

    if len(imagenes) != len(mascaras):
        raise ValueError(f"Número de imágenes ({len(imagenes)}) y máscaras ({len(mascaras)}) no coincide")

    X = []
    y = []

    for img_path, mask_path in zip(imagenes, mascaras):
        # Cargar imagen
        img = Image.open(img_path).convert('RGB')
        img = img.resize(tamaño, Image.Resampling.LANCZOS)
        img_array = np.array(img)

        # Cargar máscara preservando líneas finas
        mask = Image.open(mask_path)
        
        # Redimensionar manteniendo bordes nítidos
        mask = mask.resize(tamaño, Image.Resampling.NEAREST)
        mask_array = np.array(mask)

        # Mantener valores originales (0 y 255)
        mask_array = (mask_array > 127).astype(np.uint8) * 255

        X.append(img_array)
        y.append(np.expand_dims(mask_array, axis=-1))

    return np.array(X), np.array(y)

def guardar_conjunto_npz(conjunto, nombre_archivo):
    """Guarda los datos en formato npz con compresión óptima"""
    np.savez_compressed(
        nombre_archivo,
        X_train=conjunto['X_train'],
        X_val=conjunto['X_val'],
        X_test=conjunto['X_test'],
        y_train=conjunto['y_train'],
        y_val=conjunto['y_val'],
        y_test=conjunto['y_test']
    )

def generar_conjuntos_multiples(ruta_imagenes, ruta_mascaras, n_conjuntos=3):
    """Genera n conjuntos de datos divididos aleatoriamente"""
    X, y = cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras)

    for i in range(n_conjuntos):
        random_state = 42 + i

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state
        )

        # Guardar con compresión
        guardar_conjunto_npz(
            {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            },
            f"conjunto_{i+1}.npz"
        )

    print(f"Se generaron {n_conjuntos} conjuntos de datos (.npz) con máxima calidad")

def generar_split_directorio(ruta_imagenes, ruta_mascaras, output_dir, seed=42, tamaño=(640, 640)):
    """Divide el dataset en 70/15/15 y guarda imágenes/máscaras en carpetas."""
    imagenes = sorted(glob.glob(os.path.join(ruta_imagenes, "*.tif")))
    mascaras = sorted(glob.glob(os.path.join(ruta_mascaras, "*.png")))

    if len(imagenes) != len(mascaras):
        raise ValueError(f"Número de imágenes ({len(imagenes)}) y máscaras ({len(mascaras)}) no coincide")

    # Emparejar por nombre base
    pares = [(img, os.path.join(ruta_mascaras, os.path.basename(img).replace('.tif', '.png').replace('.TIF', '.png')))
             for img in imagenes if os.path.exists(os.path.join(ruta_mascaras, os.path.basename(img).replace('.tif', '.png').replace('.TIF', '.png')))]
    if len(pares) == 0:
        raise ValueError("No se encontraron pares válidos de imagen/máscara.")

    # Split 70/15/15
    train_pairs, temp_pairs = train_test_split(pares, test_size=0.3, random_state=seed)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.5, random_state=seed)

    split_dict = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }

    for split, pairs in split_dict.items():
        img_dir = os.path.join(output_dir, split, 'images')
        mask_dir = os.path.join(output_dir, split, 'masks')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        for img_path, mask_path in pairs:
            # Copiar y redimensionar imagen
            img = Image.open(img_path).convert('RGB')
            img = img.resize(tamaño, Image.Resampling.LANCZOS)
            img.save(os.path.join(img_dir, os.path.basename(img_path)))
            # Copiar y redimensionar máscara
            mask = Image.open(mask_path)
            mask = mask.resize(tamaño, Image.Resampling.NEAREST)
            mask.save(os.path.join(mask_dir, os.path.basename(mask_path)))

    print(f"Split {output_dir} generado con {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test.")

# Uso:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--masks_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, nargs=2, default=[640, 640])
    args = parser.parse_args()

    generar_split_directorio(args.images_dir, args.masks_dir, args.output_dir, seed=args.seed, tamaño=tuple(args.size))
