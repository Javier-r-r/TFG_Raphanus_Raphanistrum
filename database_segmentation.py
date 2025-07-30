import numpy as np
import os
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

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

# Uso:
if __name__ == "__main__":
    ruta_imagenes = "../database_petals"
    ruta_mascaras = "../masks_petals"
    generar_conjuntos_multiples(ruta_imagenes, ruta_mascaras, n_conjuntos=3)
