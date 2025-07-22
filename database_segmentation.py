import numpy as np
import os
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

def cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras, tamaño=(128, 128)):
    """Carga imágenes y máscaras desde directorios"""
    imagenes = sorted(glob.glob(os.path.join(ruta_imagenes, "*.tif")))
    mascaras = sorted(glob.glob(os.path.join(ruta_mascaras, "*.png")))
    
    if len(imagenes) != len(mascaras):
        raise ValueError(f"Número de imágenes ({len(imagenes)}) y máscaras ({len(mascaras)}) no coincide")

    X = []
    y = []

    for img_path, mask_path in zip(imagenes, mascaras):
        # Cargar imagen
        img = Image.open(img_path).convert('RGB')
        img = img.resize(tamaño)
        img_array = np.array(img)

        # Cargar máscara
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(tamaño)
        mask_array = np.array(mask)
        
        # Binarizar máscara si es necesario
        if mask_array.max() > 1:
            mask_array = (mask_array > 0).astype(np.uint8)

        X.append(img_array)
        y.append(mask_array)

    return np.array(X), np.expand_dims(np.array(y), axis=-1)

def generar_conjuntos_multiples(ruta_imagenes, ruta_mascaras, n_conjuntos=3):
    """Genera n conjuntos diferentes de datos divididos aleatoriamente"""
    X, y = cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras)
    
    for i in range(n_conjuntos):
        random_state = 42 + i  
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state
        )
        
        # Guardar cada conjunto individualmente
        np.savez(
            f"conjunto_{i+1}.npz",
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )
    
    print(f"Se generaron {n_conjuntos} conjuntos de datos (.npz)")

# Uso:
if __name__ == "__main__":
    ruta_imagenes = "../database_petals"
    ruta_mascaras = "../masks_petals"
    generar_conjuntos_multiples(ruta_imagenes, ruta_mascaras, n_conjuntos=3)
