import numpy as np
import os
from PIL import Image
import glob

def cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras, tamaño=(128, 128)):
    """
    Carga imágenes (TIF) y máscaras (PNG) desde directorios
    """
    # Obtener listas de archivos
    imagenes = sorted(glob.glob(os.path.join(ruta_imagenes, "*.tif")))
    mascaras = sorted(glob.glob(os.path.join(ruta_mascaras, "*.png")))
    
    if len(imagenes) != len(mascaras):
        raise ValueError(f"Número de imágenes ({len(imagenes)}) y máscaras ({len(mascaras)}) no coincide")

    X = []
    y = []

    for img_path, mask_path in zip(imagenes, mascaras):
        # Cargar imagen TIF
        img = Image.open(img_path).convert('RGB')
        img = img.resize(tamaño)
        img_array = np.array(img)

        # Cargar máscara PNG (asegurando que sea binaria)
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(tamaño)
        mask_array = np.array(mask)
        
        # Normalizar máscara a 0-1 si es necesario
        if mask_array.max() > 1:
            mask_array = (mask_array > 0).astype(np.uint8)

        X.append(img_array)
        y.append(mask_array)

    return np.array(X), np.expand_dims(np.array(y), axis=-1)

def generar_conjuntos_segmentacion(ruta_imagenes, ruta_mascaras):
    """
    Genera y guarda los conjuntos de datos
    """
    X, y = cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras)
    
    # Dividir los datos (aquí puedes implementar tu lógica de división)
    # Ejemplo simple: 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Guardar los conjuntos
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)
    np.save("X_val.npy", X_val)
    np.save("y_val.npy", y_val)
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

# Uso:
ruta_imagenes = "../database_petals"  # Directorio con imágenes .tif
ruta_mascaras = "../masks_petals"       # Directorio con máscaras .png

generar_conjuntos_segmentacion(ruta_imagenes, ruta_mascaras)
