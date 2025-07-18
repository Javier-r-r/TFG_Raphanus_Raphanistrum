import numpy as np
from sklearn.model_selection import train_test_split
import os
from PIL import Image

def cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras, tamaño=(128, 128)):
    """
    Carga imágenes y sus máscaras, redimensionándolas.
    
    Parámetros:
    - ruta_imagenes: Lista de rutas a las imágenes de entrada.
    - ruta_mascaras: Lista de rutas a las máscaras (debe coincidir con ruta_imagenes).
    - tamaño: Tupla (alto, ancho) para redimensionar.
    
    Retorna:
    - X: Array de imágenes (num_imagenes, alto, ancho, canales).
    - y: Array de máscaras (num_imagenes, alto, ancho, 1).
    """
    X = []
    y = []
    
    for img_path, mask_path in zip(ruta_imagenes, ruta_mascaras):
        # Cargar y redimensionar imagen
        img = Image.open(img_path)
        img = img.resize(tamaño)
        img_array = np.array(img)
        
        # Cargar y redimensionar máscara (convertir a escala de grises si es necesario)
        mask = Image.open(mask_path).convert('L')  # 'L' = 1 canal (blanco y negro)
        mask = mask.resize(tamaño)
        mask_array = np.array(mask)
        
        X.append(img_array)
        y.append(mask_array)
    
    return np.array(X), np.expand_dims(np.array(y), axis=-1)  # Añadir dimensión extra a las máscaras (para canales=1)

def generar_conjuntos_segmentacion(ruta_imagenes, ruta_mascaras, n=3, semilla=42, test_size=0.2, val_size=0.25):
    """
    Genera n conjuntos train/val/test para segmentación de imágenes.
    
    Parámetros:
    - ruta_imagenes: Lista de rutas a imágenes de entrada.
    - ruta_mascaras: Lista de rutas a máscaras (debe ser 1:1 con ruta_imagenes).
    - n: Número de repeticiones.
    - semilla: Semilla para reproducibilidad.
    """
    conjuntos = []
    X, y = cargar_imagenes_y_mascaras(ruta_imagenes, ruta_mascaras)
    
    for i in range(n):
        semilla_actual = semilla + i
        # Dividir en train+test y luego train en train+val
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=semilla_actual
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=semilla_actual
        )
        conjuntos.append((X_train, X_val, X_test, y_train, y_val, y_test))
    
    np.save("conjuntos_segmentacion.npy", conjuntos)  # Guarda los 3 conjuntos en un archivo

generar_conjuntos_segmentacion("", "")