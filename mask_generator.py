import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image

# Rutas
xml_path = '../ground_truth.xml'  # o el path al archivo que desees
image_dir = '../database_petals'
output_mask_dir = '../masks_petals'

# Crear carpeta de salida si no existe
os.makedirs(output_mask_dir, exist_ok=True)

# Parsear el XML de CVAT
tree = ET.parse(xml_path)
root = tree.getroot()

for image_tag in root.findall('.//image'):
    image_name = image_tag.attrib['name']  # ejemplo: 'rf11-2p.tif'
    width = int(image_tag.attrib['width'])
    height = int(image_tag.attrib['height'])

    # Crear imagen de máscara vacía (blanco y negro)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Añadir polígonos (venas) como líneas
    for polyline in image_tag.findall('polyline'):
        points = polyline.attrib['points']
        point_list = np.array([
            [int(float(x)), int(float(y))]
            for x, y in (pair.split(',') for pair in points.split(';'))
        ], dtype=np.int32)

        # Dibuja línea blanca (valor 255)
        cv2.polylines(mask, [point_list], isClosed=False, color=255, thickness=2)

    # Guardar máscara en formato PNG
    mask_filename = image_name.replace('.tif', '.png')
    mask_path = os.path.join(output_mask_dir, mask_filename)
    Image.fromarray(mask).save(mask_path)

    print(f"Guardada máscara para {image_name} en {mask_path}")
