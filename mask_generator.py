"""Generate per-image vein masks from a CVAT XML export.

This script reads a CVAT-style ``ground_truth.xml`` file containing image
entries with ``polyline`` annotations, rasterizes each polyline onto a
black mask and saves the result as a PNG in ``output_mask_dir``. The
polylines are drawn as white lines (value 255) with configurable thickness.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image


xml_path = '../ground_truth.xml'
image_dir = '../database_petals'
output_mask_dir = '../masks_petals'

os.makedirs(output_mask_dir, exist_ok=True)

tree = ET.parse(xml_path)
root = tree.getroot()

for image_tag in root.findall('.//image'):
    image_name = image_tag.attrib['name']
    width = int(image_tag.attrib['width'])
    height = int(image_tag.attrib['height'])

    mask = np.zeros((height, width), dtype=np.uint8)

    for polyline in image_tag.findall('polyline'):
        points = polyline.attrib['points']
        point_list = np.array([
            [int(float(x)), int(float(y))]
            for x, y in (pair.split(',') for pair in points.split(';'))
        ], dtype=np.int32)

        cv2.polylines(mask, [point_list], isClosed=False, color=255, thickness=2)

    mask_filename = image_name.replace('.tif', '.png')
    mask_path = os.path.join(output_mask_dir, mask_filename)
    Image.fromarray(mask).save(mask_path)

    print(f"Guardada máscara para {image_name} en {mask_path}")
