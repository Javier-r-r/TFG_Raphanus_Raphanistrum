import cv2
import numpy as np

def generate_petal_mask_from_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """
    Generate a petal mask from an RGB image using HSV color thresholding.
    Returns a binary mask (uint8, 0/1).
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([40, 255, 255])
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    petal_mask = cv2.bitwise_or(mask_yellow, mask_green)
    petal_mask = cv2.morphologyEx(petal_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    petal_mask = (petal_mask > 0).astype(np.uint8)
    return petal_mask