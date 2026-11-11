import logging
logger = logging.getLogger(__name__)

import cv2
import numpy as np


def load_image(path):
    # Load image as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Image not found at path: {path}")

    normalized = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    logger.info("Image loaded successfully from path: %s", path)
    return normalized

def calculate_density(image, gamma=1.0):
    # Calculate the density of the image
    density = 1.0 - (image / 255.0)
    return density ** gamma