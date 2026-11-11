import logging
logger = logging.getLogger(__name__)

from boundvor import BoundedVoronoi
import numpy as np

def _calc_bounding_box(image_shape):
    height, width = image_shape
    return np.array([
        [0.0, 0.0],
        [width, 0.0],
        [width, height],
        [0.0, height]
    ])

def calculate_voronoi(points, image_shape):
    bounding_box = _calc_bounding_box(image_shape)
    vor = BoundedVoronoi(points, bounds=bounding_box)
    logger.info("Calculated Voronoi diagram with %d points.", len(points))
    return vor