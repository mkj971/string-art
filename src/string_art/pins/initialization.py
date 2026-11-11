import logging
logger = logging.getLogger(__name__)

import numpy as np


def generate_initial_points(density_map, num_points):

    if num_points <= 0:
        logger.warning("Number of points requested is non-positive.")
        raise ValueError("Number of points must be a positive integer.")
    
    if np.sum(density_map) <= 0:
        logger.warning("Density map has no positive values.")
        raise ValueError("Density map must contain positive values to generate points.")

    p_map = density_map / np.sum(density_map)

    if num_points > np.count_nonzero(p_map):
        logger.warning("Number of points requested exceeds the number of selectable pixels.")
        raise ValueError("Number of points exceeds the number of selectable pixels.")
    
    p_map = p_map.ravel()

    rng = np.random.default_rng()
    indices = rng.choice(
        a=p_map.size,
        size=num_points,
        replace=False,
        p=p_map
    )

    y, x = np.unravel_index(indices, density_map.shape)
    logger.info("Generated %d initial points based on the density map.", num_points)
    return np.column_stack((x, y)).astype(np.float64)

