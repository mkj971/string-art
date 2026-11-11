import logging
logger = logging.getLogger(__name__)

import numpy as np
import cv2

from string_art.pins.voronoi import calculate_voronoi

def calculate_weighted_centroid(polygon, density_map):
    mask = np.zeros(density_map.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], color=1)

    weights = density_map[mask == 1]
    ys, xs = np.where(mask == 1)

    total_weight = np.sum(weights)
    if total_weight <= 0:
        logger.debug("Voronoi cell has zero density. No centroid calculated.")
        return None
    
    cx = np.sum(xs * weights) / total_weight
    cy = np.sum(ys * weights) / total_weight
    logger.debug("Weighted centroid calculated: (%.2f, %.2f)", cx, cy)
    
    return np.array([cx, cy])


def lloyd_iteration(points, density_map, relax_strength=0.5):
    voronoi_cells = calculate_voronoi(points, density_map.shape)

    new_points = points.astype(np.float64, copy=True)

    for i, point in enumerate(points):
        region = voronoi_cells.regions[i]
        polygon = voronoi_cells.vertices[region]
        centroid = calculate_weighted_centroid(polygon, density_map)
        if centroid is not None:
            new_point = point + relax_strength * (centroid - point)
            new_points[i] = new_point

    distances = np.linalg.norm(new_points - points, axis=1)
    logger.debug(
        "Lloyd iteration completed. Average movement: %.4f px, "
        "Maximum movement: %.4f px",
        np.mean(distances),
        np.max(distances)
    )

    return new_points

def lloyd_relaxation(points, density_map, iterations=30, relax_strength=0.5,  return_history=False):

    points = points.astype(np.float64, copy=True)
    history = [points.copy()] if return_history else None

    for iteration in range(iterations):
        points = lloyd_iteration(points, density_map, relax_strength)

        if return_history:
            history.append(points.copy())

        logger.info("Lloyd relaxation iteration %d/%d", iteration + 1, iterations)

    if return_history:
        return points, history
    return points