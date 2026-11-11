import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_image(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()


def plot_points(image, points):
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.scatter(points[:, 0], points[:, 1], s=3, c="red")
    plt.axis("off")
    plt.show()


def plot_voronoi(vor, image=None):
    fig, ax = plt.subplots()

    if image is not None:
        ax.imshow(image, cmap="gray", alpha=0.5)

    for region in vor.regions:
        if not region or -1 in region:
            continue

        polygon = vor.vertices[region]
        polygon = np.vstack((polygon, polygon[0]))

        ax.plot(
            polygon[:, 0],
            polygon[:, 1],
            color="black",
            linewidth=0.5
        )

    ax.scatter(
        vor.points[:, 0],
        vor.points[:, 1],
        s=5,
        c="red",
        zorder=3
    )

    ax.set_aspect("equal")
    ax.invert_yaxis() if image is None else None
    ax.set_title("Bounded Voronoi Diagram")
    logger.info("Voronoi diagram plotted successfully.")
    plt.show()



def plot_comparison(image, initial_points, final_points):

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(image, cmap="gray")
    axes[0].scatter(initial_points[:, 0], initial_points[:, 1], s=3, c="red")
    axes[0].set_title("Initial Points")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].scatter(final_points[:, 0], final_points[:, 1], s=3, c="red")
    axes[1].set_title("After Lloyd Relaxation")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def animate_stippling(image, history, interval=300):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    height, width = image.shape

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Points on Image")

    axes[1].set_facecolor("white")
    axes[1].set_title("Voronoi Stippling")

    scatters = []

    for ax, color in zip(axes, ["red", "black"]):
        scatter = ax.scatter(history[0][:, 0], history[0][:, 1], s=4, c=color)
        scatters.append(scatter)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal")
        ax.axis("off")

    def update(frame):
        for scatter in scatters:
            scatter.set_offsets(history[frame])

        fig.suptitle(f"Lloyd Relaxation - Iteration {frame}/{len(history)-1}")

        return scatters

    anim = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval,
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show()

    return anim
