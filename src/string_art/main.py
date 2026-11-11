import logging
import numpy as np

from string_art.image.preprocessing import load_image, calculate_density
from string_art.pins.initialization import generate_initial_points
from string_art.pins.stippling import lloyd_relaxation
from string_art.visualization import animate_stippling


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    image = load_image("images/dog.jpg")
    density = calculate_density(image, gamma=2.0)

    initial_points = generate_initial_points(density, num_points=5000)
    final_points, history = lloyd_relaxation(initial_points, density, iterations=15, relax_strength=0.5, return_history=True)

    anim = animate_stippling(image, history, interval=300)


if __name__ == "__main__":
    main()
