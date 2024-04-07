import numpy as np

def generate_circle_masks(side_length, pixel_size, num_circles, outer_cir_radius, inner_cir_radius):
    """
    Generate a set of small, non-overlapping circular masks with their centers evenly distributed
    along the circumference of a larger circle, all contained within a square canvas.

    Args:
        side_length (float): The side length of the square canvas that the circles will be placed on,
                             measured in millimeters (mm). This defines the workspace for the circle generation.
        pixel_size (float): The size of each pixel in the canvas, given in millimeters (mm), which determines
                            the resolution of the created masks.
        num_circles (int): The number of non-overlapping small circles to generate.
        outer_cir_radius (float): The radius of the larger circle on whose circumference the centers of the
                                  small circles will be placed, measured in millimeters (mm).
        inner_cir_radius (float): The radius of each small inner circle mask, measured in millimeters (mm).
                                  The small circles will have their centers on the larger circle's circumference
                                  and should not overlap with each other.

    Returns:
        A list containing the circle masks, where each mask is represented by a boolean NumPy array with True values indicating the presence of the circle mask and False values indicating absence.
    """

    # Initialize a list to store circle masks
    circle_masks = []
    num_pixels = int(side_length / pixel_size) + 1

    # Generate coordinates for non-overlapping circles
    for k in range(num_circles):
        # Calculate the angle for current circle
        theta = 2 * np.pi * k / num_circles

        # Calculate the center coordinates of each circle
        center_x = side_length / 2 + outer_cir_radius * np.cos(theta)
        center_y = side_length / 2 + outer_cir_radius * np.sin(theta)

        # Generate a mask for the current circle
        y, x = np.ogrid[:num_pixels, :num_pixels]
        mask = (x * pixel_size - center_x) ** 2 + (y * pixel_size - center_y) ** 2 <= inner_cir_radius ** 2

        # Append the circle mask to the list
        circle_masks.append(mask)

    return circle_masks