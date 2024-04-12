import numpy as np
import cv2
from skimage.feature import canny
from scipy import ndimage as ndi

def _circle_mask(num_col, radius, xcenter=None, ycenter=None):
    """
    Generates a boolean mask for a circle given its radius and optional center coordinates.
    If the center is not specified, it defaults to the center of the array.

    Parameters:
    - num_col (int): Size of the square array to generate the mask for.
    - radius (float): Radius of the circle.
    - xcenter (float, optional): X-coordinate of the circle's center. Defaults to the center of the array.
    - ycenter (float, optional): Y-coordinate of the circle's center. Defaults to the center of the array.

    Returns:
    - numpy.ndarray: A boolean array where True values represent points inside the circle.
    """
    if xcenter == None:
        xcenter = (num_col - 1) / 2.0
    if ycenter == None:
        ycenter = (num_col - 1) / 2.0
    x = np.linspace(0, num_col - 1, num_col) - xcenter
    y = np.linspace(0, num_col - 1, num_col) - ycenter
    X, Y = np.meshgrid(x, y)
    mask = X ** 2 + Y ** 2 < radius ** 2
    return mask

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
    theta_list = np.linspace(-np.pi, np.pi, num_circles, endpoint=False)
    for theta in theta_list:

        # Calculate the center coordinates of each circle
        center_x = side_length / 2 + outer_cir_radius * np.cos(theta)
        center_y = side_length / 2 + outer_cir_radius * np.sin(theta)

        # Generate a mask for the current circle
        # y, x = np.ogrid[:num_pixels, :num_pixels]
        # mask = (x * pixel_size - center_x) ** 2 + (y * pixel_size - center_y) ** 2 <= inner_cir_radius ** 2
        mask = _circle_mask(num_pixels, inner_cir_radius/pixel_size, center_x/pixel_size, center_y/pixel_size)

        # Append the circle mask to the list
        circle_masks.append(mask)

    return circle_masks




def detect_hough_circles(phantom, radius_range=None, vmin=0, vmax=None, min_dist=100, HoughCircles_params1=300, HoughCircles_params2=1):
    """Detects circles in an image using the Hough Circle Transform.

    Args:
        phantom (numpy.ndarray): The 2D image to detect circles in.
        radius_range (list of int, optional): The minimum and maximum radius of
            circles to detect. Defaults to a range based on the image size.
        vmin (int, optional): Minimum value for clipping the image before
            detection. Defaults to 0.
        vmax (int, optional): Maximum value for clipping the image before
            detection. If None, the 90th percentile of the image is used.
            Defaults to None.
        min_dist (int, optional): Minimum distance between the centers of the
            detected circles. If too small, multiple neighbor circles may be
            falsely detected in addition to a true one. If too large, some
            circles may be missed. Defaults to 100.

    Returns:
        numpy.ndarray: An array of detected circles, each represented by the
            center coordinates (x, y) and radius. Returns an empty array if no
            circles are detected.
    """
    # Set the default radius range based on the image size if not provided
    if radius_range is None:
        radius_range = [phantom.shape[0] // 50, 2 * phantom.shape[0] // 5]

    # Set the default maximum value for clipping if not provided
    if vmax is None:
        vmax = np.quantile(phantom, 0.9)

    # Clip and normalize the image
    tg_phan = np.clip(phantom, vmin, vmax)
    tg_img = np.uint8(255 * (tg_phan - vmin) / (vmax - vmin))

    # Detect circles using the Hough Circle Transform
    detected_circles = cv2.HoughCircles(tg_img, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
                                        param1=HoughCircles_params1, param2=HoughCircles_params2, minRadius=radius_range[0], maxRadius=radius_range[1])

    return detected_circles[0]


def segment_object(phantom, vmin, vmax, canny_sigma, roi_radius=None, bbox=None):
    """Segments an object within a given region of interest (ROI) or bounding box in an image.

    This function creates a segmentation mask for an object in an image. The image values
    are clipped and normalized based on provided minimum and maximum values. Canny edge
    detection is then applied to the normalized image. The edges are filled to create a
    binary mask that segments the object.

    Args:
        phantom (np.array): The input image to segment.
        vmin (float): The minimum value for clipping the image.
        vmax (float): The maximum value for clipping the image.
        canny_sigma (float): The standard deviation for the Gaussian filter used in
                             Canny edge detection.
        roi_radius (int, optional): The radius of the circular region of interest. If not
                                    provided, it defaults to half the image size.
        bbox (tuple of int, optional): The bounding box within which to perform segmentation,
                                       specified as (r_min, c_min, r_max, c_max). If not
                                       provided, the entire image is used.

    Returns:
        np.array: A binary segmentation mask of the object.
    """
    # Set default ROI radius to half the image size if not provided
    if roi_radius is None:
        roi_radius = phantom.shape[0] // 2
    mask = _circle_mask(phantom.shape[0], roi_radius, xcenter=phantom.shape[0] // 2, ycenter=phantom.shape[0] // 2)

    # Apply bounding box if provided
    if bbox is not None:
        r_min, c_min, r_max, c_max = bbox
        mask_bbox = np.zeros(mask.shape, dtype=bool)
        mask_bbox[r_min:r_max, c_min:c_max] = True
        mask = mask & mask_bbox

    # Clip and normalize the phantom values
    tg = mask * np.clip(phantom - vmin, 0, vmax - vmin) / (vmax - vmin) * 255

    # Perform Canny edge detection
    edges = canny(tg, sigma=canny_sigma)

    # Fill in the edges to create a mask
    filled_edges = ndi.binary_fill_holes(edges)

    # Convert boolean mask to an integer type
    segmentation_mask = filled_edges.astype(np.uint8)

    return segmentation_mask