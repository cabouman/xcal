"""Internal rod segmentation.

The approach is Wenrui Li's from xcal 1 (phantom.py at tag v0.1.0),
with the hand-chosen constants replaced by values derived from the
declared rod diameters and the data:

1. Find the rod circles with the Hough transform.
2. Match circles to the declared rods by attenuation and diameter.
3. Segment each rod's ACTUAL shape inside its own window: clip the
   window to an automatically chosen value range, run Canny edge
   detection, and fill the closed edges (Wenrui's segment_object).
   The clip range comes from Otsu's threshold inside the window.
4. Validate against the declared geometry and fail with the numbers
   in the message rather than return a silent wrong answer.

The masks are the measured shapes.  The declared diameters only size
the search, seed the matching, and validate the result.
"""

import numpy as np
from scipy import ndimage


def _otsu(values):
    """Otsu's threshold for a 1D array of values."""
    hist, edges = np.histogram(values, bins=128)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(float)
    total = w.sum()
    best_t, best_score = centers[0], -1.0
    csum = np.cumsum(w)
    cmean = np.cumsum(w * centers)
    for i in range(1, len(centers)):
        w0 = csum[i - 1]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = cmean[i - 1] / w0
        m1 = (cmean[-1] - cmean[i - 1]) / w1
        score = w0 * w1 * (m0 - m1) ** 2
        if score > best_score:
            best_score = score
            best_t = centers[i]
    return best_t


def _detect_circles(image, radius_range, min_dist):
    """Wenrui's Hough circle detection, with the value range chosen
    automatically: the split between background and rods comes from
    Otsu on the whole image, so the normalization does not amplify
    background noise.  Returns an array of (x, y, radius) or None."""
    import cv2
    t = _otsu(image.ravel())
    background = image[image < t]
    foreground = image[image >= t]
    if background.size == 0 or foreground.size == 0:
        return None
    vmin = float(background.mean())
    vmax = float(np.quantile(foreground, 0.95))
    if vmax <= vmin:
        return None
    img8 = np.uint8(255 * np.clip(image - vmin, 0, vmax - vmin)
                    / (vmax - vmin))
    circles = cv2.HoughCircles(
        img8, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist,
        param1=100, param2=10,
        minRadius=int(radius_range[0]), maxRadius=int(radius_range[1]))
    return None if circles is None else circles[0]


def _segment_window(image, center, half_width, canny_sigma):
    """Wenrui's segment_object on one rod's window: clip to an Otsu
    derived range, Canny, fill.  Returns a full-size boolean mask of
    the component containing the center."""
    from skimage.feature import canny
    rows, cols = image.shape
    cy, cx = int(round(center[0])), int(round(center[1]))
    r0, r1 = max(0, cy - half_width), min(rows, cy + half_width + 1)
    c0, c1 = max(0, cx - half_width), min(cols, cx + half_width + 1)
    window = image[r0:r1, c0:c1]

    # The clip range for edge detection: between the window's two
    # populations (background and rod), found by Otsu.
    t = _otsu(window.ravel())
    background = window[window < t]
    foreground = window[window >= t]
    if background.size == 0 or foreground.size == 0:
        return None
    vmin = float(background.mean())
    vmax = float(foreground.mean())
    if vmax <= vmin:
        return None

    normalized = np.clip(window - vmin, 0, vmax - vmin) / (vmax - vmin)
    edges = canny(normalized * 255, sigma=canny_sigma)
    filled = ndimage.binary_fill_holes(edges)

    labels, n = ndimage.label(filled)
    if n == 0:
        return None
    label_at_center = labels[cy - r0, cx - c0]
    if label_at_center == 0:
        sizes = ndimage.sum(filled, labels, range(1, n + 1))
        label_at_center = int(np.argmax(sizes)) + 1
    mask = np.zeros(image.shape, dtype=bool)
    mask[r0:r1, c0:c1] = (labels == label_at_center)
    return mask


def segment_rods(recon, rods, mm_per_voxel, energies, verbose=1):
    """Segment the rods in a reconstruction.

    Args:
        recon (numpy.ndarray): Volume with shape (rows, cols, slices),
            in 1/mm.
        rods (list of Rod): The rods expected in this scan.
        mm_per_voxel (float): Voxel size in mm.
        energies (numpy.ndarray): Fit energy grid in keV, used to rank
            the rods by expected attenuation for matching.
        verbose (int): 1 prints what was found.

    Returns:
        tuple: (labels, masks).  labels is a uint8 volume, 0 for
        background and k+1 for the k-th rod.  masks is a list of
        float32 volumes holding each rod's measured shape.
    """
    from . import _physics
    rows, cols, n_slices = recon.shape

    # Average the central half of the slices into one 2D image.
    lo = n_slices // 4
    hi = max(lo + 1, (3 * n_slices) // 4)
    image = np.nanmean(recon[:, :, lo:hi], axis=2)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    radii_vox = [0.5 * r.diameter / mm_per_voxel for r in rods]
    if min(radii_vox) < 3:
        raise ValueError(
            f"the smallest rod is only {min(radii_vox):.1f} voxels in "
            f"radius at {mm_per_voxel:.4g} mm per voxel; the "
            f"reconstruction is too coarse to segment it.")

    # Step 1: find the rod circles.  Rods can differ in brightness by
    # more than a factor of ten, and one normalization cannot show
    # them all to the detector at once.  So detection repeats: find
    # circles, blank them to the background level, renormalize the
    # remainder, and detect again, until every declared rod has a
    # circle or a round finds nothing new.
    radius_range = (0.6 * min(radii_vox), 1.5 * max(radii_vox))
    min_dist = max(2.0 * min(radii_vox), 8)
    remaining = image.copy()
    yy, xx = np.ogrid[:rows, :cols]
    circles = []
    # The background noise level, measured robustly, guards the later
    # rounds: once every rod is blanked, only noise remains, and no
    # candidate circle may pass the contrast test.
    noise = 1.4826 * np.median(np.abs(image - np.median(image)))
    for _ in range(len(rods)):
        found = _detect_circles(remaining, radius_range, min_dist)
        if found is None or len(found) == 0:
            break
        background_level = float(np.median(remaining))
        accepted = False
        for x, y, r in found:
            if any((x - c[0]) ** 2 + (y - c[1]) ** 2 < min_dist ** 2
                   for c in circles):
                continue
            inside = ((yy - y) ** 2 + (xx - x) ** 2) <= (0.7 * r) ** 2
            if not inside.any():
                continue
            if remaining[inside].mean() < background_level + 4 * noise:
                continue
            circles.append((x, y, r))
            remaining[(yy - y) ** 2 + (xx - x) ** 2
                      <= (1.3 * r) ** 2] = background_level
            accepted = True
            break
        if not accepted:
            break
    if len(circles) < len(rods):
        raise ValueError(
            f"Segmentation failed: circle detection found "
            f"{len(circles)} circle(s) but {len(rods)} rod(s) were "
            f"declared, with radii searched between "
            f"{radius_range[0]:.0f} and {radius_range[1]:.0f} voxels.  "
            f"Check the declared diameters and the voxel size "
            f"({mm_per_voxel:.4g} mm/voxel).")
    circles = np.array(circles[:max(len(rods) * 2, len(rods))])

    # Step 2: match circles to rods by attenuation and diameter.
    # Both sides are normalized by their geometric means, because the
    # measured effective attenuation differs from the expected value
    # by a common spectrum-dependent scale.
    yy, xx = np.ogrid[:rows, :cols]
    stats = []
    for x, y, r in circles:
        inside = ((yy - y) ** 2 + (xx - x) ** 2) <= (0.8 * r) ** 2
        stats.append((float(image[inside].mean()) if inside.any()
                      else 0.0, 2 * r * mm_per_voxel))
    measured_mu = np.array([max(s[0], 1e-9) for s in stats])
    measured_diam = np.array([s[1] for s in stats])
    band = energies[(energies >= energies[len(energies) // 3])
                    & (energies <= energies[(2 * len(energies)) // 3])]
    expected_mu = np.array([float(np.mean(
        _physics.attenuation_coefficients(r.material, band)))
        for r in rods])
    m_rel = np.log(measured_mu) - np.mean(np.log(measured_mu))
    e_rel = np.log(expected_mu) - np.mean(np.log(expected_mu))
    cost = np.zeros((len(rods), len(circles)))
    for i, rod in enumerate(rods):
        for j in range(len(circles)):
            cost[i, j] = (abs(m_rel[j] - e_rel[i])
                          + 3.0 * abs(np.log(measured_diam[j]
                                             / rod.diameter)))
    from scipy.optimize import linear_sum_assignment
    rod_idx, circle_idx = linear_sum_assignment(cost)

    # Step 3: segment each rod's actual shape in its own window.
    labels3d = np.zeros(recon.shape, dtype=np.uint8)
    masks = [None] * len(rods)
    for i, j in zip(rod_idx, circle_idx):
        rod = rods[i]
        x, y, r_hough = circles[j]
        half_width = int(round(1.8 * max(r_hough, radii_vox[i])))
        sigma = max(2.0, radii_vox[i] / 50.0)
        shape2d = _segment_window(image, (y, x), half_width, sigma)
        if shape2d is None or not shape2d.any():
            raise ValueError(
                f"Segmentation failed: no closed shape was found for "
                f"the {rod.material.name} rod near "
                f"(row {y:.0f}, column {x:.0f}).")

        # Step 4: validate the measured shape against the declaration.
        diam = 2 * mm_per_voxel * np.sqrt(shape2d.sum() / np.pi)
        if not 0.6 * rod.diameter <= diam <= 1.6 * rod.diameter:
            raise ValueError(
                f"Segmentation failed: the shape matched to the "
                f"{rod.material.name} rod measures {diam:.3g} mm "
                f"across, but the declared diameter is "
                f"{rod.diameter:.3g} mm.  Check the declared diameters "
                f"and the voxel size ({mm_per_voxel:.4g} mm/voxel).")

        mask = np.zeros(recon.shape, dtype=np.float32)
        mask[:, :, lo:hi] = shape2d[:, :, None].astype(np.float32)
        masks[i] = mask
        labels3d[shape2d, lo:hi] = i + 1
        if verbose:
            print(f"xcal:   {rod.material.name} rod: measured "
                  f"{diam:.3g} mm (declared {rod.diameter:.3g}), mean "
                  f"LAC {measured_mu[j]:.4g} 1/mm")

    # Any rod outside the projector's circular region of
    # reconstruction would be silently truncated by the forward
    # projection.
    ror = ((yy - (rows - 1) / 2) ** 2 / ((rows / 2 - 1) ** 2)
           + (xx - (cols - 1) / 2) ** 2 / ((cols / 2 - 1) ** 2)) <= 1.0
    outside = (labels3d[:, :, (lo + hi) // 2] > 0) & ~ror
    if outside.any():
        raise ValueError(
            "Segmentation failed: a rod extends outside the circular "
            "region of reconstruction, so its forward projection would "
            "be silently truncated.  Enlarge the reconstruction with "
            "ct_model.scale_recon_shape(...) and recalibrate.")
    return labels3d, masks
