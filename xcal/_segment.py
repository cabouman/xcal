"""Internal rod segmentation.

The reconstruction of the calibration object holds K rods of known
count and approximate diameter.  The algorithm finds them without any
per-scan tuning:

1. Average the central slices into one 2D image.
2. Convolve with a disk of the nominal rod radius (a matched filter).
3. Greedily take the K strongest peaks, masking a disk around each
   peak after taking it, so weak rods are found after strong ones.
4. Threshold each rod inside its own local window with Otsu's rule,
   so rods of very different attenuation never share a threshold.
5. Match the found blobs to the declared rods by attenuation and
   diameter, erode one voxel against partial-volume bias, and extrude
   along the rod axis.
6. Validate against the declared geometry, and fail with the numbers
   in the message instead of returning a silent wrong answer.

Only numpy and scipy.ndimage are used.
"""

import numpy as np
from scipy import ndimage


def _disk(radius_vox):
    r = max(int(round(radius_vox)), 1)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(float)


def _otsu(values):
    """Otsu's threshold for a 1D array of values."""
    hist, edges = np.histogram(values, bins=128)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(float)
    total = w.sum()
    best_t, best_score = centers[0], -1.0
    csum = np.cumsum(w)
    cmean = np.cumsum(w * centers)
    mean_all = cmean[-1] / total
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


def segment_rods(recon, rods, ct_model, energies, verbose=1):
    """Segment the rods in a reconstruction.

    Args:
        recon (numpy.ndarray): Volume with shape (rows, cols, slices),
            in 1/mm when the model's geometry is in mm.
        rods (list of Rod): The rods expected in this scan.
        ct_model: The mbirtorch model, used for the voxel size.
        energies (numpy.ndarray): Fit energy grid in keV, used to rank
            the rods by expected attenuation for matching.
        verbose (int): 1 prints what was found.

    Returns:
        numpy.ndarray: Label volume of recon's shape, 0 background,
        k+1 for the k-th rod of this scan.
    """
    from . import _physics
    mm_per_voxel = float(ct_model.get_params('delta_voxel'))
    rows, cols, n_slices = recon.shape

    # Step 1: average the central half of the slices.
    lo = n_slices // 4
    hi = max(lo + 1, (3 * n_slices) // 4)
    image = np.nanmean(recon[:, :, lo:hi], axis=2)
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    radii_vox = [0.5 * r.diameter / mm_per_voxel for r in rods]
    mean_radius = float(np.mean(radii_vox))
    if mean_radius < 1.5:
        raise ValueError(
            f"the rods are only about {mean_radius:.1f} voxels in "
            f"radius at {mm_per_voxel:.4g} mm per voxel; the "
            f"reconstruction is too coarse to segment them.")

    # Step 2: matched filter with a disk of the mean rod radius.
    kernel = _disk(mean_radius)
    kernel /= kernel.sum()
    filtered = ndimage.convolve(image, kernel, mode='constant')

    # Step 3: greedy peak peel, one peak per rod.
    centers = []
    peeled = filtered.copy()
    for _ in rods:
        peak = np.unravel_index(np.argmax(peeled), peeled.shape)
        win = int(round(2 * max(radii_vox)))
        r0, r1 = max(0, peak[0] - win), min(rows, peak[0] + win + 1)
        c0, c1 = max(0, peak[1] - win), min(cols, peak[1] + win + 1)
        patch = filtered[r0:r1, c0:c1]
        strong = patch >= 0.5 * patch.max()
        yy, xx = np.mgrid[r0:r1, c0:c1]
        weight = np.where(strong, patch, 0.0)
        cy = float((yy * weight).sum() / weight.sum())
        cx = float((xx * weight).sum() / weight.sum())
        centers.append((cy, cx))
        yy2, xx2 = np.ogrid[:rows, :cols]
        peeled[(yy2 - cy) ** 2 + (xx2 - cx) ** 2
               <= (1.5 * max(radii_vox)) ** 2] = -np.inf

    # Step 4: per-rod Otsu threshold in a local window.
    blobs = []
    for cy, cx in centers:
        win = int(round(2.5 * max(radii_vox)))
        r0, r1 = max(0, int(cy) - win), min(rows, int(cy) + win + 1)
        c0, c1 = max(0, int(cx) - win), min(cols, int(cx) + win + 1)
        window = image[r0:r1, c0:c1]
        thr = _otsu(window.ravel())
        binary = window > thr
        labels, _ = ndimage.label(binary)
        center_label = labels[int(cy) - r0, int(cx) - c0]
        if center_label == 0:
            # The centroid fell on a background pixel; take the
            # largest component instead.
            sizes = ndimage.sum(binary, labels,
                                range(1, labels.max() + 1))
            center_label = int(np.argmax(sizes)) + 1
        blob = np.zeros_like(image, dtype=bool)
        blob[r0:r1, c0:c1] = (labels == center_label)
        blob = ndimage.binary_fill_holes(blob)
        blobs.append(blob)

    # Step 5: match blobs to rods by attenuation and diameter.
    measured_mu = [float(image[b].mean()) for b in blobs]
    measured_diam = [2 * mm_per_voxel * np.sqrt(b.sum() / np.pi)
                     for b in blobs]
    band = energies[(energies >= energies[len(energies) // 3])
                    & (energies <= energies[(2 * len(energies)) // 3])]
    expected_mu = [float(np.mean(
        _physics.attenuation_coefficients(r.material, band)))
        for r in rods]
    cost = np.zeros((len(rods), len(blobs)))
    for i, r in enumerate(rods):
        for j in range(len(blobs)):
            cost[i, j] = (abs(np.log(max(measured_mu[j], 1e-9)
                                     / expected_mu[i]))
                          + abs(np.log(measured_diam[j] / r.diameter)))
    from scipy.optimize import linear_sum_assignment
    rod_idx, blob_idx = linear_sum_assignment(cost)

    # Step 6: validate, erode, extrude.
    labels3d = np.zeros(recon.shape, dtype=np.uint8)
    for i, j in zip(rod_idx, blob_idx):
        rod = rods[i]
        diam = measured_diam[j]
        if not 0.5 * rod.diameter <= diam <= 1.7 * rod.diameter:
            raise ValueError(
                f"Segmentation failed: the blob matched to the "
                f"{rod.material.name} rod measures {diam:.3g} mm "
                f"across, but the declared diameter is "
                f"{rod.diameter:.3g} mm.  Check the declared diameters "
                f"and the model's pixel size "
                f"({mm_per_voxel:.4g} mm/voxel).")
        blob = ndimage.binary_erosion(blobs[j])
        if not blob.any():
            blob = blobs[j]
        labels3d[blob, lo:hi] = i + 1
        if verbose:
            print(f"xcal:   {rod.material.name} rod: measured "
                  f"{diam:.3g} mm (declared {rod.diameter:.3g}), mean "
                  f"LAC {measured_mu[j]:.4g} 1/mm")

    # Check the rods against the projector's region-of-reconstruction
    # mask: anything outside the inscribed circle is silently truncated
    # by the forward projection.
    yy2, xx2 = np.ogrid[:rows, :cols]
    ror = ((yy2 - (rows - 1) / 2) ** 2 / ((rows / 2 - 1) ** 2)
           + (xx2 - (cols - 1) / 2) ** 2 / ((cols / 2 - 1) ** 2)) <= 1.0
    outside = (labels3d[:, :, (lo + hi) // 2] > 0) & ~ror
    if outside.any():
        raise ValueError(
            "Segmentation failed: a rod extends outside the circular "
            "region of reconstruction, so its forward projection would "
            "be silently truncated.  Enlarge the reconstruction with "
            "ct_model.scale_recon_shape(...) and recalibrate.")
    return labels3d
