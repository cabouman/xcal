"""Shared demo utilities."""

import numpy as np
from scipy import ndimage


def segment_targets(recon, targets, mm_per_voxel, method='quantile'):
    """Segment the calibration targets in a reconstruction.

    Three methods.  'quantile': threshold at the image quantile of
    the declared total target area, take the N largest connected
    regions, fill holes, and pair regions with targets in order of
    increasing material density.  'otsu': 2-level Otsu threshold,
    largest connected region, holes filled; single target only.
    'disk': a disk of the declared diameter at the position where
    the image is brightest under it (a matched filter); single
    target only.

    Args:
        recon (numpy.ndarray): Volume with shape (rows, cols,
            slices).
        targets (list of Target): The declared targets.
        mm_per_voxel (float): Voxel size in mm.
        method (str): 'quantile', 'otsu', or 'disk'.

    Returns:
        list of numpy.ndarray: One float32 mask volume per target,
        in target order.
    """
    img = np.asarray(recon)[:, :, 0]
    if method != 'quantile' and len(targets) != 1:
        raise ValueError(f"method '{method}' segments a single "
                         f"target, got {len(targets)}.")

    if method == 'quantile':
        area = sum(np.pi * (0.5 * t.size / mm_per_voxel) ** 2
                   for t in targets)
        threshold = float(np.quantile(img, 1.0 - area / img.size))
        cc, n = ndimage.label(img >= threshold)
        if n < len(targets):
            raise ValueError(f"segmentation found {n} regions for "
                             f"{len(targets)} declared targets.")
        sizes = ndimage.sum(cc > 0, cc, range(1, n + 1))
        keep = np.argsort(sizes)[::-1][:len(targets)] + 1
        # Thin streak artifacts connected to a region survive the
        # threshold.  Fill the region solid, open with a small disk
        # to sever the streaks, and keep the largest piece.
        r_open = max(2, int(round(0.05 * min(
            0.5 * t.size / mm_per_voxel for t in targets))))
        yk, xk = np.ogrid[-r_open:r_open + 1, -r_open:r_open + 1]
        element = yk**2 + xk**2 <= r_open**2

        def clean(component):
            solid = ndimage.binary_fill_holes(component)
            opened = ndimage.binary_opening(solid, structure=element)
            pieces, m = ndimage.label(opened)
            if m > 1:
                counts = ndimage.sum(opened, pieces, range(1, m + 1))
                opened = pieces == int(np.argmax(counts)) + 1
            return ndimage.binary_fill_holes(opened)

        shapes = [clean(cc == k) for k in keep]
        shapes.sort(key=lambda s: float(img[s].mean()))
        order = np.argsort([t.material.density for t in targets])
    elif method == 'otsu':
        from mbirtorch.preprocess import multi_threshold_otsu
        threshold = multi_threshold_otsu(img, classes=2)[0]
        binary = img >= threshold
        cc, n = ndimage.label(binary)
        sizes = ndimage.sum(binary, cc, range(1, n + 1))
        shapes = [ndimage.binary_fill_holes(
            cc == int(np.argmax(sizes)) + 1)]
        order = [0]
    elif method == 'disk':
        radius = 0.5 * targets[0].size / mm_per_voxel
        r_k = int(round(radius))
        yk, xk = np.ogrid[-r_k:r_k + 1, -r_k:r_k + 1]
        kernel = (yk**2 + xk**2 <= r_k**2).astype(float)
        score = ndimage.convolve(img, kernel / kernel.sum(),
                                 mode='constant')
        cy, cx = np.unravel_index(np.argmax(score), score.shape)
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        shapes = [(yy - cy)**2 + (xx - cx)**2 <= radius**2]
        order = [0]
    else:
        raise ValueError(f"unknown segmentation method {method!r}.")

    masks = [None] * len(targets)
    for shape, ti in zip(shapes, order):
        diam = 2 * mm_per_voxel * np.sqrt(shape.sum() / np.pi)
        if not 0.6 <= diam / targets[ti].size <= 1.6:
            raise ValueError(
                f"segmentation failed: the region paired with "
                f"{targets[ti].material.name} measures {diam:.3g} mm "
                f"(declared {targets[ti].size:g} mm).")
        mask = np.zeros(np.asarray(recon).shape, np.float32)
        mask[:, :, :] = shape[:, :, None]
        masks[ti] = mask
    return masks


def simulate_scanner(gt_system, cal_target, voltage,
                     n_views,
                     n_det_rows, n_det_channels, pixel_mm, photons,
                     seed):
    """Stand in for the scanner and its preprocessing.

    With real data, mbirtorch preprocessing reads the scanner file
    and returns a sinogram and an mbirtorch CT model.  This function
    returns the same pair for a simulated scan, plus the ground
    truth (gt) masks, which only a simulation can know.
    """
    import mbirtorch
    import xcal
    angles = np.linspace(0, np.pi, n_views,
                         endpoint=False).astype(np.float32)
    ct_model = mbirtorch.ParallelBeamModel(
        (n_views, n_det_rows, n_det_channels), angles)
    ct_model.set_params(delta_det_channel=pixel_mm,
                        delta_det_row=pixel_mm,
                        alu_unit='mm', alu_value=1.0)
    ct_model.auto_set_recon_geometry()

    gt_masks = xcal.cylinder_masks(cal_target, ct_model)
    sino = xcal.simulate_scan(gt_system, cal_target, ct_model,
                              voltage=voltage, target_masks=gt_masks,
                              photons=photons, seed=seed)
    return sino, ct_model, gt_masks


def load_als_scan(path, center_offset_channels, ring_snr, snr_db,
                  pixel_mm, downsample):
    """Stand in for mbirtorch preprocessing of one ALS scan file.

    Reads the normalized transmission, removes ring artifacts, and
    returns the sinogram and the mbirtorch CT model with the given
    reconstruction parameters applied, downsampled for speed.

    Args:
        path (str): The HDF5 scan file.
        center_offset_channels (float): Detector center offset in
            original channels.
        ring_snr (float): Stripe detection threshold of
            remove_all_stripe.
        snr_db (float): mbirtorch regularization parameter.
        pixel_mm (float): Detector pixel pitch in mm.
        downsample (int): Channel and view downsampling factor.

    Returns:
        tuple: (sinogram, ct_model).
    """
    import h5py
    import mbirtorch
    import mbirtorch.preprocess as mtp
    with h5py.File(path, 'r') as f:
        trans = f['data_norm'][()]          # (views, 1, channels)

    # Average transmission over channel blocks; subsample views.
    n_views, _, n_chan = trans.shape
    n_chan -= n_chan % downsample
    trans = trans[:, :, :n_chan].reshape(n_views, 1, -1, downsample)
    trans = trans.mean(axis=3)[::downsample]
    sino = -np.log(np.clip(trans, 1e-6, None))

    # The detector's fixed per-channel gain error (about 6%)
    # reconstructs as ring artifacts.  Remove it in the sinogram.
    sino = mtp.remove_all_stripe(sino, snr=ring_snr)
    sino = sino.astype(np.float32)

    angles = -np.linspace(-0.5 * np.pi, 1.5 * np.pi, n_views,
                          endpoint=True)[::downsample]
    ct_model = mbirtorch.ParallelBeamModel(sino.shape,
                                           angles.astype(np.float32))
    ct_model.set_params(delta_det_channel=pixel_mm * downsample,
                        delta_det_row=pixel_mm * downsample,
                        det_channel_offset=center_offset_channels
                        * pixel_mm,
                        snr_db=snr_db,
                        alu_unit='mm', alu_value=1.0)
    ct_model.auto_set_recon_geometry()
    return sino, ct_model


def save_segmentation_plot(recon, targets, masks, filename,
                           title=None):
    """Write a review image of a segmentation.

    The center slice of the reconstruction, with a colorbar, and
    each target's mask outlined in its own color.  A legend outside
    the image names each color's material and size.

    Args:
        recon (numpy.ndarray): Reconstructed volume with shape
            (rows, cols, slices).
        targets (list of Target): The targets, in mask order.
        masks (list of numpy.ndarray): One mask volume per target.
        filename (str): Output image path.
        title (str, optional): Title above the image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = ['red', 'cyan', 'orange', 'magenta', 'lime', 'yellow']
    recon = np.asarray(recon)
    s = recon.shape[2] // 2
    fig, ax = plt.subplots(figsize=(8.0, 6))
    im = ax.imshow(recon[:, :, s], origin='lower')
    fig.colorbar(im, ax=ax, fraction=0.046)
    handles = []
    for i, (tg, m) in enumerate(zip(targets, masks)):
        c = colors[i % len(colors)]
        ax.contour(np.asarray(m)[:, :, s], levels=[0.5], colors=[c],
                   linewidths=0.9)
        handles.append(Line2D([0], [0], color=c,
                              label=f'{tg.material.name} '
                                    f'({tg.size:g} mm)'))
    ax.legend(handles=handles, loc='center left',
              bbox_to_anchor=(1.15, 0.5))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
