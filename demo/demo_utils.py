"""Shared demo utilities."""

import numpy as np
from scipy import ndimage


def segment_targets(recon, targets, mm_per_voxel, method='quantile'):
    """Segment the calibration targets in a reconstruction.

    Three methods.  'quantile': locate the N targets as the N
    largest connected regions above the image quantile of the
    declared total target area, re-measure each boundary with a
    2-level Otsu threshold in a window around the region, and pair
    regions with targets in order of increasing material density.
    'otsu': 2-level Otsu threshold, largest connected region, holes
    filled; single target only.
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

        # A single global threshold biases the boundaries: the
        # bright targets' skirts lie above it and the dim targets'
        # edges below it.  Re-measure each boundary with a local
        # 2-level Otsu threshold in a window around the region.
        from mbirtorch.preprocess import multi_threshold_otsu
        shapes = []
        for k in keep:
            comp = ndimage.binary_fill_holes(cc == k)
            rows_any = np.where(np.any(comp, axis=1))[0]
            cols_any = np.where(np.any(comp, axis=0))[0]
            r0, r1 = rows_any[0], rows_any[-1]
            c0, c1 = cols_any[0], cols_any[-1]
            mr, mc = (r1 - r0 + 1) // 2, (c1 - c0 + 1) // 2
            r0 = max(0, r0 - mr)
            r1 = min(img.shape[0], r1 + mr + 1)
            c0 = max(0, c0 - mc)
            c1 = min(img.shape[1], c1 + mc + 1)
            window = img[r0:r1, c0:c1]
            local_t = multi_threshold_otsu(window, classes=2)[0]
            wcc, wn = ndimage.label(window >= local_t)
            wsizes = ndimage.sum(window >= local_t, wcc,
                                 range(1, wn + 1))
            wshape = clean(wcc == int(np.argmax(wsizes)) + 1)
            shape = np.zeros_like(comp)
            shape[r0:r1, c0:c1] = wshape
            shapes.append(shape)
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


def remove_stripes(sino, row_smooth=10):
    """Remove per-channel stripe offsets from a log-domain sinogram.

    The view average of each channel gives a per-channel profile.
    A linear ramp through the profile's two ends is removed, the
    remainder is high-pass filtered along channels with a
    reflective boundary, and the high-pass part is subtracted from
    every view.  Only fine-scale per-channel structure is removed;
    the object's broad profile is untouched.

    Args:
        sino (numpy.ndarray): Log-domain sinogram with shape
            (views, rows, channels).
        row_smooth (float): Standard deviation in channels of the
            Gaussian low-pass that defines the high-pass split.
            Profile structure narrower than about this many
            channels is treated as stripes.

    Returns:
        numpy.ndarray: The destriped sinogram, float32.
    """
    sino = np.asarray(sino, dtype=np.float64).copy()
    n_chan = sino.shape[2]
    n = np.arange(n_chan)
    k = max(4, n_chan // 64)        # samples averaged at each end
    for r in range(sino.shape[1]):
        # Per-channel profile: the view average of each channel.
        profile = sino[:, r, :].mean(axis=0)
        # Linear ramp through the two ends of the profile.
        end0 = profile[:k].mean()
        end1 = profile[-k:].mean()
        c0, c1 = 0.5 * (k - 1), n_chan - 1 - 0.5 * (k - 1)
        ramp = end0 + (end1 - end0) * (n - c0) / (c1 - c0)
        residual = profile - ramp
        # High pass: the residual minus its Gaussian smoothing.
        low = ndimage.gaussian_filter1d(residual, row_smooth,
                                        mode='reflect')
        stripes = residual - low
        # Subtract the stripe estimate from every view.
        sino[:, r, :] -= stripes[None, :]
    return sino.astype(np.float32)


def remove_stripes_2d(sino, row_smooth=10, col_smooth=50):
    """Remove stripe offsets that drift slowly across views.

    Generalizes :func:`remove_stripes`: instead of one stripe
    profile from the average over all views, the stripe estimate
    varies slowly with view.  The sinogram is smoothed along views,
    each view of the smoothed sinogram is reduced to its channel
    high pass (end ramp removed), and that estimate, fine-scale
    along channels but low-frequency along views, is subtracted
    from the original sinogram.

    Args:
        sino (numpy.ndarray): Log-domain sinogram with shape
            (views, rows, channels).
        row_smooth (float): Standard deviation in channels of the
            Gaussian defining the high-pass split within each view.
        col_smooth (float): Standard deviation in views of the
            Gaussian smoothing along the view direction.

    Returns:
        numpy.ndarray: The destriped sinogram, float32.
    """
    sino = np.asarray(sino, dtype=np.float64).copy()
    n_chan = sino.shape[2]
    n = np.arange(n_chan)
    k = max(4, n_chan // 64)        # samples averaged at each end
    c0, c1 = 0.5 * (k - 1), n_chan - 1 - 0.5 * (k - 1)
    for r in range(sino.shape[1]):
        plane = sino[:, r, :]                       # (views, channels)
        # Smooth along views so the estimate is low-frequency there.
        smooth = ndimage.gaussian_filter1d(plane, col_smooth, axis=0,
                                           mode='reflect')
        # Per view: linear ramp through the two ends.
        end0 = smooth[:, :k].mean(axis=1, keepdims=True)
        end1 = smooth[:, -k:].mean(axis=1, keepdims=True)
        ramp = end0 + (end1 - end0) * (n[None, :] - c0) / (c1 - c0)
        residual = smooth - ramp
        # High pass along channels: residual minus its smoothing.
        low = ndimage.gaussian_filter1d(residual, row_smooth, axis=1,
                                        mode='reflect')
        sino[:, r, :] -= residual - low
    return sino.astype(np.float32)


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


def get_sino_and_model(path, center_offset_channels, snr_db, pixel_mm,
                  mask_subsampling_factor=1):
    """Stand in for mbirtorch preprocessing of one ALS scan file.

    Reads the full-resolution normalized transmission, removes
    stripes with remove_stripes_2d, removes each view's background
    offset, and returns the sinogram and the mbirtorch CT model.
    The model's reconstruction grid uses voxels mask_subsampling_factor
    times the detector pitch, so the reconstruction that makes the
    masks runs fast while the calibration fits the full-resolution
    sinogram.

    Args:
        path (str): The HDF5 scan file.
        center_offset_channels (float): Detector center offset in
            channels.
        snr_db (float): mbirtorch regularization parameter.
        pixel_mm (float): Detector pixel pitch in mm.
        mask_subsampling_factor (int): Reconstruction voxel size as a
            multiple of the detector pitch.

    Returns:
        tuple: (sinogram, ct_model).
    """
    import h5py
    import mbirtorch
    import mbirtorch.preprocess as mtp
    with h5py.File(path, 'r') as f:
        trans = f['data_norm'][()]          # (views, 1, channels)
    n_views = trans.shape[0]
    sino = -np.log(np.clip(trans, 1e-6, None))

    # The detector's per-channel gain error (about 6%) reconstructs
    # as ring artifacts.  Remove it in the sinogram, then remove
    # each view's background offset, estimated from the outer
    # 0.05 mm of object-free channels at each detector edge.  The
    # destriper's smoothing widths are in samples: 40 channels is
    # 0.026 mm, and 200 views is about 27 degrees of rotation.
    sino = remove_stripes_2d(sino, row_smooth=40, col_smooth=200)
    sino = mtp.correct_background_offset(
        sino, edge_width=int(round(0.052 / pixel_mm)),
        option='per_view')
    sino = np.asarray(sino).astype(np.float32)

    angles = -np.linspace(-0.5 * np.pi, 1.5 * np.pi, n_views,
                          endpoint=True)
    ct_model = mbirtorch.ParallelBeamModel(sino.shape,
                                           angles.astype(np.float32))
    ct_model.set_params(delta_det_channel=pixel_mm,
                        delta_det_row=pixel_mm,
                        det_channel_offset=center_offset_channels
                        * pixel_mm,
                        snr_db=snr_db,
                        alu_unit='mm', alu_value=1.0)
    ct_model.auto_set_recon_geometry()
    if mask_subsampling_factor != 1:
        rows, cols, slices = ct_model.get_params('recon_shape')
        ct_model.set_params(
            delta_voxel=mask_subsampling_factor * pixel_mm,
            recon_shape=(rows // mask_subsampling_factor,
                         cols // mask_subsampling_factor, slices))
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
