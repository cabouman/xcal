"""Calibration target masks.

A mask set is a Python list with one float32 volume per calibration
target, ordered like the targets list, so masks[k] belongs to
targets[k].  Values lie in [0, 1] and mean the fraction of the voxel
occupied by the target; forward projection of a mask gives the
target's path length along every ray.

Segmenting targets from a reconstruction is the application's job,
not xcal's: it depends on the scan, and the user must see and judge
it.  The demos show how, using mbirtorch's segmentation utilities;
:func:`save_segmentation_plot` writes the review image.  For
simulations, :func:`cylinder_masks` builds ideal cylindrical masks
as ground truth.
"""

import numpy as np

from . import _physics

__all__ = ['cylinder_masks', 'save_segmentation_plot']


def save_segmentation_plot(recon, targets, masks, filename,
                           title=None):
    """Write a review image of a segmentation.

    The center slice of the reconstruction with each target's mask
    outlined in its own color; a legend outside the image names each
    color's material and size, so the image itself stays clean.

    Args:
        recon (numpy.ndarray): Reconstructed volume with shape
            (rows, cols, slices).
        targets (list of Target): The targets, in mask order.
        masks (list of numpy.ndarray): One mask volume per target,
            from :func:`segment_targets` or a mask builder.
        filename (str): Output image path.
        title (str, optional): Title above the image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = ['red', 'cyan', 'orange', 'magenta', 'lime', 'yellow']
    recon = np.asarray(recon)
    s = recon.shape[2] // 2
    fig, ax = plt.subplots(figsize=(7.4, 6))
    ax.imshow(recon[:, :, s], origin='lower')
    handles = []
    for i, (tg, m) in enumerate(zip(targets, masks)):
        c = colors[i % len(colors)]
        ax.contour(np.asarray(m)[:, :, s], levels=[0.5], colors=[c],
                   linewidths=0.9)
        handles.append(Line2D([0], [0], color=c,
                              label=f'{tg.material.name} '
                                    f'({tg.size:g} mm)'))
    ax.legend(handles=handles, loc='center left',
              bbox_to_anchor=(1.02, 0.5))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def cylinder_masks(targets, ct_model, diameters=None, centers=None):
    """Build ideal cylindrical masks for the targets.

    Boundary voxels hold coverage fractions, so the masks are
    accurate to a fraction of a voxel.  Simulations use these as
    ground truth; a calibration may also use them when the target
    geometry is trusted.  Other shapes get their own builders.

    Args:
        targets (list of Target): The targets.
        ct_model (TomographyModel): The scan geometry; provides the
            grid and voxel size.
        diameters (list of float, optional): Cylinder diameters in
            mm.  Defaults to each target's declared size.
        centers (list of tuple, optional): (row, column) offsets of
            each cylinder center from the rotation axis, in mm.
            Defaults to evenly spaced positions on a circle inside
            the field of view.

    Returns:
        list of numpy.ndarray: One float32 mask volume per target, in
        target order, values in [0, 1].
    """
    scale = _physics.mm_per_alu(ct_model)
    rows, cols, slices = ct_model.get_params('recon_shape')
    mm_per_voxel = float(ct_model.get_params('delta_voxel')) * scale

    if diameters is None:
        diameters = [t.size for t in targets]
    if centers is None:
        fov_mm = 0.5 * min(rows, cols) * mm_per_voxel
        ring = 0.55 * fov_mm
        centers = [(ring * np.sin(2 * np.pi * k / len(targets)),
                    ring * np.cos(2 * np.pi * k / len(targets)))
                   for k in range(len(targets))]
    if len(centers) != len(targets) or len(diameters) != len(targets):
        raise ValueError(
            f"centers has {len(centers)} and diameters has "
            f"{len(diameters)} entries for {len(targets)} targets.")

    masks = []
    for diameter, (dy_mm, dx_mm) in zip(diameters, centers):
        cy = (rows - 1) / 2 + dy_mm / mm_per_voxel
        cx = (cols - 1) / 2 + dx_mm / mm_per_voxel
        disk = _antialiased_disk(rows, cols, cy, cx,
                                 0.5 * diameter / mm_per_voxel)
        vol = np.zeros((rows, cols, slices), np.float32)
        vol[:, :, :] = disk[:, :, None]
        masks.append(vol)
    return masks


def _antialiased_disk(rows, cols, cy, cx, radius_vox, supersample=4):
    """A 2D float mask of a disk, boundary voxels holding coverage
    fractions computed by supersampling."""
    disk = np.zeros((rows, cols), dtype=np.float32)
    r_out = int(np.ceil(radius_vox)) + 2
    r0 = max(0, int(cy) - r_out)
    r1 = min(rows, int(cy) + r_out + 1)
    c0 = max(0, int(cx) - r_out)
    c1 = min(cols, int(cx) + r_out + 1)
    s = supersample
    offs = (np.arange(s) + 0.5) / s - 0.5
    oy, ox = np.meshgrid(offs, offs, indexing='ij')
    for y in range(r0, r1):
        for x in range(c0, c1):
            d2 = (y + oy - cy) ** 2 + (x + ox - cx) ** 2
            disk[y, x] = float((d2 <= radius_vox ** 2).mean())
    return disk
