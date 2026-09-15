"""Builds and reviews calibration target masks.

A mask set is a Python list with one float32 volume per calibration
target, ordered like the targets list, so masks[k] belongs to
targets[k].  Values lie in [0, 1] and mean the fraction of the voxel
occupied by the target; forward projection of a mask gives the
target's path length along every ray.

Segmenting targets from a reconstruction is the application's job,
not xcal's: it depends on the scan, and the user must see and judge
it.  The demos show how, using mbirtorch's segmentation utilities.
For simulations, :func:`cylinder_masks` builds ideal cylindrical
masks as ground truth.
"""

import numpy as np

from . import _physics

__all__ = ['cylinder_masks']


def cylinder_masks(targets, ct_model, diameters, centers=None):
    """Builds an ideal cylindrical mask for each target.

    Boundary voxels hold coverage fractions, so the masks are
    accurate to a fraction of a voxel.  Simulations use these as
    ground truth, and a calibration may also use them when the
    target geometry is trusted.

    Args:
        targets (list of Target): The calibration targets, in the
            order the masks are returned.
        ct_model (TomographyModel): The scan geometry, which
            provides the reconstruction grid and voxel size.
        diameters (list of float): The cylinder diameter of each
            target, in mm.
        centers (list of tuple, optional): The (row, column) offset
            of each cylinder center from the rotation axis, in mm.
            It defaults to evenly spaced positions on a circle
            inside the field of view.

    Returns:
        list of numpy.ndarray: One float32 mask volume per target,
        in target order, with values in [0, 1].
    """
    scale = _physics.mm_per_alu(ct_model)
    rows, cols, slices = ct_model.get_params('recon_shape')
    mm_per_voxel = float(ct_model.get_params('delta_voxel')) * scale

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
    """Returns a 2D float mask of a disk, its boundary voxels
    holding coverage fractions computed by supersampling."""
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
