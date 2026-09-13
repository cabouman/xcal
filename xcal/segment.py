"""Calibration target masks: segmented from a reconstruction, or
built as ideal shapes.

A mask set is a Python list with one float32 volume per calibration
target, ordered like the targets list, so masks[k] belongs to
targets[k].  Values lie in [0, 1] and mean the fraction of the voxel
occupied by the target; forward projection of a mask gives the
target's path length along every ray.

Masks come from one of two places:

* :func:`segment_targets` measures them from a reconstruction, which
  is what real calibrations use.  The targets may be any regular
  shape; the mask records what is there.
* :func:`cylinder_masks` builds ideal cylindrical masks, which
  simulations use as ground truth.
"""

import numpy as np

from . import _physics
from . import _segment

__all__ = ['segment_targets', 'cylinder_masks']

# The energy band used only to rank targets by expected attenuation
# when matching segmented shapes to declared targets.  The match is
# scale invariant, so any band with the usual material ordering
# works.
_MATCH_BAND_KEV = (20.0, 100.0)


def segment_targets(recon, targets, ct_model, verbose=1):
    """Segment the calibration targets in a reconstruction and return
    their masks.

    Look at the returned masks before calibrating: overlay them on
    the reconstruction and check that every target's shape is
    sensible.

    Args:
        recon (numpy.ndarray): Reconstructed volume with shape
            (rows, cols, slices), from the model's recon method, in
            the model's units (1/ALU).
        targets (list of Target): The targets expected in this scan.
        ct_model (TomographyModel): The model the reconstruction came
            from; provides the voxel size and units.
        verbose (int, optional): 1 prints what was found.

    Returns:
        list of numpy.ndarray: One float32 mask volume per target, in
        target order, values 0 or 1 (measured shapes).
    """
    scale = _physics.mm_per_alu(ct_model)
    mm_per_voxel = float(ct_model.get_params('delta_voxel')) * scale
    recon = np.asarray(recon) / scale       # to 1/mm
    band = np.linspace(_MATCH_BAND_KEV[0], _MATCH_BAND_KEV[1], 81)
    _, masks = _segment.segment_targets(recon, targets, mm_per_voxel,
                                        band, verbose=verbose)
    return masks


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
