"""Simulation of calibration scans.

A simulation needs a fully specified :class:`~xcal.System`, one whose
facts are all plain values.  :func:`simulate_scan` then generates the
sinogram one scan would measure, using the same tomography model that
the calibration will use, so the simulation and the calibration agree
about the geometry.
"""

import numpy as np

from . import _physics
from .system import System, SynchrotronSource

__all__ = ['simulate_scan']


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


def simulate_scan(system, rods, ct_model, voltage=None, filters=None,
                  rod_centers=None, photons=40000, seed=0):
    """Simulate the sinogram of one calibration scan.

    The signature mirrors :meth:`~xcal.Calibrator.add_scan`: the same
    system, rods, model, voltage, and filters describe a scan on both
    sides, so a demo simulates with one call and calibrates with the
    next.

    Args:
        system (System): A fully specified system (no estimates and
            no candidate lists); its effective spectrum is the truth.
        rods (list of Rod): The rods in the scan.
        ct_model (TomographyModel): The scan geometry.  Its alu_unit
            and alu_value parameters state the physical units.
        voltage (float, optional): Source voltage in kV.  Required
            for tube sources, ignored for synchrotron sources.
        filters (list of Filter, optional): The filters in the beam.
            Defaults to all filters in the system.
        rod_centers (list of tuple, optional): (row, column) offsets
            of each rod center from the rotation axis, in mm.
            Defaults to evenly spaced positions on a circle inside
            the field of view.
        photons (int, optional): Air photons per detector element for
            Poisson noise.  None simulates without noise.
        seed (int, optional): Random seed for the noise.

    Returns:
        numpy.ndarray: The log-domain sinogram, float32, shaped for
        the model, ready for :meth:`~xcal.Calibrator.add_scan`.
    """
    if not isinstance(system, System):
        raise TypeError(f"system must be an xcal.System, got "
                        f"{system!r}.")
    system._require_fully_specified('simulate_scan')
    if isinstance(system.source, SynchrotronSource):
        voltage = None
    elif voltage is None:
        raise ValueError("voltage is required for tube sources.")

    from .calibrator import Calibrator
    mm_per_alu = Calibrator._mm_per_alu(ct_model)
    rows, cols, slices = ct_model.get_params('recon_shape')
    mm_per_voxel = float(ct_model.get_params('delta_voxel')) * mm_per_alu

    if rod_centers is None:
        fov_mm = 0.5 * min(rows, cols) * mm_per_voxel
        ring = 0.55 * fov_mm
        rod_centers = [(ring * np.sin(2 * np.pi * k / len(rods)),
                        ring * np.cos(2 * np.pi * k / len(rods)))
                       for k in range(len(rods))]
    if len(rod_centers) != len(rods):
        raise ValueError(f"rod_centers has {len(rod_centers)} entries "
                         f"for {len(rods)} rods.")

    # Energy grid and truth spectrum.
    R = system.effective_spectrum(voltage=voltage, filters=filters)
    if isinstance(system.source, SynchrotronSource):
        energies = _physics.default_energy_grid(float(R.energies[-1]))
    else:
        energies = _physics.default_energy_grid(voltage)
    spec = R(energies)
    spec = spec / np.trapezoid(spec, energies)

    # Path length of every ray through every rod, in mm.
    total = None
    for rod, (dy_mm, dx_mm) in zip(rods, rod_centers):
        cy = (rows - 1) / 2 + dy_mm / mm_per_voxel
        cx = (cols - 1) / 2 + dx_mm / mm_per_voxel
        disk = _antialiased_disk(rows, cols, cy, cx,
                                 0.5 * rod.diameter / mm_per_voxel)
        vol = np.zeros((rows, cols, slices), np.float32)
        vol[:, :, :] = disk[:, :, None]
        path = np.asarray(ct_model.forward_project(vol)) * mm_per_alu
        mu = _physics.attenuation_coefficients(rod.material, energies)
        term = path[..., None] * mu
        total = term if total is None else total + term

    trans = np.trapezoid(np.exp(-total) * spec, energies, axis=-1)
    if photons is not None:
        rng = np.random.default_rng(seed)
        counts = rng.poisson(np.clip(trans, 0, None) * photons) / photons
        trans = np.clip(counts, 1.0 / photons, None)
    return -np.log(np.clip(trans, 1e-12, None)).astype(np.float32)
