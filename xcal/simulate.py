"""Simulates calibration scans.

A simulation needs a fully specified :class:`~xcal.System`, one whose
facts are all plain values.  :func:`simulate_scan` then generates the
sinogram one scan would measure, using the same tomography model that
the calibration will use, so the simulation and the calibration agree
about the geometry.
"""

import numpy as np

from . import _physics
from .segment import cylinder_masks
from .system import System, SynchrotronSource

__all__ = ['simulate_scan']


def simulate_scan(system, targets, ct_model, voltage=None,
                  filters=None, target_masks=None, photons=40000,
                  seed=0):
    """Simulates the sinogram of one calibration scan.

    The signature mirrors :meth:`~xcal.Calibrator.add_scan`: the same
    system, rods, model, voltage, and filters describe a scan on both
    sides, so a demo simulates with one call and calibrates with the
    next.

    Args:
        system (System): A fully specified system (no estimates and
            no candidate lists); its effective spectrum is the truth.
        targets (list of Target): The calibration targets in the
            scan.
        ct_model (TomographyModel): The scan geometry.  Its alu_unit
            and alu_value parameters state the physical units.
        voltage (float, optional): Peak tube voltage (kVp) in kV.  Required
            for tube sources, ignored for synchrotron sources.
        filters (list of Filter, optional): The filters in the beam.
            Defaults to all filters in the system.
        target_masks (list of numpy.ndarray, optional): One mask
            volume per target, values in [0, 1].  Defaults to
            cylinders of each target's declared size, evenly spaced
            on a circle (from :func:`~xcal.cylinder_masks`).  Pass
            the same masks to :meth:`~xcal.Calibrator.add_scan` for a
            ground truth calibration.
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

    scale = _physics.mm_per_alu(ct_model)
    masks = (target_masks if target_masks is not None
             else cylinder_masks(targets, ct_model))

    # Energy grid and truth spectrum.
    R = system.effective_spectrum(voltage=voltage, filters=filters)
    if isinstance(system.source, SynchrotronSource):
        energies = _physics.default_energy_grid(float(R.energies[-1]))
    else:
        energies = _physics.default_energy_grid(voltage)
    spec = R(energies)
    spec = spec / np.trapezoid(spec, energies)

    # Path length of every ray through every target, in mm.
    total = None
    for target, vol in zip(targets, masks):
        path = np.asarray(ct_model.forward_project(vol)) * scale
        mu = _physics.attenuation_coefficients(target.material,
                                                energies)
        term = path[..., None] * mu
        total = term if total is None else total + term

    trans = np.trapezoid(np.exp(-total) * spec, energies, axis=-1)
    if photons is not None:
        rng = np.random.default_rng(seed)
        counts = rng.poisson(np.clip(trans, 0, None) * photons) / photons
        trans = np.clip(counts, 1.0 / photons, None)
    return -np.log(np.clip(trans, 1e-12, None)).astype(np.float32)
