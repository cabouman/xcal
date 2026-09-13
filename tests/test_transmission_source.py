"""Transmission source: table interpolation and thickness recovery.

Covers the third source type without a reconstruction: the ground
truth transmission spectrum is drawn from the shipped Geant4 table at
an off-grid voltage and thickness, and the fit recovers the target
thickness from synthetic transmissions.
"""
import numpy as np
import pytest

import xcal
from xcal import _fit, _materials, _physics


def _source_table_at(voltage, energies):
    """Voltage-interpolated thickness table on the fit grid, the same
    computation the calibrator performs."""
    voltages, th_mm, e_tab, spectra = _physics.transmission_source_table()
    per_th = []
    for ti in range(len(th_mm)):
        ext = _physics.prepare_for_interpolation(spectra[:, ti])
        row = _physics.interpolate_rows(voltages, ext, voltage)
        per_th.append(np.clip(row, 0.0, None))
    table = np.stack([np.interp(energies, e_tab, r, left=0.0, right=0.0)
                      for r in per_th])
    return th_mm, table


def test_voltage_interpolation_zero_above_cutoff():
    energies = _physics.default_energy_grid(150)
    th_mm, table = _source_table_at(110.0, energies)
    for row in table:
        above = energies > 116     # a few keV above the 110 kV cutoff
        assert row[above].max() <= 0.02 * row.max()
        assert row.max() > 0


def test_fit_recovers_target_thickness():
    energies = _physics.default_energy_grid(150)
    th_mm, table = _source_table_at(110.0, energies)

    # Ground truth: thickness 4 um, between the 3 and 5 um grid points.
    gt_src = _physics.interpolate_rows(th_mm, table, 0.004)
    al = _materials.resolve('Al', 'filter')
    csi = _materials.resolve('CsI', 'scintillator')
    gt = (gt_src * _physics.filter_transmission(al, 2.0, energies)
          * _physics.scintillator_response(csi, 0.3, energies))
    gt_n = gt / np.trapezoid(gt, energies)

    ti = _materials.resolve('Ti', 'target')
    mu = _physics.attenuation_coefficients(ti, energies)
    paths = np.linspace(0.05, 1.5, 300)
    A = np.exp(-np.outer(paths, mu))
    y = np.trapezoid(A * gt_n, energies, axis=-1)

    problem = _fit.FitProblem(
        energies,
        scans=[{'A': A, 'y': y, 'w': 1.0 / y,
                'filter_indices': [0],
                'source': ('table', th_mm, table)}],
        source_param=xcal.estimate(0.001, 0.007),
        filters=[{'mu_candidates':
                  [_physics.attenuation_coefficients(al, energies)],
                  'thickness': 2.0}],
        detector={'curve_candidates':
                  [_physics.scintillator_curves(csi, energies)],
                  'thickness': 0.3})
    sol = problem.solve(verbose=0)
    assert sol['source_value'] == pytest.approx(0.004, abs=0.0006)
    assert sol['cost'] < 1e-8
