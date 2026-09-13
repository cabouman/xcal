"""Fit engine test on a small synthetic problem with a known answer.

No reconstruction or segmentation: path lengths are synthesized
directly, so this tests the discrete search and the continuous
optimization alone.  Runtime is a few seconds on CPU.
"""
import numpy as np
import pytest

from xcal import _fit, _materials, _physics


def test_fit_recovers_material_and_thickness():
    rng = np.random.default_rng(3)
    energies = _physics.default_energy_grid(100)

    al = _materials.resolve('Al', 'filter')
    cu = _materials.resolve('Cu', 'filter')
    csi = _materials.resolve('CsI', 'scintillator')
    gos = _materials.resolve('GOS', 'scintillator')
    ti = _materials.resolve('Ti', 'rod')

    # Ground truth: Al 2 mm filter, CsI 0.3 mm detector, flat source.
    src = np.exp(-0.5 * ((energies - 45) / 18) ** 2)
    gt = (src
          * _physics.filter_transmission(al, 2.0, energies)
          * _physics.scintillator_response(csi, 0.3, energies))
    gt_n = gt / np.trapezoid(gt, energies)

    mu_ti = _physics.attenuation_coefficients(ti, energies)
    paths = np.linspace(0.05, 1.5, 400)
    A = np.exp(-np.outer(paths, mu_ti))
    y = np.trapezoid(A * gt_n, energies, axis=-1)
    y = y * (1 + 0.002 * rng.standard_normal(y.shape))

    problem = _fit.FitProblem(
        energies,
        scans=[{
            'A': A, 'y': y, 'w': 1.0 / y,
            'filter_indices': [0],
            'source': ('fixed', src),
        }],
        source_param=None,
        filters=[{
            'mu_candidates': [
                _physics.attenuation_coefficients(al, energies),
                _physics.attenuation_coefficients(cu, energies)],
            'thickness': __import__('xcal').estimate(0, 10),
        }],
        detector={
            'curve_candidates': [
                _physics.scintillator_curves(csi, energies),
                _physics.scintillator_curves(gos, energies)],
            'thickness': __import__('xcal').estimate(0.01, 0.5),
        })
    sol = problem.solve(max_iterations=800, verbose=0)

    assert sol['combo'][0] == 0, "expected Al to win over Cu"
    assert sol['combo'][1] == 0, "expected CsI to win over GOS"
    assert sol['filter_thicknesses'][0] == pytest.approx(2.0, abs=0.3)
    assert sol['detector_thickness'] == pytest.approx(0.3, abs=0.1)


def test_fixed_parameters_are_not_optimized():
    energies = _physics.default_energy_grid(60)
    al = _materials.resolve('Al', 'filter')
    csi = _materials.resolve('CsI', 'scintillator')
    src = np.ones_like(energies)
    gt = (src * _physics.filter_transmission(al, 1.0, energies)
          * _physics.scintillator_response(csi, 0.2, energies))
    gt_n = gt / np.trapezoid(gt, energies)
    ti = _materials.resolve('Ti', 'rod')
    mu = _physics.attenuation_coefficients(ti, energies)
    paths = np.linspace(0.1, 1.0, 50)
    A = np.exp(-np.outer(paths, mu))
    y = np.trapezoid(A * gt_n, energies, axis=-1)

    problem = _fit.FitProblem(
        energies,
        scans=[{'A': A, 'y': y, 'w': np.ones_like(y),
                'filter_indices': [0], 'source': ('fixed', src)}],
        source_param=None,
        filters=[{'mu_candidates':
                  [_physics.attenuation_coefficients(al, energies)],
                  'thickness': 1.0}],
        detector={'curve_candidates':
                  [_physics.scintillator_curves(csi, energies)],
                  'thickness': 0.2})
    sol = problem.solve(verbose=0)
    assert sol['filter_thicknesses'][0] == 1.0
    assert sol['detector_thickness'] == 0.2
    assert sol['cost'] < 1e-10
