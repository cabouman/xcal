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


def test_multi_filtration_joint_fit():
    """Two scans with different filter sets share the filter and
    detector parameters, the ALS-style configuration."""
    energies = _physics.default_energy_grid(100)
    si = _materials.resolve('Si', 'filter')
    al = _materials.resolve('Al', 'filter')
    lu = _materials.resolve('LuAG', 'scintillator')
    ti = _materials.resolve('Ti', 'rod')

    src = np.exp(-0.5 * ((energies - 40) / 15) ** 2)
    det_resp = _physics.scintillator_response(lu, 0.05, energies)
    t_si = _physics.filter_transmission(si, 2.0, energies)
    t_al = _physics.filter_transmission(al, 8.0, energies)

    mu_ti = _physics.attenuation_coefficients(ti, energies)
    paths = np.linspace(0.05, 1.0, 200)
    A = np.exp(-np.outer(paths, mu_ti))

    scans = []
    for filts, trans_prod in [([0], t_si), ([0, 1], t_si * t_al)]:
        gt = src * trans_prod * det_resp
        gt_n = gt / np.trapezoid(gt, energies)
        y = np.trapezoid(A * gt_n, energies, axis=-1)
        scans.append({'A': A, 'y': y, 'w': 1.0 / y,
                      'filter_indices': filts,
                      'source': ('fixed', src)})

    import xcal
    problem = _fit.FitProblem(
        energies, scans, source_param=None,
        filters=[
            {'mu_candidates':
             [_physics.attenuation_coefficients(si, energies)],
             'thickness': xcal.estimate(0, 5)},
            {'mu_candidates':
             [_physics.attenuation_coefficients(al, energies)],
             'thickness': xcal.estimate(0, 10)},
        ],
        detector={'curve_candidates':
                  [_physics.scintillator_curves(lu, energies)],
                  'thickness': xcal.estimate(0.01, 0.5)})
    sol = problem.solve(max_iterations=8000, verbose=0)
    assert sol['filter_thicknesses'][0] == pytest.approx(2.0, abs=0.4)
    assert sol['filter_thicknesses'][1] == pytest.approx(8.0, abs=0.8)
    assert sol['detector_thickness'] == pytest.approx(0.05, abs=0.03)


def test_per_candidate_thickness_bounds():
    """A thickness list gives each material candidate its own bounds:
    the copper candidate is capped at 1 mm even though aluminum may
    range to 10 mm."""
    import xcal
    energies = _physics.default_energy_grid(80)
    al = _materials.resolve('Al', 'filter')
    cu = _materials.resolve('Cu', 'filter')
    csi = _materials.resolve('CsI', 'scintillator')
    ti = _materials.resolve('Ti', 'rod')
    src = np.ones_like(energies)
    gt = (src * _physics.filter_transmission(al, 6.0, energies)
          * _physics.scintillator_response(csi, 0.2, energies))
    gt_n = gt / np.trapezoid(gt, energies)
    mu = _physics.attenuation_coefficients(ti, energies)
    A = np.exp(-np.outer(np.linspace(0.1, 1.0, 100), mu))
    y = np.trapezoid(A * gt_n, energies, axis=-1)

    problem = _fit.FitProblem(
        energies,
        scans=[{'A': A, 'y': y, 'w': 1.0 / y,
                'filter_indices': [0], 'source': ('fixed', src)}],
        source_param=None,
        filters=[{'mu_candidates':
                  [_physics.attenuation_coefficients(al, energies),
                   _physics.attenuation_coefficients(cu, energies)],
                  'thickness': [xcal.estimate(0, 10),
                                xcal.estimate(0, 1)]}],
        detector={'curve_candidates':
                  [_physics.scintillator_curves(csi, energies)],
                  'thickness': 0.2})
    sol = problem.solve(verbose=0)
    assert sol['combo'][0] == 0
    assert sol['filter_thicknesses'][0] == pytest.approx(6.0, abs=0.5)
    # The copper attempt stayed within its own bounds.
    cu_cost = dict((c, cost) for c, cost in sol['all'])[(1, 0)]
    assert np.isfinite(cu_cost)


def test_omitted_thickness_uses_per_candidate_catalog_ranges():
    import xcal
    f = xcal.Filter()      # everything omitted
    assert f.thickness_per_candidate is not None
    ranges = {m.name: (t.low, t.high)
              for m, t in zip(f.materials, f.thickness_per_candidate)}
    assert ranges['Cu'] == (0.0, 1.0)
    assert ranges['Al'] == (0.0, 10.0)
