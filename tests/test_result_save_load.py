"""Save/load round trip for CalibrationResult."""
import os
import tempfile

import numpy as np
import pytest

import xcal
from xcal import _physics
from xcal.calibrator import CalibrationResult


def _make_result():
    f1 = xcal.Filter(material='Al', thickness=xcal.estimate(0, 10),
                     name='beam')
    det = xcal.Scintillator(material=['CsI', 'GOS'],
                            thickness=xcal.estimate(0.01, 0.5))
    system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=xcal.estimate(5, 45)),
        filters=[f1], detector=det)
    targets = [xcal.Target('Ti', 1.0)]
    cal = xcal.Calibrator(system, targets)
    cal.scans = [{'voltage': 80.0, 'filters': [f1], 'targets': targets,
                  'weights': None}]
    energies = _physics.default_energy_grid(80)
    solution = {'combo': (0, 0), 'cost': 1e-5, 'source_value': 13.0,
                'filter_thicknesses': [3.0], 'detector_thickness': 0.33,
                'iterations': 100, 'all': [((0, 0), 1e-5)]}
    mu = _physics.attenuation_coefficients(targets[0].material, energies)
    A = np.exp(-np.outer(np.linspace(0.1, 1, 20), mu))
    fit_scans = [{'A': A, 'y': np.linspace(0.2, 0.9, 20),
                  'w': np.ones(20), 'filter_indices': [0],
                  'source': ('fixed', np.ones_like(energies))}]
    return CalibrationResult(cal, energies, solution, [None], [None],
                             fit_scans)


def test_save_load_round_trip():
    import shutil
    pytest.importorskip('spekpy')       # ReflectionSource spectrum
    res = _make_result()
    path = tempfile.mkdtemp()
    try:
        res.save(path)
        for name in ('summary.txt', 'feasible_system.yaml',
                     'est_system.yaml', 'fit_data.h5',
                     'plots/spectrum.png',
                     'plots/transmission_fit.png'):
            assert os.path.exists(os.path.join(path, name)), name

        loaded = CalibrationResult.load(path)
        assert loaded.params == res.params

        # The parameter table rebuilds with its provenance.
        rows = {r['name']: r for r in loaded.parameters()}
        assert rows['source takeoff angle']['origin'] == 'estimated'
        assert rows['source takeoff angle']['low'] == 5
        assert rows['detector material']['origin'] == 'estimated'

        # The response functions rebuild from the stored parameters.
        R = loaded.effective_spectrum(voltage=80)
        E = np.linspace(1, 80, 200)
        assert np.trapezoid(R(E), E) == pytest.approx(1.0, abs=5e-3)
        assert R(90.0) == 0.0

        f = loaded.filter_response(loaded.filters[0])
        assert 0 < f(60.0) < 1

        y, pred = loaded.transmission_fit(0)
        assert y.shape == pred.shape == (20,)

    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_effective_spectrum_requires_voltage_for_tube():
    res = _make_result()
    with pytest.raises(ValueError, match='voltage'):
        res.effective_spectrum()
