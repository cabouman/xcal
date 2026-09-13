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
    rods = [xcal.Rod('Ti', 1.0)]
    cal = xcal.Calibrator(system, rods)
    cal.scans = [{'voltage': 80.0, 'filters': [f1], 'rods': rods,
                  'weights': None}]
    energies = _physics.default_energy_grid(80)
    solution = {'combo': (0, 0), 'cost': 1e-5, 'source_value': 13.0,
                'filter_thicknesses': [3.0], 'detector_thickness': 0.33,
                'iterations': 100, 'all': [((0, 0), 1e-5)]}
    mu = _physics.attenuation_coefficients(rods[0].material, energies)
    A = np.exp(-np.outer(np.linspace(0.1, 1, 20), mu))
    fit_scans = [{'A': A, 'y': np.linspace(0.2, 0.9, 20),
                  'w': np.ones(20), 'filter_indices': [0],
                  'source': ('fixed', np.ones_like(energies))}]
    return CalibrationResult(cal, energies, solution, [None], [None],
                             [None], [None], fit_scans)


def test_save_load_round_trip():
    res = _make_result()
    path = tempfile.mktemp(suffix='.h5')
    try:
        res.save(path)
        loaded = CalibrationResult.load(path)
        assert loaded.params == res.params

        # The response functions rebuild from the stored parameters.
        R = loaded.effective_spectrum(voltage=80)
        E = np.linspace(1, 80, 200)
        assert np.trapezoid(R(E), E) == pytest.approx(1.0, abs=1e-3)
        assert R(90.0) == 0.0

        f = loaded.filter_response(loaded.filters[0])
        assert 0 < f(60.0) < 1

        y, pred = loaded.transmission_fit(0)
        assert y.shape == pred.shape == (20,)

        with pytest.raises(ValueError, match='not stored'):
            loaded.reconstruction(0)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_effective_spectrum_requires_voltage_for_tube():
    res = _make_result()
    with pytest.raises(ValueError, match='voltage'):
        res.effective_spectrum()
