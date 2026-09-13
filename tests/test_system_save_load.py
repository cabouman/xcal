"""System save and load round trips."""
import os
import tempfile

import numpy as np
import pytest

import xcal


def _round_trip(system):
    path = tempfile.mktemp(suffix='.yaml')
    try:
        system.save(path)
        return xcal.load_system(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_fully_specified_round_trip():
    gt = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=20.0),
        filters=[xcal.Filter('Al', thickness=5.0, name='beam')],
        detector=xcal.Scintillator('CsI', thickness=0.25))
    back = _round_trip(gt)
    E = np.linspace(1.5, 99.5, 200)
    a = gt.effective_spectrum(voltage=100)(E)
    b = back.effective_spectrum(voltage=100)(E)
    assert np.allclose(a, b)
    assert back.filters[0].name == 'beam'


def test_feasible_round_trip():
    feasible = xcal.System(
        source=xcal.ReflectionSource(
            takeoff_angle=xcal.estimate(5, 45, initial=12)),
        filters=[xcal.Filter(material=['Al', 'Cu'])],
        detector=xcal.Scintillator(thickness=xcal.estimate(0.01, 0.5)))
    back = _round_trip(feasible)
    assert back.source.takeoff_angle.low == 5
    assert back.source.takeoff_angle.initial == 12
    assert [m.name for m in back.filters[0].materials] == ['Al', 'Cu']
    # Omitted thickness stays omitted: per-candidate catalog bounds.
    assert back.filters[0].thickness_per_candidate is not None
    assert back.detector.thickness.high == pytest.approx(0.5)


def test_synchrotron_custom_spectrum_round_trip():
    # A user adds a spectrum by dropping a file in xcal/source_models.
    E = np.linspace(1, 100, 100)
    counts = np.exp(-0.5 * ((E - 30) / 10) ** 2)
    csv_path = os.path.join(os.path.dirname(xcal.__file__),
                            'source_models',
                            'synchrotron_test_gauss.csv')
    with open(csv_path, 'w') as f:
        f.write('energy_keV,photon_counts\n')
        for e, c in zip(E, counts):
            f.write(f'{e},{c}\n')
    try:
        s = xcal.System(
            source=xcal.SynchrotronSource('test_gauss'),
            filters=[xcal.Filter('Si', thickness=2.0)],
            detector=xcal.Scintillator('LuAG', thickness=0.05))
        back = _round_trip(s)
        assert back.source.spectrum == 'test_gauss'
        a = s.effective_spectrum()(E[5:])
        b = back.effective_spectrum()(E[5:])
        assert np.allclose(a, b)
    finally:
        os.unlink(csv_path)


def test_load_rejects_other_files():
    path = tempfile.mktemp(suffix='.yaml')
    with open(path, 'w') as f:
        f.write('hello: world\n')
    try:
        with pytest.raises(ValueError, match='xcal system'):
            xcal.load_system(path)
    finally:
        os.unlink(path)
