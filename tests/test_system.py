"""System description validation tests."""
import pytest

import xcal


def test_estimate_validation():
    e = xcal.estimate(0, 10)
    assert e.initial == 5.0
    with pytest.raises(ValueError, match='low < high'):
        xcal.estimate(5, 5)
    with pytest.raises(ValueError, match='outside'):
        xcal.estimate(0, 1, initial=2)


def test_rod_validation():
    rod = xcal.Target(material='Ti', size=1.0)
    assert rod.material.name == 'Ti'
    with pytest.raises(ValueError, match='positive'):
        xcal.Target(material='Ti', size=0)


def test_filter_defaults_come_from_catalog():
    f = xcal.Filter()
    assert {m.name for m in f.materials} == {'Al', 'Cu', 'Si'}
    # The default thickness estimate spans the catalog ranges.
    assert f.thickness.low == 0.0
    assert f.thickness.high == 10.0


def test_scintillator_defaults():
    s = xcal.Scintillator()
    assert len(s.materials) == 7
    assert s.thickness.high == pytest.approx(0.5)


def test_system_validation():
    f = xcal.Filter(material='Al', thickness=1.0)
    det = xcal.Scintillator(material='CsI', thickness=0.3)
    src = xcal.ReflectionSource()
    system = xcal.System(source=src, filters=[f], detector=det)
    assert system.filter_label(f) == 'filter 1 (Al)'

    with pytest.raises(ValueError, match='twice'):
        xcal.System(source=src, filters=[f, f], detector=det)
    with pytest.raises(TypeError, match='Scintillator'):
        xcal.System(source=src, filters=[f], detector=None)
    with pytest.raises(TypeError, match='source'):
        xcal.System(source=f, filters=[], detector=det)


def test_filter_name_appears_in_label():
    f = xcal.Filter(material=['Al', 'Cu'], thickness=xcal.estimate(0, 10),
                    name='beam filter')
    det = xcal.Scintillator(material='CsI', thickness=0.3)
    system = xcal.System(source=xcal.ReflectionSource(), filters=[f],
                         detector=det)
    assert system.filter_label(f) == 'filter 1 (beam filter)'


def test_synchrotron_source_forms():
    s = xcal.SynchrotronSource()
    assert s.spectrum == 'als_bm832'
    energies, counts = s.table()
    assert energies.shape == counts.shape and counts.max() > 0
    with pytest.raises(ValueError, match='available'):
        xcal.SynchrotronSource('als_unknown')


def test_energy_grid():
    syn = xcal.System(source=xcal.SynchrotronSource(),
                      detector=xcal.Scintillator('CsI', thickness=0.25))
    E = syn.energy_grid()
    assert E[0] == 1.5 and E[-1] == pytest.approx(99.5)

    tube = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=20.0),
        detector=xcal.Scintillator('CsI', thickness=0.25))
    E = tube.energy_grid(50)
    assert E[-1] == pytest.approx(49.5)
    with pytest.raises(ValueError, match='voltage'):
        tube.energy_grid()
