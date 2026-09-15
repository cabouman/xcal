"""Calibrator input validation tests (no reconstruction)."""
import numpy as np
import pytest

import xcal


class _StubShape:
    def get_params(self, name):
        return (8, 8, 1)


def _dummy_masks(n):
    m = np.zeros((8, 8, 1), np.float32)
    m[3:5, 3:5, 0] = 1.0
    return [m] * n


def _system():
    f = xcal.Filter(material='Al', thickness=1.0)
    det = xcal.Scintillator(material='CsI', thickness=0.3)
    return xcal.System(source=xcal.ReflectionSource(takeoff_angle=13.0),
                       filters=[f], detector=det), f


def test_add_scan_requires_voltage_for_tube():
    system, _ = _system()
    cal = xcal.Calibrator(system, [xcal.Target('Ti')])
    with pytest.raises(ValueError, match='voltage'):
        cal.add_scan(np.zeros((4, 1, 8)), None, [])


def test_add_scan_rejects_foreign_objects():
    system, f = _system()
    rod = xcal.Target('Ti')
    cal = xcal.Calibrator(system, [rod])
    other_rod = xcal.Target('Ti')
    with pytest.raises(ValueError, match='targets list'):
        cal.add_scan(np.zeros((4, 1, 8)), None, [], voltage=80,
                     targets=[other_rod])
    other_filter = xcal.Filter(material='Al', thickness=1.0)
    with pytest.raises(ValueError, match="System's filters"):
        cal.add_scan(np.zeros((4, 1, 8)), None, [], voltage=80,
                     filters=[other_filter])


def test_add_scan_shape_checks():
    system, _ = _system()
    cal = xcal.Calibrator(system, [xcal.Target('Ti')])
    with pytest.raises(ValueError, match='views, rows, channels'):
        cal.add_scan(np.zeros((4, 8)), None, [], voltage=80)
    with pytest.raises(ValueError, match='weights shape'):
        cal.add_scan(np.zeros((4, 1, 8)), None, [], voltage=80,
                     weights=np.zeros((4, 1, 7)))


def test_calibrate_requires_scans_and_used_filters():
    system, f = _system()
    cal = xcal.Calibrator(system, [xcal.Target('Ti')])
    with pytest.raises(ValueError, match='no scans'):
        cal.calibrate()
    # A filter that appears in no scan is an error: unused = extra
    # filter beyond the scans' declared sets.
    unused = xcal.Filter(material='Cu', thickness=1.0, name='unused')
    system2 = xcal.System(source=xcal.ReflectionSource(takeoff_angle=13.0),
                          filters=[f, unused],
                          detector=xcal.Scintillator(material='CsI',
                                                     thickness=0.3))
    cal2 = xcal.Calibrator(system2, [xcal.Target('Ti')])
    cal2.add_scan(np.zeros((4, 1, 8)), _StubShape(), _dummy_masks(1), voltage=80, filters=[f])
    with pytest.raises(ValueError, match='appears in no scan'):
        cal2.calibrate()
