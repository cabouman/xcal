"""Physics layer tests."""
import numpy as np
import pytest

from xcal import _materials, _physics


def test_energy_validation_rejects_out_of_range():
    with pytest.raises(ValueError, match='NIST'):
        _physics.check_energies(np.array([0.5, 10.0]))
    with pytest.raises(ValueError, match='NIST'):
        _physics.check_energies(np.array([10.0, 30000.0]))
    with pytest.raises(ValueError, match='increasing'):
        _physics.check_energies(np.array([10.0, 5.0]))


def test_filter_transmission_range_and_monotonicity():
    al = _materials.resolve('Al', 'filter')
    E = np.linspace(10, 150, 141)
    t1 = _physics.filter_transmission(al, 1.0, E)
    t5 = _physics.filter_transmission(al, 5.0, E)
    assert np.all((t1 > 0) & (t1 <= 1))
    assert np.all(t5 <= t1)
    # Transmission rises with energy away from edges.
    assert t1[-1] > t1[0]


def test_scintillator_response_positive():
    csi = _materials.resolve('CsI', 'scintillator')
    E = np.linspace(10, 150, 141)
    resp = _physics.scintillator_response(csi, 0.33, E)
    assert np.all(resp >= 0)
    assert resp.max() > 0


def test_default_energy_grid_convention():
    E = _physics.default_energy_grid(80)
    assert E[0] == pytest.approx(1.5)
    assert E[-1] == pytest.approx(79.5)
    assert len(E) == 79


def test_prepare_for_interpolation_zero_above_cutoff():
    # Two triangular spectra with cutoffs at bins 4 and 8.
    n = 12
    s0 = np.zeros(n)
    s0[1:5] = [1, 2, 1, 0.5]
    s1 = np.zeros(n)
    s1[1:9] = [1, 2, 3, 2, 1.5, 1, 0.6, 0.3]
    ext = _physics.prepare_for_interpolation(np.stack([s0, s1]))
    # The lower spectrum's own cutoff bin is preserved.
    assert ext[0][4] == pytest.approx(0.5)
    # Midway interpolation is close to zero above the midway cutoff.
    mid = 0.5 * (ext[0] + ext[1])
    mid = np.clip(mid, 0, None)
    cutoff_mid = (4 + 8) // 2
    assert np.all(mid[cutoff_mid + 1:-1] <= 0.16)


def test_transmission_source_table_loads():
    voltages, th_mm, energies, spectra = \
        _physics.transmission_source_table()
    assert list(voltages) == [40.0, 80.0, 150.0]
    assert len(th_mm) == 4
    assert th_mm[0] == pytest.approx(0.001)
    assert spectra.shape == (3, 4, len(energies))
    # Each spectrum is zero above its own voltage.
    assert spectra[0, 0][int(voltages[0]) + 2:].max() == 0
    assert spectra[0, 0].max() > 0


def test_als_spectrum_loads():
    energies, counts = _physics.load_als_spectrum()
    assert energies.ndim == 1 and counts.shape == energies.shape
    assert counts.max() > 0


def test_spectral_function_zero_outside_support():
    f = _physics.SpectralFunction(np.array([1.0, 2.0, 3.0]),
                                  np.array([0.0, 1.0, 0.0]))
    assert f(2.0) == pytest.approx(1.0)
    assert f(0.5) == 0.0
    assert f(4.0) == 0.0
    out = f(np.array([1.5, 2.5]))
    assert out.shape == (2,)
