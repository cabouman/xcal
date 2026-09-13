"""Segmentation tests on synthetic images, with no reconstruction."""
import numpy as np
import pytest

import xcal
from xcal import _segment, _physics


class _StubModel:
    """Stands in for an mbirtorch model: only delta_voxel is used."""

    def __init__(self, delta_voxel):
        self._d = delta_voxel

    def get_params(self, name):
        assert name == 'delta_voxel'
        return self._d


def _make_volume(rows, cols, slices, rods_at, mm):
    vol = np.zeros((rows, cols, slices), dtype=float)
    for (cy, cx, radius_mm, level) in rods_at:
        disk = _segment._antialiased_disk(rows, cols, cy, cx,
                                          radius_mm / mm)
        vol += level * disk[:, :, None]
    return vol


def test_two_rods_found_and_matched():
    mm = 0.05
    rows = cols = 128
    energies = _physics.default_energy_grid(100)
    # V attenuates far more than Mg; Mg rod is larger.
    rods_at = [(40.0, 40.0, 0.25, 0.9),    # V-like: small, strong
               (88.0, 88.0, 0.5, 0.06)]    # Mg-like: large, weak
    vol = _make_volume(rows, cols, 4, rods_at, mm)
    vol += 0.003 * np.random.default_rng(0).standard_normal(vol.shape)

    rods = [xcal.Rod('V', 0.5), xcal.Rod('Mg', 1.0)]
    labels, masks = _segment.segment_rods(vol, rods, _StubModel(mm),
                                          energies, verbose=0)
    assert labels.max() == 2
    # The V rod (label 1) is at the first position.
    ys, xs = np.where(labels[:, :, 2] == 1)
    assert abs(ys.mean() - 40) < 2 and abs(xs.mean() - 40) < 2
    ys, xs = np.where(labels[:, :, 2] == 2)
    assert abs(ys.mean() - 88) < 2 and abs(xs.mean() - 88) < 2
    assert len(masks) == 2
    # The float masks integrate to the declared disk areas.
    area_mm2 = masks[0][:, :, 2].sum() * mm * mm
    assert area_mm2 == pytest.approx(np.pi * 0.25 ** 2, rel=0.05)


def test_wrong_declared_diameter_raises():
    mm = 0.05
    energies = _physics.default_energy_grid(100)
    vol = _make_volume(128, 128, 4, [(64.0, 64.0, 0.25, 0.5)], mm)
    rods = [xcal.Rod('Ti', 2.0)]     # declared 2 mm, actual 0.5 mm
    with pytest.raises(ValueError, match='[Ss]egmentation failed'):
        _segment.segment_rods(vol, rods, _StubModel(mm), energies,
                              verbose=0)


def test_too_coarse_grid_raises():
    energies = _physics.default_energy_grid(100)
    vol = np.zeros((32, 32, 2))
    rods = [xcal.Rod('Ti', 0.1)]
    with pytest.raises(ValueError, match='coarse'):
        _segment.segment_rods(vol, rods, _StubModel(0.1), energies,
                              verbose=0)
