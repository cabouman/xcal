"""Segmentation tests on synthetic images, with no reconstruction."""
import numpy as np
import pytest

import xcal
from xcal import _segment, _physics
from xcal.segment import _antialiased_disk


def _make_volume(rows, cols, slices, rods_at, mm):
    vol = np.zeros((rows, cols, slices), dtype=float)
    for (cy, cx, radius_mm, level) in rods_at:
        disk = _antialiased_disk(rows, cols, cy, cx, radius_mm / mm)
        vol += level * disk[:, :, None]
    return vol


def test_two_rods_found_matched_and_shaped():
    mm = 0.02
    rows = cols = 192
    energies = _physics.default_energy_grid(100)
    # V attenuates far more than Mg; Mg rod is larger.
    rods_at = [(60.0, 60.0, 0.25, 0.9),    # V-like: small, strong
               (130.0, 130.0, 0.5, 0.06)]  # Mg-like: large, weak
    vol = _make_volume(rows, cols, 4, rods_at, mm)
    vol += 0.003 * np.random.default_rng(0).standard_normal(vol.shape)

    targets = [xcal.Target('V', 0.5), xcal.Target('Mg', 1.0)]
    labels, masks = _segment.segment_targets(vol, targets, mm, energies,
                                          verbose=0)
    assert labels.max() == 2
    ys, xs = np.where(labels[:, :, 2] == 1)
    assert abs(ys.mean() - 60) < 3 and abs(xs.mean() - 60) < 3
    ys, xs = np.where(labels[:, :, 2] == 2)
    assert abs(ys.mean() - 130) < 3 and abs(xs.mean() - 130) < 3
    # The masks are measured shapes with about the right area.
    area_mm2 = masks[0][:, :, 2].sum() * mm * mm
    assert area_mm2 == pytest.approx(np.pi * 0.25 ** 2, rel=0.2)


def test_actual_shape_is_captured_not_idealized():
    """A deliberately dented rod: the mask must follow the dent."""
    mm = 0.02
    rows = cols = 160
    energies = _physics.default_energy_grid(100)
    disk = _antialiased_disk(rows, cols, 80.0, 80.0, 0.5 / mm)
    dent = _antialiased_disk(rows, cols, 80.0, 80.0 + 0.5 / mm, 0.2 / mm)
    shape = np.clip(disk - dent, 0, 1)
    vol = np.zeros((rows, cols, 4))
    vol[:, :, :] = 0.3 * shape[:, :, None]

    targets = [xcal.Target('Al', 1.0)]
    labels, masks = _segment.segment_targets(vol, targets, mm, energies,
                                          verbose=0)
    mask = masks[0][:, :, 2] > 0.5
    # The dent region is outside the mask.
    assert not mask[78:83, 96:100].any()
    # The far side of the rod is inside the mask.
    assert mask[78:83, 60:64].all()


def test_wrong_declared_diameter_raises():
    mm = 0.02
    energies = _physics.default_energy_grid(100)
    vol = _make_volume(192, 192, 4, [(96.0, 96.0, 0.25, 0.5)], mm)
    targets = [xcal.Target('Ti', 2.0)]     # declared 2 mm, actual 0.5 mm
    with pytest.raises(ValueError, match='[Ss]egmentation failed'):
        _segment.segment_targets(vol, targets, mm, energies, verbose=0)


def test_too_coarse_grid_raises():
    energies = _physics.default_energy_grid(100)
    vol = np.zeros((32, 32, 2))
    targets = [xcal.Target('Ti', 0.1)]
    with pytest.raises(ValueError, match='coarse'):
        _segment.segment_targets(vol, targets, 0.1, energies, verbose=0)
