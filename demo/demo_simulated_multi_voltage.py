"""Demo: calibrate a simulated three-voltage scanner.

Simulates CT scans of four metal rods at 80, 130, and 180 kV with a
known reflection source, filter, and scintillator, then runs the full
xcal calibration and compares the estimates to the ground truth.

Runs on a laptop CPU in about two minutes.  Ground truth: takeoff
angle 13 degrees, 3 mm Al filter, 0.33 mm CsI scintillator.
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import mbirtorch
import xcal
from xcal import _physics, _materials
from xcal._segment import _antialiased_disk

if __name__ == '__main__':
    rng = np.random.default_rng(7)
    t0 = time.time()
    out = './output/simulated_multi_voltage'
    os.makedirs(out, exist_ok=True)

    # ---------------- Ground truth ----------------
    GT_ANGLE = 13.0        # degrees
    GT_FLTR_TH = 3.0       # mm Al
    GT_SCINT_TH = 0.33     # mm CsI
    VOLTAGES = [80.0, 130.0, 180.0]
    PHOTONS = 40000        # air photons per detector element

    pitch_mm = 0.05
    n_views, n_rows, n_chan = 96, 6, 192

    rods = [xcal.Rod('V', 0.5), xcal.Rod('Ti', 0.5),
            xcal.Rod('Al', 1.0), xcal.Rod('Mg', 1.0)]

    energies = _physics.default_energy_grid(max(VOLTAGES))
    gt_fltr = _physics.filter_transmission(
        _materials.resolve('Al', 'filter'), GT_FLTR_TH, energies)
    gt_scint = _physics.scintillator_response(
        _materials.resolve('CsI', 'scintillator'), GT_SCINT_TH, energies)
    gt_specs = {}
    for v in VOLTAGES:
        src = _physics.reflection_source_table(v, [GT_ANGLE], energies)[0]
        gt_specs[v] = src * gt_fltr * gt_scint

    # ---------------- Simulate the scans ----------------
    def make_model():
        angles = np.linspace(0, np.pi, n_views,
                             endpoint=False).astype(np.float32)
        m = mbirtorch.ParallelBeamModel((n_views, n_rows, n_chan), angles,
                                        compile_mode='off')
        m.set_params(delta_det_channel=pitch_mm, delta_det_row=pitch_mm)
        m.auto_set_recon_geometry()
        m.configure_devices(devices=['cpu'])
        return m

    model = make_model()
    rows, cols, slices = model.get_params('recon_shape')
    mm = model.get_params('delta_voxel')

    center_r = 0.28 * rows
    masks = []
    for k, rod in enumerate(rods):
        theta = 2 * np.pi * k / len(rods)
        cy = (rows - 1) / 2 + center_r * np.sin(theta)
        cx = (cols - 1) / 2 + center_r * np.cos(theta)
        disk = _antialiased_disk(rows, cols, cy, cx,
                                 0.5 * rod.diameter / mm)
        vol = np.zeros((rows, cols, slices), np.float32)
        vol[:, :, :] = disk[:, :, None]
        masks.append(vol)

    mu_curves = [_physics.attenuation_coefficients(r.material, energies)
                 for r in rods]
    paths = [model.forward_project(m) for m in masks]

    sinos = []
    for v in VOLTAGES:
        spec_n = gt_specs[v] / np.trapezoid(gt_specs[v], energies)
        total = np.zeros(paths[0].shape + (len(energies),), np.float64)
        for L, mu in zip(paths, mu_curves):
            total += L[..., None] * mu
        trans = np.trapezoid(np.exp(-total) * spec_n, energies, axis=-1)
        counts = rng.poisson(np.clip(trans, 0, None) * PHOTONS) / PHOTONS
        sinos.append(-np.log(np.clip(counts, 1.0 / PHOTONS,
                                     None)).astype(np.float32))
    print(f"simulation done in {time.time()-t0:.0f} s")

    # ---------------- Calibrate ----------------
    system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=xcal.estimate(5, 45)),
        filters=[xcal.Filter(material=['Al', 'Cu'],
                             thickness=xcal.estimate(0, 10))],
        detector=xcal.Scintillator(thickness=xcal.estimate(0.001, 0.5)),
    )

    cal = xcal.Calibrator(system, rods)
    for v, sino in zip(VOLTAGES, sinos):
        cal.add_scan(sino, make_model(), voltage=v)
    result = cal.calibrate()

    print()
    print(result.summary())
    print()
    print(f"Ground truth: takeoff angle {GT_ANGLE} deg; "
          f"Al filter {GT_FLTR_TH} mm; CsI scintillator {GT_SCINT_TH} mm")

    # ---------------- Compare to ground truth ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, v in zip(axes, VOLTAGES):
        gt = gt_specs[v] / np.trapezoid(gt_specs[v], energies)
        R = result.effective_spectrum(voltage=v)
        ax.plot(energies, gt, label='ground truth')
        ax.plot(energies, R(energies), '--', label='estimate')
        ax.set_title(f'{v:.0f} kV')
        ax.set_xlabel('Energy (keV)')
        ax.legend()
        ax.grid(True)
    fig.suptitle('Effective spectrum: ground truth vs estimate')
    fig.tight_layout()
    fig.savefig(f'{out}/spectra.png', dpi=130)

    result.save(f'{out}/calibration.h5')
    print(f"total time {time.time()-t0:.0f} s; output in {out}")
    result.show()
