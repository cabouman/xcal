"""Demo: calibrate a simulated synchrotron beamline with two
filtrations.

Simulates ALS-style scans: a known synchrotron source spectrum, two
rods scanned under a low filtration (Si only) and a high filtration
(Si plus Al).  The source is known, so the calibration estimates the
two filter thicknesses and the scintillator.  Ground truth: 2 mm Si,
8 mm Al, 0.05 mm LuAG.

Runs on a laptop CPU in a few minutes.
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
    rng = np.random.default_rng(11)
    t0 = time.time()
    out = './output/simulated_multi_filtration'
    os.makedirs(out, exist_ok=True)

    GT_SI_TH = 2.0      # mm
    GT_AL_TH = 8.0      # mm
    GT_SCINT_TH = 0.05  # mm LuAG
    PHOTONS = 40000

    pitch_mm = 0.02
    n_views, n_rows, n_chan = 96, 6, 160

    # The fit energy grid comes from the source table's range.
    e_tab, counts = _physics.load_als_spectrum()
    energies = _physics.default_energy_grid(float(e_tab.max()))
    src = np.interp(energies, e_tab, counts, left=0.0, right=0.0)

    lu = _materials.resolve('LuAG', 'scintillator')
    si = _materials.resolve('Si', 'filter')
    al = _materials.resolve('Al', 'filter')
    gt_scint = _physics.scintillator_response(lu, GT_SCINT_TH, energies)
    t_si = _physics.filter_transmission(si, GT_SI_TH, energies)
    t_al = _physics.filter_transmission(al, GT_AL_TH, energies)
    gt_specs = {'low': src * t_si * gt_scint,
                'high': src * t_si * t_al * gt_scint}

    rods = [xcal.Rod('Al', 0.6), xcal.Rod('Ti', 0.4)]

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

    # Both rods in one scan, repeated under each filtration.
    masks = []
    offsets = [(-0.22, -0.22), (0.22, 0.22)]
    for rod, (dy, dx) in zip(rods, offsets):
        disk = _antialiased_disk(rows, cols,
                                 (rows - 1) / 2 + dy * rows,
                                 (cols - 1) / 2 + dx * cols,
                                 0.5 * rod.diameter / mm)
        vol = np.zeros((rows, cols, slices), np.float32)
        vol[:, :, :] = disk[:, :, None]
        masks.append(vol)
    mu_curves = [_physics.attenuation_coefficients(r.material, energies)
                 for r in rods]
    paths = [model.forward_project(m) for m in masks]

    sinos = {}
    for name, spec in gt_specs.items():
        spec_n = spec / np.trapezoid(spec, energies)
        total = np.zeros(paths[0].shape + (len(energies),), np.float64)
        for L, mu in zip(paths, mu_curves):
            total += L[..., None] * mu
        trans = np.trapezoid(np.exp(-total) * spec_n, energies, axis=-1)
        c = rng.poisson(np.clip(trans, 0, None) * PHOTONS) / PHOTONS
        sinos[name] = -np.log(np.clip(c, 1.0 / PHOTONS,
                                      None)).astype(np.float32)
    print(f"simulation done in {time.time()-t0:.0f} s")

    # ---------------- Calibrate ----------------
    si_filter = xcal.Filter(material='Si',
                            thickness=xcal.estimate(0, 5), name='Si')
    al_filter = xcal.Filter(material='Al',
                            thickness=xcal.estimate(0, 10), name='Al')
    system = xcal.System(
        source=xcal.SynchrotronSource('als_bm832'),
        filters=[si_filter, al_filter],
        detector=xcal.Scintillator(thickness=xcal.estimate(0.01, 0.5)),
    )

    cal = xcal.Calibrator(system, rods)
    cal.add_scan(sinos['low'], make_model(), filters=[si_filter])
    cal.add_scan(sinos['high'], make_model(),
                 filters=[si_filter, al_filter])
    result = cal.calibrate()

    print()
    print(result.summary())
    print()
    print(f"Ground truth: Si {GT_SI_TH} mm, Al {GT_AL_TH} mm, "
          f"LuAG {GT_SCINT_TH} mm")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, spec) in zip(axes, gt_specs.items()):
        gt = spec / np.trapezoid(spec, energies)
        filts = [si_filter] if name == 'low' else [si_filter, al_filter]
        R = result.effective_spectrum(filters=filts)
        ax.plot(energies, gt, label='ground truth')
        ax.plot(energies, R(energies), '--', label='estimate')
        ax.set_title(f'{name} filtration')
        ax.set_xlabel('Energy (keV)')
        ax.legend()
        ax.grid(True)
    fig.suptitle('Effective spectrum: ground truth vs estimate')
    fig.tight_layout()
    fig.savefig(f'{out}/spectra.png', dpi=130)
    print(f"total time {time.time()-t0:.0f} s; output in {out}")
