"""Demo: calibrate a simulated synchrotron beamline with two
filtrations.

Simulates ALS-style scans: a known synchrotron source spectrum and
two rods, scanned under a low filtration (Si filter only) and a high
filtration (Si plus Al).  The source is known, so the calibration
estimates the two filter thicknesses and the scintillator.

Ground truth: 2 mm Si, 8 mm Al, 0.05 mm LuAG.  Runs on a laptop in
a few minutes.
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import mbirtorch
import xcal

if __name__ == '__main__':
    t0 = time.time()
    out = './output/simulated_multi_filtration'
    os.makedirs(out, exist_ok=True)

    # ---------------- The truth to recover ----------------
    si_true = xcal.Filter('Si', thickness=2.0, name='Si')
    al_true = xcal.Filter('Al', thickness=8.0, name='Al')
    truth = xcal.System(
        source=xcal.SynchrotronSource('als_bm832'),
        filters=[si_true, al_true],
        detector=xcal.Scintillator('LuAG', thickness=0.05),
    )
    rods = [xcal.Rod('Al', 0.6), xcal.Rod('Ti', 0.4)]

    # ---------------- The simulated scanner ----------------
    def make_model():
        n_views, n_rows, n_chan = 96, 4, 320
        angles = np.linspace(0, np.pi, n_views,
                             endpoint=False).astype(np.float32)
        m = mbirtorch.ParallelBeamModel((n_views, n_rows, n_chan),
                                        angles)
        m.set_params(delta_det_channel=0.01, delta_det_row=0.01,
                     alu_unit='mm', alu_value=1.0)
        m.auto_set_recon_geometry()
        return m

    # ---------------- Simulate and calibrate ----------------
    si_filter = xcal.Filter('Si', thickness=xcal.estimate(0, 5),
                            name='Si')
    al_filter = xcal.Filter('Al', thickness=xcal.estimate(0, 10),
                            name='Al')
    system = xcal.System(
        source=xcal.SynchrotronSource('als_bm832'),
        filters=[si_filter, al_filter],
        detector=xcal.Scintillator(thickness=xcal.estimate(0.01, 0.5)),
    )
    cal = xcal.Calibrator(system, rods)
    for i, (truth_filts, filts) in enumerate(
            [([si_true], [si_filter]),
             ([si_true, al_true], [si_filter, al_filter])]):
        model = make_model()
        sino = xcal.simulate_scan(truth, rods, model,
                                  filters=truth_filts, photons=40000,
                                  seed=i)
        cal.add_scan(sino, model, filters=filts)
        print(f'simulated scan {i} ({time.time()-t0:.0f} s)')
    cal_result = cal.calibrate()

    print()
    print(cal_result.summary())
    print()
    print('Ground truth: Si 2.0 mm, Al 8.0 mm, LuAG 0.05 mm')

    # ---------------- Compare to the truth ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    E = np.linspace(1.5, 99.5, 400)
    for ax, (truth_filts, filts, label) in zip(axes, [
            ([si_true], [si_filter], 'low filtration'),
            ([si_true, al_true], [si_filter, al_filter],
             'high filtration')]):
        ax.plot(E, truth.effective_spectrum(filters=truth_filts)(E),
                label='ground truth')
        ax.plot(E, cal_result.effective_spectrum(filters=filts)(E), '--',
                label='estimate')
        ax.set_title(label)
        ax.set_xlabel('Energy (keV)')
        ax.legend()
        ax.grid(True)
    fig.suptitle('Effective spectrum: ground truth vs estimate')
    fig.tight_layout()
    fig.savefig(f'{out}/spectra.png', dpi=130)
    print(f'total time {time.time()-t0:.0f} s; output in {out}')
