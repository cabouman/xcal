"""Demo: calibrate a simulated three-voltage scanner.

Simulates CT scans of four metal rods at 80, 130, and 180 kV with a
known reflection source, filter, and scintillator, then runs the full
xcal calibration and compares the estimates to the ground truth.

Ground truth: takeoff angle 13 degrees, 3 mm Al filter, 0.33 mm CsI
scintillator.  Runs on a laptop in a few minutes.
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import mbirtorch
import xcal

if __name__ == '__main__':
    t0 = time.time()
    out = './output/simulated_multi_voltage'
    os.makedirs(out, exist_ok=True)

    # ---------------- The truth to recover ----------------
    truth = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=13.0),
        filters=[xcal.Filter('Al', thickness=3.0)],
        detector=xcal.Scintillator('CsI', thickness=0.33),
    )
    rods = [xcal.Rod('V', 0.5), xcal.Rod('Ti', 0.5),
            xcal.Rod('Al', 1.0), xcal.Rod('Mg', 1.0)]
    voltages = [80.0, 130.0, 180.0]

    # ---------------- The simulated scanner ----------------
    # Parallel beam, 0.025 mm pixels, so the thinnest rod spans 20
    # pixels of radius and segmentation error stays small.
    def make_model():
        n_views, n_rows, n_chan = 96, 4, 384
        angles = np.linspace(0, np.pi, n_views,
                             endpoint=False).astype(np.float32)
        m = mbirtorch.ParallelBeamModel((n_views, n_rows, n_chan),
                                        angles)
        m.set_params(delta_det_channel=0.025, delta_det_row=0.025,
                     alu_unit='mm', alu_value=1.0)
        m.auto_set_recon_geometry()
        return m

    # ---------------- Simulate and calibrate ----------------
    system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=xcal.estimate(5, 45)),
        filters=[xcal.Filter(material=['Al', 'Cu'],
                             thickness=xcal.estimate(0, 10))],
        detector=xcal.Scintillator(thickness=xcal.estimate(0.001, 0.5)),
    )
    cal = xcal.Calibrator(system, rods)
    for i, v in enumerate(voltages):
        model = make_model()
        sino = xcal.simulate_scan(truth, rods, model, voltage=v,
                                  photons=40000, seed=i)
        cal.add_scan(sino, model, voltage=v)
        print(f'simulated {v:.0f} kV scan ({time.time()-t0:.0f} s)')
    result = cal.calibrate()

    print()
    print(result.summary())
    print()
    print('Ground truth: takeoff angle 13 deg; Al filter 3.0 mm; '
          'CsI scintillator 0.33 mm')

    # ---------------- Compare to the truth ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, v in zip(axes, voltages):
        E = np.linspace(1.5, v - 0.5, 4 * int(v))
        ax.plot(E, truth.effective_spectrum(voltage=v)(E),
                label='ground truth')
        ax.plot(E, result.effective_spectrum(voltage=v)(E), '--',
                label='estimate')
        ax.set_title(f'{v:.0f} kV')
        ax.set_xlabel('Energy (keV)')
        ax.legend()
        ax.grid(True)
    fig.suptitle('Effective spectrum: ground truth vs estimate')
    fig.tight_layout()
    fig.savefig(f'{out}/spectra.png', dpi=130)

    result.save(f'{out}/calibration.h5')
    print(f'total time {time.time()-t0:.0f} s; output in {out}')
    result.show()
