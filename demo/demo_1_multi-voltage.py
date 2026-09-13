"""Demo 1: calibrate a simulated three-voltage scanner.

The setup follows the simulated experiment of the XCal paper
(Optics Express 2025, Table 2): four 1 mm cylindrical calibration
targets of V, Ti, Al, and Mg, a reflection tube, voltages 50, 100,
and 150 kV, parallel beam, one detector row of 1024 pixels at
0.005 mm.  The paper used 15 measurement views; here each scan is a
full scan of 360 views, as a real user would take, and the fit uses
a small subset of views.

Ground truth to recover: takeoff angle 20 degrees, Al filter 5.0 mm,
CsI scintillator 0.25 mm.

The target masks are the calibration's third input.  With
USE_GROUND_TRUTH_MASKS = True (the default), the ideal cylinder
masks the simulation used are passed to the calibration, so the demo
tests the fit alone.  With False, each scan is reconstructed and the
targets are segmented from the reconstruction, so the demo tests the
whole measurement pipeline.
"""
import csv
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import mbirtorch
import xcal

USE_GROUND_TRUTH_MASKS = True

if __name__ == '__main__':
    t0 = time.time()
    out = './output/simulated_multi_voltage'
    os.makedirs(out, exist_ok=True)

    # ---------------- The truth to recover ----------------
    truth = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=20.0),
        filters=[xcal.Filter('Al', thickness=5.0)],
        detector=xcal.Scintillator('CsI', thickness=0.25),
    )
    targets = [xcal.Target('V', 1.0), xcal.Target('Ti', 1.0),
               xcal.Target('Al', 1.0), xcal.Target('Mg', 1.0)]
    voltages = [50.0, 100.0, 150.0]

    # ---------------- The simulated scanner ----------------
    # Paper Table 2 geometry, with a full scan of views.  One model
    # per scan, because real scans can differ in alignment.
    def make_model():
        n_views, n_rows, n_chan = 360, 1, 1024
        angles = np.linspace(0, np.pi, n_views,
                             endpoint=False).astype(np.float32)
        m = mbirtorch.ParallelBeamModel((n_views, n_rows, n_chan),
                                        angles)
        m.set_params(delta_det_channel=0.005, delta_det_row=0.005,
                     alu_unit='mm', alu_value=1.0)
        m.auto_set_recon_geometry()
        return m

    # ---------------- The unknowns to estimate ----------------
    # Materials and thicknesses omitted: the candidates and their
    # bounds come from the catalog (Al 0 to 10 mm, Cu 0 to 1 mm; the
    # seven scintillators, 0.001 to 0.5 mm).
    unknown_system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=xcal.estimate(5, 45)),
        filters=[xcal.Filter(material=['Al', 'Cu'])],
        detector=xcal.Scintillator(),
    )

    # ---------------- Simulate, get masks, calibrate ----------------
    cal = xcal.Calibrator(unknown_system, targets)
    for i, v in enumerate(voltages):
        model = make_model()
        true_masks = xcal.cylinder_masks(targets, model)
        sino = xcal.simulate_scan(truth, targets, model, voltage=v,
                                  target_masks=true_masks,
                                  photons=40000, seed=i)
        if USE_GROUND_TRUTH_MASKS:
            masks = true_masks
        else:
            print(f'reconstructing the {v:.0f} kV scan...')
            recon, _ = model.recon(sino)
            masks = xcal.segment_targets(recon, targets, model)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(np.asarray(recon)[:, :, 0], origin='lower')
            for m in masks:
                ax.contour(m[:, :, 0] > 0.5, levels=[0.5],
                           colors='r', linewidths=0.7)
            ax.set_title(f'{v:.0f} kV reconstruction and masks')
            fig.savefig(f'{out}/segmentation_{v:.0f}kV.png', dpi=120)
        cal.add_scan(sino, model, masks, voltage=v)
        print(f'{v:.0f} kV scan ready ({time.time()-t0:.0f} s)')
    result = cal.calibrate()

    # ---------------- Report ----------------
    print()
    print(result.summary())
    print()
    print('Ground truth: takeoff angle 20 deg; Al filter 5.0 mm; '
          'CsI scintillator 0.25 mm')

    with open(f'{out}/summary.txt', 'w') as f:
        f.write(result.summary() + '\n')
    with open(f'{out}/parameters.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'value', 'units',
                                               'origin', 'low', 'high',
                                               'note'])
        writer.writeheader()
        writer.writerows(result.parameters())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, v in zip(axes, voltages):
        E = np.linspace(1.5, v - 0.5, 4 * int(v))
        gt = truth.effective_spectrum(voltage=v)(E)
        est = result.effective_spectrum(voltage=v)(E)
        nrmse = np.linalg.norm(est - gt) / np.linalg.norm(gt)
        ax.plot(E, gt, label='ground truth')
        ax.plot(E, est, '--', label='estimate')
        ax.set_title(f'{v:.0f} kV,  NRMSE {nrmse:.4f}')
        ax.set_xlabel('Energy (keV)')
        ax.legend()
        ax.grid(True)
    fig.suptitle('Effective spectrum: ground truth vs estimate '
                 '(paper Table 3 NRMSE: 0.0017, 0.0010, 0.0008)')
    fig.tight_layout()
    fig.savefig(f'{out}/spectra.png', dpi=130)

    result.save(f'{out}/calibration.h5')
    print(f'total time {time.time()-t0:.0f} s; output in {out}')
