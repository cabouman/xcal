"""Demo 1: calibrate a simulated three-voltage scanner.

The setup follows the simulated experiment of the XCal paper
(Optics Express 2025, Table 2): four 1 mm cylindrical calibration
targets, a reflection tube, voltages 50, 100, and 150 kV, parallel
beam, one detector row of 1024 pixels at 0.005 mm.  The paper used
15 measurement views; here each scan is a full scan of 360 views, as
a real user would take, and the fit uses a small subset of views.

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

# ===================== User parameters =====================

# Ground truth to recover.
GT_TAKEOFF_ANGLE = 20.0     # degrees
GT_FILTER_MATERIAL = 'Al'
GT_FILTER_THICKNESS = 5.0   # mm
GT_SCINT_MATERIAL = 'CsI'
GT_SCINT_THICKNESS = 0.25   # mm

# Calibration targets: cylinders, one material each.
TARGET_MATERIALS = ['V', 'Ti', 'Al', 'Mg']
TARGET_DIAMETER = 1.0       # mm

# Scans.
VOLTAGES = [50.0, 100.0, 150.0]     # kV
PHOTONS = 40000                     # air photons per detector element

# Scan geometry (paper Table 2, with a full scan of views).
N_VIEWS = 360
N_DET_ROWS = 1
N_DET_CHANNELS = 1024
PIXEL_MM = 0.005

# True: calibrate from the ideal masks the simulation used.
# False: reconstruct each scan and segment the targets.
USE_GROUND_TRUTH_MASKS = True

OUTPUT_DIR = './output/demo_1_multi_voltage'

# ===========================================================


def make_model(n_views, n_det_rows, n_det_channels, pixel_mm):
    """Build the mbirtorch CT model for one scan: the parallel beam
    geometry, in mm units."""
    angles = np.linspace(0, np.pi, n_views,
                         endpoint=False).astype(np.float32)
    ct_model = mbirtorch.ParallelBeamModel(
        (n_views, n_det_rows, n_det_channels), angles)
    ct_model.set_params(delta_det_channel=pixel_mm,
                        delta_det_row=pixel_mm,
                        alu_unit='mm', alu_value=1.0)
    ct_model.auto_set_recon_geometry()
    return ct_model


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------- The ground truth (gt) system -------------
    gt_system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=GT_TAKEOFF_ANGLE),
        filters=[xcal.Filter(GT_FILTER_MATERIAL,
                             thickness=GT_FILTER_THICKNESS)],
        detector=xcal.Scintillator(GT_SCINT_MATERIAL,
                                   thickness=GT_SCINT_THICKNESS),
    )
    targets = [xcal.Target(m, TARGET_DIAMETER)
               for m in TARGET_MATERIALS]

    # ---------------- The feasible systems ----------------
    # The system with its unknowns marked: the set of systems the
    # calibration may choose from.  Materials and thicknesses
    # omitted: the candidates and their bounds come from the catalog
    # (Al 0 to 10 mm, Cu 0 to 1 mm; the seven scintillators, 0.001
    # to 0.5 mm).
    feasible_system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=xcal.estimate(5, 45)),
        filters=[xcal.Filter(material=['Al', 'Cu'])],
        detector=xcal.Scintillator(),
    )

    # ---------------- Simulate, get masks, calibrate ----------------
    cal = xcal.Calibrator(feasible_system, targets)
    for i, kv in enumerate(VOLTAGES):
        # One mbirtorch CT model per scan: real scans can differ in
        # alignment.
        ct_model = make_model(N_VIEWS, N_DET_ROWS, N_DET_CHANNELS,
                              PIXEL_MM)
        true_masks = xcal.cylinder_masks(targets, ct_model)
        sino = xcal.simulate_scan(gt_system, targets, ct_model,
                                  voltage=kv,
                                  target_masks=true_masks,
                                  photons=PHOTONS, seed=i)
        if USE_GROUND_TRUTH_MASKS:
            masks = true_masks
        else:
            print(f'reconstructing the {kv:.0f} kV scan...')
            recon, _ = ct_model.recon(sino)
            masks = xcal.segment_targets(recon, targets, ct_model)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(np.asarray(recon)[:, :, 0], origin='lower')
            for m in masks:
                ax.contour(m[:, :, 0] > 0.5, levels=[0.5],
                           colors='r', linewidths=0.7)
            ax.set_title(f'{kv:.0f} kV reconstruction and masks')
            fig.savefig(f'{OUTPUT_DIR}/segmentation_{kv:.0f}kV.png',
                        dpi=120)
        cal.add_scan(sino, ct_model, masks, voltage=kv)
        print(f'{kv:.0f} kV scan ready ({time.time()-t0:.0f} s)')
    est_system, fit_info = cal.calibrate()

    # ---------------- Report ----------------
    print()
    print(fit_info.summary())
    print()
    print(f'Ground truth: takeoff angle {GT_TAKEOFF_ANGLE} deg; '
          f'{GT_FILTER_MATERIAL} filter {GT_FILTER_THICKNESS} mm; '
          f'{GT_SCINT_MATERIAL} scintillator {GT_SCINT_THICKNESS} mm')

    with open(f'{OUTPUT_DIR}/summary.txt', 'w') as f:
        f.write(fit_info.summary() + '\n')
    with open(f'{OUTPUT_DIR}/parameters.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'value', 'units',
                                               'origin', 'low', 'high',
                                               'note'])
        writer.writeheader()
        writer.writerows(fit_info.parameters())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, kv in zip(axes, VOLTAGES):
        E = np.linspace(1.5, kv - 0.5, 4 * int(kv))
        gt = gt_system.effective_spectrum(voltage=kv)(E)
        est = est_system.effective_spectrum(voltage=kv)(E)
        nrmse = np.linalg.norm(est - gt) / np.linalg.norm(gt)
        ax.plot(E, gt, label='ground truth')
        ax.plot(E, est, '--', label='estimate')
        ax.set_title(f'{kv:.0f} kV,  NRMSE {nrmse:.4f}')
        ax.set_xlabel('Energy (keV)')
        ax.legend()
        ax.grid(True)
    fig.suptitle('Effective spectrum: ground truth vs estimate '
                 '(paper Table 3 NRMSE: 0.0017, 0.0010, 0.0008)')
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/spectra.png', dpi=130)

    fit_info.save(f'{OUTPUT_DIR}/calibration.h5')

    # ---------------- Reuse the estimated parts ----------------
    # The estimated components are ordinary values, so a new system
    # can mix them with a different filtration: here, the estimated
    # source and detector behind a 0.5 mm Cu filter that was never
    # scanned.
    cu_system = xcal.System(
        source=est_system.source,
        filters=[xcal.Filter('Cu', thickness=0.5)],
        detector=est_system.detector,
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    E = np.linspace(1.5, 99.5, 400)
    ax.plot(E, est_system.effective_spectrum(voltage=100)(E),
            label='estimated system (Al 5 mm)')
    ax.plot(E, cu_system.effective_spectrum(voltage=100)(E), '--',
            label='same source and detector, Cu 0.5 mm')
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Effective spectrum (1/keV)')
    ax.set_title('Reconfigured filtration at 100 kV, no recalibration')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/reconfigured_spectrum.png', dpi=130)

    print(f'total time {time.time()-t0:.0f} s; output in {OUTPUT_DIR}')
