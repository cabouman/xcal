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
import os
import time

import numpy as np
import mbirtorch
import xcal
from demo_utils import (simulate_scanner, segment_targets,
                        save_segmentation_plot)

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


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

    # ------------- The ground truth (gt) system -------------
    gt_system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=GT_TAKEOFF_ANGLE),
        filters=[xcal.Filter(GT_FILTER_MATERIAL,
                             thickness=GT_FILTER_THICKNESS)],
        detector=xcal.Scintillator(GT_SCINT_MATERIAL,
                                   thickness=GT_SCINT_THICKNESS),
    )
    # The calibration target: a set of rods, one per material.
    cal_target = [xcal.Target(m, TARGET_DIAMETER)
                          for m in TARGET_MATERIALS]

    # ---------------- The feasible system ----------------
    # The system with its unknowns marked.  Omitted materials and
    # thicknesses get their candidates and bounds from the catalog.
    feasible_system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=xcal.estimate(5, 45)),
        filters=[xcal.Filter(material=['Al', 'Cu'])],
        detector=xcal.Scintillator(),
    )

    # ---------------- Acquire the scans ----------------
    # A real user gets each scan's sinogram and CT model from
    # mbirtorch preprocessing.  Here the scanner is simulated.
    scans = []
    for i, kvp in enumerate(VOLTAGES):
        sino, ct_model, gt_masks = simulate_scanner(
            gt_system, cal_target, kvp, N_VIEWS, N_DET_ROWS,
            N_DET_CHANNELS, PIXEL_MM, PHOTONS, seed=i)
        scans.append((kvp, sino, ct_model, gt_masks))
        print(f'{kvp:.0f} kV scan acquired ({time.time()-t0:.0f} s)')

    # ---------------- Get the target masks ----------------
    # The masks identify where the calibration target is and must
    # be provided to xcal.
    masks_per_scan = []
    for kvp, sino, ct_model, gt_masks in scans:
        if USE_GROUND_TRUTH_MASKS:
            masks = gt_masks
        else:
            print(f'reconstructing the {kvp:.0f} kV scan...')
            recon, _ = ct_model.recon(sino)
            masks = segment_targets(
                recon, cal_target,
                float(ct_model.get_params('delta_voxel')))
            save_segmentation_plot(
                recon, cal_target, masks,
                f'{OUTPUT_DIR}/plots/segmentation_{kvp:.0f}kV.png',
                title=f'{kvp:.0f} kV reconstruction and masks')
        masks_per_scan.append(masks)

    # ---------------- Add the scans to the calibrator ----------------
    cal = xcal.Calibrator(feasible_system, cal_target)
    for (kvp, sino, ct_model, _), masks in zip(scans, masks_per_scan):
        cal.add_scan(sino, ct_model, masks, voltage=kvp)

    # -------------------- Calibrate --------------------
    # The calibrator estimates the unknown scanner parameters by searching over the feasible parameter set
    # for the values that minimize the reconstruction error.
    cal_result = cal.calibrate()
    est_system = cal_result.est_system

    # ---------------- Report ----------------
    print()
    print(cal_result.summary())
    print()
    print(f'Ground truth: takeoff angle {GT_TAKEOFF_ANGLE} deg; '
          f'{GT_FILTER_MATERIAL} filter {GT_FILTER_THICKNESS} mm; '
          f'{GT_SCINT_MATERIAL} scintillator {GT_SCINT_THICKNESS} mm')

    # Save the whole calibration as one directory.
    cal_result.save(OUTPUT_DIR)

    # The ground truth exists in simulation, so redraw the spectrum
    # plot with it.  (Paper Table 3 NRMSE: 0.0017, 0.0010, 0.0008.)
    cal_result.save_plots(OUTPUT_DIR, compare_to=gt_system)

    # ---------------- Reuse the estimated parts ----------------
    # This section illustrates how estimated parameters can be combined with new pararmeters
    # to determine the spectral response for a new system.
    cu_system = xcal.System(
        source=est_system.source,
        filters=[xcal.Filter('Cu', thickness=0.5)],
        detector=est_system.detector,
    )
    cu_system.save_plot(
        f'{OUTPUT_DIR}/plots/reconfigured_spectrum.png',
        voltage=100, compare_to=est_system)

    print(f'total time {time.time()-t0:.0f} s; output in {OUTPUT_DIR}')
