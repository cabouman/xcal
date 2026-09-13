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


def simulate_scanner(gt_system, cal_target, voltage,
                     n_views,
                     n_det_rows, n_det_channels, pixel_mm, photons,
                     seed):
    """Stand in for the scanner and its preprocessing.

    With real data, mbirtorch preprocessing reads the scanner file
    and returns a sinogram and an mbirtorch CT model.  This function
    returns the same pair for a simulated scan, plus the ground
    truth (gt) masks, which only a simulation can know.
    """
    angles = np.linspace(0, np.pi, n_views,
                         endpoint=False).astype(np.float32)
    ct_model = mbirtorch.ParallelBeamModel(
        (n_views, n_det_rows, n_det_channels), angles)
    ct_model.set_params(delta_det_channel=pixel_mm,
                        delta_det_row=pixel_mm,
                        alu_unit='mm', alu_value=1.0)
    ct_model.auto_set_recon_geometry()

    gt_masks = xcal.cylinder_masks(cal_target, ct_model)
    sino = xcal.simulate_scan(gt_system, cal_target, ct_model,
                              voltage=voltage, target_masks=gt_masks,
                              photons=photons, seed=seed)
    return sino, ct_model, gt_masks


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
    # The calibration target: the physical object that is scanned.
    # Here it is a set of rods, one per specified material.
    cal_target = [xcal.Target(m, TARGET_DIAMETER)
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

    # ---------------- Acquire the scans ----------------
    # Data collection is its own phase, separate from everything
    # after it.  A real user gets each scan's sinogram and CT model
    # from mbirtorch preprocessing of a scanner file; here the
    # scanner itself is simulated.  This phase produces one list:
    # the N scans, each holding (voltage, sinogram, ct_model,
    # gt_masks).
    scans = []
    for i, kvp in enumerate(VOLTAGES):
        sino, ct_model, gt_masks = simulate_scanner(
            gt_system, cal_target, kvp, N_VIEWS, N_DET_ROWS,
            N_DET_CHANNELS, PIXEL_MM, PHOTONS, seed=i)
        scans.append((kvp, sino, ct_model, gt_masks))
        print(f'{kvp:.0f} kV scan acquired ({time.time()-t0:.0f} s)')

    # ---------------- Get the target masks ----------------
    # The masks are the calibration's third input. In this simulation, we can use the gt_mask.
    # However, in application, the masks must be obtained by segmenting a reconstruction of the calibration target.
    masks_per_scan = []
    for kvp, sino, ct_model, gt_masks in scans:
        if USE_GROUND_TRUTH_MASKS:
            masks = gt_masks
        else:
            print(f'reconstructing the {kvp:.0f} kV scan...')
            recon, _ = ct_model.recon(sino)
            masks = xcal.segment_targets(recon, cal_target, ct_model,
                                         system=feasible_system,
                                         voltage=kvp)
            xcal.save_segmentation_plot(
                recon, cal_target, masks,
                f'{OUTPUT_DIR}/plots/segmentation_{kvp:.0f}kV.png',
                title=f'{kvp:.0f} kV reconstruction and masks')
        masks_per_scan.append(masks)

    # ---------------- Add the scans to the calibrator ----------------
    cal = xcal.Calibrator(feasible_system, cal_target)
    for (kvp, sino, ct_model, _), masks in zip(scans, masks_per_scan):
        cal.add_scan(sino, ct_model, masks, voltage=kvp)

    # -------------------- Calibrate --------------------
    # The central step of the whole demo.  The calibrator searches
    # the feasible systems for the one whose predicted transmissions
    # best match every scan, and returns the complete calibration
    # result.  Its est_system property is the estimated system with
    # every value filled in.
    cal_result = cal.calibrate()
    est_system = cal_result.est_system

    # ---------------- Report ----------------
    print()
    print(cal_result.summary())
    print()
    print(f'Ground truth: takeoff angle {GT_TAKEOFF_ANGLE} deg; '
          f'{GT_FILTER_MATERIAL} filter {GT_FILTER_THICKNESS} mm; '
          f'{GT_SCINT_MATERIAL} scintillator {GT_SCINT_THICKNESS} mm')

    # Save the whole calibration: summary.txt, the feasible and
    # estimated systems as YAML, the fit data as HDF5, and plots.
    cal_result.save(OUTPUT_DIR)

    # In a simulation the ground truth exists, so redraw the
    # spectrum plot with it for comparison.  (Paper Table 3 NRMSE:
    # 0.0017, 0.0010, 0.0008.)
    cal_result.save_plots(OUTPUT_DIR, compare_to=gt_system)

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
    cu_system.save_plot(
        f'{OUTPUT_DIR}/plots/reconfigured_spectrum.png',
        voltage=100, compare_to=est_system)

    print(f'total time {time.time()-t0:.0f} s; output in {OUTPUT_DIR}')
