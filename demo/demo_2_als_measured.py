"""Demo 2: calibrate ALS Beamline 8.3.2 from measured data.

The measured experiment of the XCal paper (Optics Express 2025):
four 1 mm calibration rods (V, Ti, Al, Mg), each scanned alone at
ALS Beamline 8.3.2 under two filtrations, low (Si only) and high
(Si plus Al).  A synchrotron has no voltage knob, so the scans
differ by filtration and the source spectrum is known exactly; the
calibration estimates the two filter thicknesses and the
scintillator material and thickness.

The data (1.1 GB, eight full 360-degree scans) downloads
automatically on first run.  There is no ground truth; the paper's
estimates (Table 9) are the quality target:
Si 2.557 mm, Al 9.494 mm, LuAG 0.0506 mm
(nominal: Si 2.0 mm, Al 8.0 mm, LuAG 0.050 mm).
"""
import os
import time

import numpy as np
import mbirtorch
import xcal
from demo_utils import (get_sino_and_model, segment_targets,
                        save_segmentation_plot)

# ===================== User parameters =====================

DATA_URL = ('https://www.datadepot.rcac.purdue.edu/bouman/data/'
            'demo_xcal_data.tgz')

# Calibration targets: cylinders, one material each, scanned alone.
TARGET_MATERIALS = ['V', 'Ti', 'Al', 'Mg']
TARGET_DIAMETER = 1.0       # mm

# The two filtration conditions and which filters each scan saw.
FILTRATIONS = ['low', 'high']       # low: Si only; high: Si + Al

# Scan geometry.  One full-resolution sinogram per scan; the
# reconstruction that makes the masks uses voxels
# MASK_SUBSAMPLING_FACTOR times the detector pitch, for speed.
PIXEL_MM = 0.00065              # detector pixel pitch
MASK_SUBSAMPLING_FACTOR = 4     # mask voxel / detector pitch

# Scan metadata: detector center offset of each scan in original
# channels.  A real instrument provides these with the data.
CENTER_OFFSETS = {
    ('low', 'V'): -31, ('low', 'Ti'): -24,
    ('low', 'Mg'): -35, ('low', 'Al'): -10,
    ('high', 'V'): -68, ('high', 'Ti'): -68,
    ('high', 'Mg'): 128, ('high', 'Al'): 140,
}

# Per-scan reconstruction parameters, chosen by inspecting each
# reconstruction.  snr_db is the mbirtorch regularization
# parameter.  Stripes are removed with demo_utils.remove_stripes_2d
# and every scan is segmented with the 2-level Otsu method.
RECON_PARAMS = {
    ('low', 'V'):   {'snr_db': 30},
    ('low', 'Ti'):  {'snr_db': 30},
    ('low', 'Al'):  {'snr_db': 25},
    ('low', 'Mg'):  {'snr_db': 25},
    ('high', 'V'):  {'snr_db': 20},
    ('high', 'Ti'): {'snr_db': 20},
    ('high', 'Al'): {'snr_db': 15},
    ('high', 'Mg'): {'snr_db': 10},
}

OUTPUT_DIR = './output/demo_2_als_measured'

# ===========================================================


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

    # ---------------- Define the feasible system ----------------
    # The source is known exactly.  The Si filter is in every scan
    # and the Al filter only in the high-filtration scans.
    si_filter = xcal.Filter('Si', thickness=xcal.estimate(0, 5),
                            name='Si')
    al_filter = xcal.Filter('Al', thickness=xcal.estimate(0, 15),
                            name='Al')
    feasible_system = xcal.System(
        source=xcal.SynchrotronSource(),
        filters=[si_filter, al_filter],
        detector=xcal.Scintillator(),
    )
    # The calibration target: one rod per material, scanned alone.
    cal_target = {m: xcal.Target(m, TARGET_DIAMETER)
                  for m in TARGET_MATERIALS}

    # ---------------- Acquire the CT sinograms and models ----------------
    # A real user gets each scan's sinogram and CT model from
    # mbirtorch preprocessing.  Here they are read from the
    # downloaded ALS files.
    data_dir = mbirtorch.download_and_extract(DATA_URL, './input')
    scans = []
    for filtration in FILTRATIONS:
        for material in TARGET_MATERIALS:
            params = RECON_PARAMS[(filtration, material)]
            path = os.path.join(data_dir,
                                f'{filtration}_fltr_{material}.h5')
            sino, ct_model = get_sino_and_model(
                path, CENTER_OFFSETS[(filtration, material)],
                params['snr_db'], PIXEL_MM, MASK_SUBSAMPLING_FACTOR)
            scans.append((filtration, material, sino, ct_model))
            print(f'{filtration} filtration, {material} rod loaded '
                  f'({time.time()-t0:.0f} s)')

    # ---------------- Compute the masks for the calibration targets ----------------
    # The masks identify where the calibration target is and must
    # be provided to xcal.
    masks_per_scan = []
    for filtration, material, sino, ct_model in scans:
        print(f'reconstructing the {filtration} filtration, '
              f'{material} rod scan...')
        recon, _ = ct_model.recon(sino, print_logs=False)
        masks = segment_targets(
            recon, [cal_target[material]],
            float(ct_model.get_params('delta_voxel')),
            method='otsu')
        save_segmentation_plot(
            recon, [cal_target[material]], masks,
            f'{OUTPUT_DIR}/plots/'
            f'segmentation_{filtration}_{material}.png',
            title=f'{filtration} filtration, {material} rod')
        masks_per_scan.append(masks)

    # ---------------- Add the scans to the calibrator ----------------
    cal = xcal.Calibrator(feasible_system, list(cal_target.values()))
    for (filtration, material, sino, ct_model), masks in \
            zip(scans, masks_per_scan):
        filters = ([si_filter] if filtration == 'low'
                   else [si_filter, al_filter])
        # Fit on 16 views spread over the unique half rotation.
        fit_views = np.linspace(0, sino.shape[0] // 2 - 1, 16,
                                dtype=int)
        cal.add_scan(sino, ct_model, masks,
                     targets=[cal_target[material]], filters=filters,
                     fit_views=fit_views)

    # -------------------- Calibrate --------------------
    # The calibrator estimates the unknown scanner parameters by searching over the feasible parameter set
    # for the values that minimize the reconstruction error.
    cal_result = cal.calibrate()
    est_system = cal_result.est_system

    # ---------------- Report results ----------------
    print()
    print(cal_result.summary())
    print()
    print('Paper (Table 9): Si 2.557 mm, Al 9.494 mm, LuAG 0.0506 mm')
    print('Nominal:         Si 2.0 mm,   Al 8.0 mm,   LuAG 0.050 mm')

    # Save the whole calibration as one directory.
    cal_result.save(OUTPUT_DIR)

    print(f'total time {time.time()-t0:.0f} s; output in {OUTPUT_DIR}')
