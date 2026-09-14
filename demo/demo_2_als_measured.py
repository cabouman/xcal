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

import h5py
import numpy as np
import mbirtorch
import mbirtorch.preprocess as mtp
from scipy import ndimage
import xcal

# ===================== User parameters =====================

DATA_URL = ('https://www.datadepot.rcac.purdue.edu/bouman/data/'
            'demo_xcal_data.tgz')

# Calibration targets: cylinders, one material each, scanned alone.
TARGET_MATERIALS = ['V', 'Ti', 'Al', 'Mg']
TARGET_DIAMETER = 1.0       # mm

# The two filtration conditions and which filters each scan saw.
FILTRATIONS = ['low', 'high']       # low: Si only; high: Si + Al

# Scan geometry and preprocessing.
PIXEL_MM = 0.00065          # detector pixel pitch
DOWNSAMPLE = 4              # channel and view downsampling factor

# Scan metadata: detector center offset of each scan in original
# channels.  A real instrument provides these with the data.
CENTER_OFFSETS = {
    ('low', 'V'): -31, ('low', 'Ti'): -24,
    ('low', 'Mg'): -35, ('low', 'Al'): -10,
    ('high', 'V'): -68, ('high', 'Ti'): -68,
    ('high', 'Mg'): 128, ('high', 'Al'): 140,
}

# Per-scan reconstruction and segmentation parameters, chosen by
# inspecting each reconstruction (2026-09-13).  ring_snr is the
# stripe detection threshold of mbirtorch's remove_all_stripe
# (smaller is more aggressive); snr_db is the mbirtorch
# regularization parameter; the noisier the scan, the lower both
# go.  seg is the segmentation method: 'otsu' (2-level Otsu
# threshold, measures the rod's actual shape) or 'disk' (best-fit
# disk of the declared diameter, for a scan too noisy for a
# threshold to trace the boundary).
RECON_SEG_PARAMS = {
    ('low', 'V'):   {'ring_snr': 3, 'snr_db': 30, 'seg': 'otsu'},
    ('low', 'Ti'):  {'ring_snr': 3, 'snr_db': 30, 'seg': 'otsu'},
    ('low', 'Al'):  {'ring_snr': 3, 'snr_db': 25, 'seg': 'otsu'},
    ('low', 'Mg'):  {'ring_snr': 3, 'snr_db': 25, 'seg': 'otsu'},
    ('high', 'V'):  {'ring_snr': 1, 'snr_db': 20, 'seg': 'otsu'},
    ('high', 'Ti'): {'ring_snr': 1, 'snr_db': 20, 'seg': 'otsu'},
    ('high', 'Al'): {'ring_snr': 3, 'snr_db': 15, 'seg': 'otsu'},
    ('high', 'Mg'): {'ring_snr': 0, 'snr_db': 10, 'seg': 'disk'},
}

OUTPUT_DIR = './output/demo_2_als_measured'

# ===========================================================


def load_scan(data_dir, filtration, material):
    """Stand in for mbirtorch preprocessing of one scanner file.

    Reads one ALS scan file, removes ring artifacts, and returns the
    sinogram and the mbirtorch CT model with this scan's
    reconstruction parameters applied, downsampled for speed.
    """
    path = os.path.join(data_dir, f'{filtration}_fltr_{material}.h5')
    with h5py.File(path, 'r') as f:
        trans = f['data_norm'][()]          # (views, 1, channels)

    # Average transmission over channel blocks; subsample views.
    n_views, _, n_chan = trans.shape
    n_chan -= n_chan % DOWNSAMPLE
    trans = trans[:, :, :n_chan].reshape(n_views, 1, -1, DOWNSAMPLE)
    trans = trans.mean(axis=3)[::DOWNSAMPLE]
    sino = -np.log(np.clip(trans, 1e-6, None))

    # The detector has fixed per-channel gain error (about 6%),
    # which reconstructs as ring artifacts; remove it in the
    # sinogram with this scan's stripe detection threshold.
    params = RECON_SEG_PARAMS[(filtration, material)]
    sino = mtp.remove_all_stripe(sino, snr=params['ring_snr'])
    sino = sino.astype(np.float32)

    angles = -np.linspace(-0.5 * np.pi, 1.5 * np.pi, n_views,
                          endpoint=True)[::DOWNSAMPLE]
    ct_model = mbirtorch.ParallelBeamModel(sino.shape,
                                           angles.astype(np.float32))
    offset = CENTER_OFFSETS[(filtration, material)] * PIXEL_MM
    ct_model.set_params(delta_det_channel=PIXEL_MM * DOWNSAMPLE,
                        delta_det_row=PIXEL_MM * DOWNSAMPLE,
                        det_channel_offset=offset,
                        snr_db=params['snr_db'],
                        alu_unit='mm', alu_value=1.0)
    ct_model.auto_set_recon_geometry()
    return sino, ct_model


if __name__ == '__main__':
    t0 = time.time()
    os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

    # ---------------- The feasible system ----------------
    # The source is known exactly: the measured ALS spectrum with no
    # parameters.  The Si filter is in every scan; the Al filter only
    # in the high-filtration scans.  The scintillator material is
    # searched over the catalog candidates.
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

    # ---------------- Acquire the scans ----------------
    # A real user gets each scan's sinogram and CT model from
    # mbirtorch preprocessing of a scanner file; here they are read
    # from the downloaded ALS files.  This phase produces one list:
    # the 8 scans, each holding (filtration, material, sinogram,
    # ct_model).
    data_dir = mbirtorch.download_and_extract(DATA_URL, './input')
    scans = []
    for filtration in FILTRATIONS:
        for material in TARGET_MATERIALS:
            sino, ct_model = load_scan(data_dir, filtration, material)
            scans.append((filtration, material, sino, ct_model))
            print(f'{filtration} filtration, {material} rod loaded '
                  f'({time.time()-t0:.0f} s)')

    # ---------------- Get the target masks ----------------
    # The masks are the calibration's third input, and making them
    # is the application's job, not xcal's: each scan is
    # reconstructed and its rod segmented here, in demo code, with
    # mbirtorch's segmentation utility.  One rod per scan, so the
    # segmentation has two classes (background and rod) and there
    # is no class-to-material pairing to do.
    def segment_rod(recon, method, diameter_mm, mm_per_voxel):
        """Segment the single rod.  'otsu': threshold at Otsu's
        value, largest connected region, holes filled -- measures
        the actual shape.  'disk': place a disk of diameter
        diameter_mm at the position where the image is brightest
        under it (a matched filter)."""
        img = np.asarray(recon)[:, :, 0]
        if method == 'otsu':
            threshold = mtp.multi_threshold_otsu(img, classes=2)[0]
            binary = img >= threshold
            cc, n = ndimage.label(binary)
            sizes = ndimage.sum(binary, cc, range(1, n + 1))
            shape = ndimage.binary_fill_holes(
                cc == int(np.argmax(sizes)) + 1)
        else:
            radius = 0.5 * diameter_mm / mm_per_voxel
            r_k = int(round(radius))
            yk, xk = np.ogrid[-r_k:r_k + 1, -r_k:r_k + 1]
            kernel = (yk**2 + xk**2 <= r_k**2).astype(float)
            score = ndimage.convolve(img, kernel / kernel.sum(),
                                     mode='constant')
            cy, cx = np.unravel_index(np.argmax(score), score.shape)
            yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
            shape = (yy - cy)**2 + (xx - cx)**2 <= radius**2
        mask = np.zeros(np.asarray(recon).shape, dtype=np.float32)
        mask[:, :, :] = shape[:, :, None]
        return mask

    masks_per_scan = []
    for filtration, material, sino, ct_model in scans:
        print(f'reconstructing the {filtration} filtration, '
              f'{material} rod scan...')
        recon, _ = ct_model.recon(sino, print_logs=False)
        masks = [segment_rod(recon,
                             RECON_SEG_PARAMS[(filtration, material)]["seg"],
                             cal_target[material].size,
                             float(ct_model.get_params('delta_voxel')))]
        xcal.save_segmentation_plot(
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
        cal.add_scan(sino, ct_model, masks,
                     targets=[cal_target[material]], filters=filters)

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
    print('Paper (Table 9): Si 2.557 mm, Al 9.494 mm, LuAG 0.0506 mm')
    print('Nominal:         Si 2.0 mm,   Al 8.0 mm,   LuAG 0.050 mm')

    # Save the whole calibration: summary.txt, the feasible and
    # estimated systems as YAML, the fit data as HDF5, and plots.
    cal_result.save(OUTPUT_DIR)

    print(f'total time {time.time()-t0:.0f} s; output in {OUTPUT_DIR}')
