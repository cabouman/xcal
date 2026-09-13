"""Demo: calibrate the ALS Beamline 8.3.2 detector from measured data.

Uses the measured dataset from the XCal paper: four rods (V, Ti, Al,
Mg), each scanned under a low filtration (Si filter only) and a high
filtration (Si plus Al).  The synchrotron source spectrum is known,
so the calibration estimates the two filter thicknesses and the
scintillator material and thickness.

The data (1.1 GB) downloads automatically on first run from
https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xcal_data.tgz

Nominal values from the paper: Si 2.0 mm, Al 8.0 mm, LuAG 50 um.
The paper's estimates (Optics Express 2025, Table 9): Si 2.557 mm,
Al 9.494 mm, LuAG 50.6 um.

Runs on a laptop CPU in about 10 minutes.
"""
import os
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
import mbirtorch
import xcal

DATA_URL = ('https://www.datadepot.rcac.purdue.edu/bouman/data/'
            'demo_xcal_data.tgz')
PIXEL_MM = 0.00065          # detector pixel pitch
DOWNSAMPLE = 4              # channel and view downsampling factor

# Per-scan detector center offsets in original channels, measured for
# this dataset (from the xcal 1 demo).
CENTER_OFFSETS = {
    ('low', 'V'): -31, ('low', 'Ti'): -24,
    ('low', 'Mg'): -35, ('low', 'Al'): -10,
    ('high', 'V'): -68, ('high', 'Ti'): -68,
    ('high', 'Mg'): 128, ('high', 'Al'): 140,
}


def load_scan(data_dir, filtration, rod_name):
    """Return (sinogram, ct_model) for one ALS scan, downsampled."""
    path = os.path.join(data_dir, f'{filtration}_fltr_{rod_name}.h5')
    with h5py.File(path, 'r') as f:
        trans = f['data_norm'][()]          # (views, 1, channels)

    # Average transmission over channel blocks and subsample views.
    n_views, _, n_chan = trans.shape
    n_chan -= n_chan % DOWNSAMPLE
    trans = trans[:, :, :n_chan].reshape(n_views, 1, -1, DOWNSAMPLE)
    trans = trans.mean(axis=3)[::DOWNSAMPLE]
    sino = -np.log(np.clip(trans, 1e-6, None)).astype(np.float32)

    angles = -np.linspace(-0.5 * np.pi, 1.5 * np.pi, n_views,
                          endpoint=True)[::DOWNSAMPLE]
    model = mbirtorch.ParallelBeamModel(sino.shape,
                                        angles.astype(np.float32))
    pitch = PIXEL_MM * DOWNSAMPLE
    offset = CENTER_OFFSETS[(filtration, rod_name)] * PIXEL_MM
    model.set_params(delta_det_channel=pitch, delta_det_row=pitch,
                     det_channel_offset=offset)
    model.auto_set_recon_geometry()
    model.configure_devices(devices=['cpu'])
    return sino, model


if __name__ == '__main__':
    t0 = time.time()
    out = './output/als_measured'
    os.makedirs(out, exist_ok=True)

    data_dir = mbirtorch.download_and_extract(DATA_URL, '../data')

    # ---------------- System description ----------------
    si_filter = xcal.Filter(material='Si',
                            thickness=xcal.estimate(0, 5), name='Si')
    al_filter = xcal.Filter(material='Al',
                            thickness=xcal.estimate(0, 10), name='Al')
    system = xcal.System(
        source=xcal.SynchrotronSource('als_bm832'),
        filters=[si_filter, al_filter],
        detector=xcal.Scintillator(thickness=xcal.estimate(0.01, 0.5)),
    )
    rods = {name: xcal.Rod(name, diameter=1.0)
            for name in ['V', 'Ti', 'Al', 'Mg']}

    # ---------------- Add the eight scans ----------------
    cal = xcal.Calibrator(system, list(rods.values()))
    for filtration, filts in [('low', [si_filter]),
                              ('high', [si_filter, al_filter])]:
        for name, rod in rods.items():
            sino, model = load_scan(data_dir, filtration, name)
            # Unweighted least squares, as in the xcal 1 ALS demo.
            cal.add_scan(sino, model, rods=[rod], filters=filts,
                         weights=np.ones_like(sino))
            print(f'added {filtration} filtration, {name} rod '
                  f'({time.time()-t0:.0f} s)')

    cal_result = cal.calibrate()

    print()
    print(cal_result.summary())
    print()
    print('Paper (Table 9): Si 2.557 mm, Al 9.494 mm, LuAG 0.0506 mm')
    print('Nominal:         Si 2.0 mm,   Al 8.0 mm,   LuAG 0.050 mm')

    # ---------------- Figures for review ----------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for si_, ax in enumerate(axes.flat):
        ax.imshow(cal_result.reconstruction(si_)[:, :, 0], origin='lower')
        labels = cal_result.segmentation(si_)[:, :, 0]
        ax.contour(labels > 0, levels=[0.5], colors='r',
                   linewidths=0.7)
        ax.set_title(f'scan {si_}')
        ax.axis('off')
    fig.suptitle('Reconstructions with segmentation outlines')
    fig.tight_layout()
    fig.savefig(f'{out}/segmentation.png', dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    energies = np.linspace(1.5, 99.5, 400)
    for filts, label in [([si_filter], 'low filtration'),
                         ([si_filter, al_filter], 'high filtration')]:
        R = cal_result.effective_spectrum(filters=filts)
        axes[0].plot(energies, R(energies), label=label)
    axes[0].set_xlabel('Energy (keV)')
    axes[0].set_ylabel('Effective spectrum (1/keV)')
    axes[0].legend()
    axes[0].grid(True)
    for si_ in range(8):
        y, pred = cal_result.transmission_fit(si_)
        axes[1].plot(y, pred, '.', markersize=1)
    axes[1].plot([0, 1], [0, 1], 'k-', linewidth=0.5)
    axes[1].set_xlabel('Measured transmission')
    axes[1].set_ylabel('Predicted transmission')
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(f'{out}/spectra_and_fit.png', dpi=130)

    cal_result.save(out)
    print(f'total time {time.time()-t0:.0f} s; output in {out}')
