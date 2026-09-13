"""One miniature run of the full Calibrator pipeline.

This is the integration guard: simulate a single small scan with a
known system, then run calibrate() through reconstruction,
segmentation, path lengths, and the fit, and check the recovered
thicknesses.  Materials are fixed so only one combination is fit,
keeping the runtime to tens of seconds.
"""
import numpy as np
import pytest

mbirtorch = pytest.importorskip('mbirtorch')

import xcal
from xcal import _physics, _materials
from xcal._segment import _antialiased_disk


def test_calibrate_pipeline_recovers_thicknesses():
    rng = np.random.default_rng(5)
    pitch_mm = 0.05
    n_views, n_rows, n_chan = 48, 4, 96
    voltage = 80.0

    angles = np.linspace(0, np.pi, n_views,
                         endpoint=False).astype(np.float32)

    def make_model():
        m = mbirtorch.ParallelBeamModel((n_views, n_rows, n_chan),
                                        angles, compile_mode='off')
        m.set_params(delta_det_channel=pitch_mm, delta_det_row=pitch_mm)
        m.auto_set_recon_geometry()
        m.configure_devices(devices=['cpu'])
        return m

    model = make_model()
    rows, cols, slices = model.get_params('recon_shape')
    mm = model.get_params('delta_voxel')

    rod = xcal.Rod('Ti', 1.0)
    disk = _antialiased_disk(rows, cols, (rows - 1) / 2 - 8,
                             (cols - 1) / 2 + 6, 0.5 * rod.diameter / mm)
    vol = np.zeros((rows, cols, slices), np.float32)
    vol[:, :, :] = disk[:, :, None]
    path = model.forward_project(vol)

    energies = _physics.default_energy_grid(voltage)
    src = _physics.reflection_source_table(voltage, [13.0], energies)[0]
    gt = (src
          * _physics.filter_transmission(
              _materials.resolve('Al', 'filter'), 3.0, energies)
          * _physics.scintillator_response(
              _materials.resolve('CsI', 'scintillator'), 0.33, energies))
    gt_n = gt / np.trapezoid(gt, energies)
    mu = _physics.attenuation_coefficients(rod.material, energies)
    trans = np.trapezoid(
        np.exp(-path[..., None] * mu) * gt_n, energies, axis=-1)
    counts = rng.poisson(np.clip(trans, 0, None) * 100000) / 100000
    sino = -np.log(np.clip(counts, 1e-5, None)).astype(np.float32)

    system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=13.0),
        filters=[xcal.Filter(material='Al',
                             thickness=xcal.estimate(0, 10))],
        detector=xcal.Scintillator(material='CsI',
                                   thickness=xcal.estimate(0.01, 0.5)),
    )
    cal = xcal.Calibrator(system, [rod])
    cal.add_scan(sino, make_model(), voltage=voltage)
    result = cal.calibrate(verbose=0)

    p = result.params
    assert p['filter 1 (Al) thickness (mm)'] == pytest.approx(3.0,
                                                              abs=0.5)
    assert p['detector thickness (mm)'] == pytest.approx(0.33, abs=0.12)

    R = result.effective_spectrum(voltage=voltage)
    E = np.linspace(1, voltage, 200)
    assert np.trapezoid(R(E), E) == pytest.approx(1.0, abs=1e-3)
    y, pred = result.transmission_fit(0)
    assert np.corrcoef(y, pred)[0, 1] > 0.999
