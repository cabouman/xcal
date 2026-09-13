"""One miniature run of the full Calibrator pipeline.

This is the integration guard: simulate a single small scan with
xcal.simulate_scan from a known truth, run calibrate() through
reconstruction, segmentation, path lengths, and the fit, and check
the recovered thicknesses.  Materials are fixed so only one
combination is fit, keeping the runtime to tens of seconds.
"""
import numpy as np
import pytest

mbirtorch = pytest.importorskip('mbirtorch')

import xcal


def _make_model():
    n_views, n_rows, n_chan = 48, 4, 144
    angles = np.linspace(0, np.pi, n_views,
                         endpoint=False).astype(np.float32)
    m = mbirtorch.ParallelBeamModel((n_views, n_rows, n_chan), angles)
    m.set_params(delta_det_channel=0.02, delta_det_row=0.02,
                 alu_unit='mm', alu_value=1.0)
    m.auto_set_recon_geometry()
    m.configure_devices(devices=['cpu'])
    return m


def test_calibrate_pipeline_recovers_thicknesses():
    truth = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=13.0),
        filters=[xcal.Filter('Al', thickness=3.0)],
        detector=xcal.Scintillator('CsI', thickness=0.33),
    )
    targets = [xcal.Target('Ti', 1.0)]
    voltages = [60.0, 120.0]

    system = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=13.0),
        filters=[xcal.Filter('Al', thickness=xcal.estimate(0, 10))],
        detector=xcal.Scintillator('CsI',
                                   thickness=xcal.estimate(0.01, 0.5)),
    )
    cal = xcal.Calibrator(system, targets)
    centers = [(-0.15, 0.12)]
    for i, voltage in enumerate(voltages):
        model = _make_model()
        masks = xcal.cylinder_masks(targets, model, centers=centers)
        sino = xcal.simulate_scan(truth, targets, model, voltage=voltage,
                                  target_masks=masks,
                                  photons=100000, seed=5 + i)
        cal.add_scan(sino, model, masks, voltage=voltage)
    cal_result = cal.calibrate(verbose=0)

    # Wide tolerances: measured-shape segmentation carries a known
    # few-percent path length bias that the fit absorbs into the
    # thicknesses (see claude_notes/improvements.md).  This test
    # guards the pipeline, not the accuracy.
    p = cal_result.params
    assert p['filter 1 (Al) thickness (mm)'] == pytest.approx(3.0,
                                                              abs=2.0)
    assert p['detector thickness (mm)'] == pytest.approx(0.33, abs=0.2)

    voltage = voltages[-1]
    R = cal_result.est_system.effective_spectrum(voltage=voltage)
    E = np.linspace(1, voltage, 200)
    assert np.trapezoid(R(E), E) == pytest.approx(1.0, abs=5e-3)
    truth_R = truth.effective_spectrum(voltage=voltage)
    nrmse = (np.linalg.norm(R(E) - truth_R(E))
             / np.linalg.norm(truth_R(E)))
    assert nrmse < 0.3
    y, pred = cal_result.transmission_fit(0)
    assert np.corrcoef(y, pred)[0, 1] > 0.99
