.. _QuickStart:

===========
Quick Start
===========

The script below calibrates a Zeiss Versa from three scans of a rod
target at three source voltages.  Scanner files are read by mbirtorch
preprocessing, which returns a sinogram and a geometry model per scan;
xcal never touches scanner formats.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import mbirtorch.preprocess as mtp
    import xcal

    # Preprocess each scan: scanner-specific, returns (sinogram, model).
    sino_40,  model_40  = mtp.zeiss.get_sino_and_model('scan_040kV.txrm')
    sino_80,  model_80  = mtp.zeiss.get_sino_and_model('scan_080kV.txrm')
    sino_150, model_150 = mtp.zeiss.get_sino_and_model('scan_150kV.txrm')

    # The calibration object: rods of known materials.
    targets = [
        xcal.Target(material='Ti', size=1.0),   # mm
        xcal.Target(material='Al', diameter=0.5),
        xcal.Target(material='Mg', diameter=0.5),
    ]

    # The system description.  A plain value is known; xcal.estimate
    # is fit within bounds; a list is searched; an omitted material is
    # searched over the catalog's standard candidates.
    beam_filter = xcal.Filter(material=['Al', 'Cu'],
                              thickness=xcal.estimate(0, 10))    # mm
    system = xcal.System(
        source=xcal.TransmissionSource(
            target_thickness=xcal.estimate(0.001, 0.007)),      # mm
        filters=[beam_filter],
        detector=xcal.Scintillator(
            thickness=xcal.estimate(0.001, 0.5)),               # mm
    )

    # Reconstruct each scan and segment the targets.  Look at the
    # masks before calibrating.
    cal = xcal.Calibrator(system, rods)
    for sino, model, kv in [(sino_40, model_40, 40),
                            (sino_80, model_80, 80),
                            (sino_150, model_150, 150)]:
        recon, _ = model.recon(sino)
        masks = segment(recon)   # your segmentation; see Target Masks
        cal.add_scan(sino, model, masks, voltage=kv)
    cal_result = cal.calibrate()
    est_system = cal_result.est_system

    # Review: prints the parameter table and plots the spectra and
    # the transmission fit.
    cal_result.show()

    # Save the calibration: one directory holding the summary, the
    # feasible and estimated systems as YAML, the fit data, and plots.
    cal_result.save('versa_calibration')

    # The effective spectrum at 80 kV, returned as a function of
    # energy in keV.  Evaluate and plot it however you choose.
    R = est_system.effective_spectrum(voltage=80)
    E = np.linspace(1, 80, 320)
    plt.plot(E, R(E))

The four value forms
--------------------

Every physical fact in the system description takes one of four
forms:

* a plain value means known and fixed: ``material='Al'``,
* ``xcal.estimate(low, high)`` means estimated within bounds,
* a list means candidates that xcal searches: ``material=['Al', 'Cu']``,
* an omitted material means xcal searches the standard candidate
  list from the :ref:`materials catalog <CatalogDocs>`.
