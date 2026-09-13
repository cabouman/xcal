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
    rods = [
        xcal.Rod(material='Ti', diameter=1.0),   # mm
        xcal.Rod(material='Al', diameter=0.5),
        xcal.Rod(material='Mg', diameter=0.5),
    ]

    # The system description.  A plain value is known; xcal.estimate
    # is fit within bounds; a list is searched; an omitted material is
    # searched over the catalog's standard candidates.
    system = xcal.System(
        source=xcal.TransmissionSource(
            target_thickness=xcal.estimate(0.001, 0.007)),      # mm
        filters=[
            xcal.Filter(material=['Al', 'Cu'],
                        thickness=xcal.estimate(0, 10)),        # mm
        ],
        detector=xcal.Scintillator(
            thickness=xcal.estimate(0.001, 0.5)),               # mm
    )

    # Add the scans and calibrate.
    cal = xcal.Calibrator(system, rods)
    cal.add_scan(sino_40,  model_40,  voltage=40)    # kV
    cal.add_scan(sino_80,  model_80,  voltage=80)
    cal.add_scan(sino_150, model_150, voltage=150)
    result = cal.calibrate()

    # Review: opens the slice viewer on the segmentation, prints the
    # parameter table, and opens the spectrum and fit plots.
    result.show()

    # Save the calibration for later use.
    result.save('versa_calibration.h5')

    # The effective spectrum at 80 kV, returned as a function of
    # energy in keV.  Evaluate and plot it however you choose.
    R = result.effective_spectrum(voltage=80)
    E = np.linspace(1, 80, 320)
    plt.plot(E, R(E))

The three value forms
---------------------

Every physical fact in the system description takes one of three
forms:

* a plain value means known and fixed: ``material='Al'``,
* ``xcal.estimate(low, high)`` means estimated within bounds,
* a list means candidates that xcal searches: ``material=['Al', 'Cu']``.

A material may also be omitted entirely, which searches the standard
candidate list from the :ref:`materials catalog <CatalogDocs>`.
