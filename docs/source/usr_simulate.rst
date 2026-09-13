.. _SimulateDocs:

==========
Simulation
==========

A simulation starts from a truth: a :class:`~xcal.System` whose facts
are all plain values.  :func:`~xcal.simulate_scan` then generates the
sinogram one scan would measure, using the same tomography model the
calibration will use, so simulation and calibration agree about the
geometry.  Its signature mirrors :meth:`~xcal.Calibrator.add_scan`.

.. code-block:: python

    truth = xcal.System(
        source=xcal.ReflectionSource(takeoff_angle=13.0),
        filters=[xcal.Filter('Al', thickness=3.0)],
        detector=xcal.Scintillator('CsI', thickness=0.33),
    )
    sino = xcal.simulate_scan(truth, targets, ct_model, voltage=80)
    masks = xcal.cylinder_masks(targets, ct_model)   # ground truth
    cal.add_scan(sino, ct_model, masks, voltage=80)

A fully specified system also states its own effective spectrum, in
the same form a calibration result uses, so a demo compares truth
and estimate directly:

.. code-block:: python

    R_true = truth.effective_spectrum(voltage=80)
    R_est = result.effective_spectrum(voltage=80)

.. autofunction:: xcal.simulate_scan

.. automethod:: xcal.System.effective_spectrum
