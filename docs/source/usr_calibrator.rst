.. _CalibratorDocs:

==========
Calibrator
==========

The calibrator takes everything you have declared, the system
description, the targets, and the scans, and produces the calibrated
result.

This page has three parts: how to give the calibrator your scans,
what happens when you run it, and how to read the result.

Adding the scans
----------------

xcal does not read scanner files.  Each scan enters as three
objects: the sinogram, the tomography model that describes the scan
geometry, and the target masks.  mbirtorch preprocessing produces the
first two from the scanner's own files, and the
:ref:`segmentation step <SegmentDocs>` produces the masks from a
reconstruction:

.. code-block:: python

    import mbirtorch.preprocess as mtp
    sino, ct_model = mtp.zeiss.get_sino_and_model('scan_080kV.txrm')
    recon, _ = ct_model.recon(sino)
    masks = xcal.segment_targets(recon, targets, ct_model)

The calibrator is constructed from the two things you declared on
the :ref:`System Description <SystemDocs>` page: ``system``, the
:class:`~xcal.System` holding the source, filters, and detector,
and ``targets``, the list of :class:`~xcal.Target` objects.  You then add each scan together with its masks
and the instrument settings for that scan:

.. code-block:: python

    cal = xcal.Calibrator(system, targets)
    cal.add_scan(sino, ct_model, masks, voltage=80)

By default, a scan is assumed to contain every target, with every
filter in the beam.  If a scan held only some targets, or only some
filters were in place, say so with the ``targets`` and ``filters``
arguments of :meth:`~xcal.Calibrator.add_scan`.

Which filters are in the beam?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A filter is a Python variable, and everywhere you mention filters
you pass the same variables in a list.  The example below is a
synchrotron calibration with two filtrations, where every piece of
the mechanism appears.

Create each physical filter once.  The variable is the filter's
identity from here on:

.. code-block:: python

    si_filter = xcal.Filter(material='Si', thickness=xcal.estimate(0, 5))
    al_filter = xcal.Filter(material='Al', thickness=xcal.estimate(0, 10))

The system lists every filter that exists in the instrument, not
which ones are in the beam:

.. code-block:: python

    system = xcal.System(source=..., filters=[si_filter, al_filter],
                         detector=...)

Each scan states which filters were in the beam for that scan:

.. code-block:: python

    cal.add_scan(sino_low,  model_low,  masks_low,
                 filters=[si_filter])
    cal.add_scan(sino_high, model_high, masks_high,
                 filters=[si_filter, al_filter])

The shared filter objects are fitted jointly across every scan they
appear in.  After calibration, you ask for a spectrum the same way:

.. code-block:: python

    R_low  = result.effective_spectrum(filters=[si_filter])
    R_high = result.effective_spectrum(filters=[si_filter, al_filter])

There are no strings to match and no numbering to remember: the
variables connect the declaration, the scans, and the results.  Two
filters of the same material are still distinct, because they are
two variables.  And on a scanner whose filtration never changes,
you omit ``filters=`` everywhere, and the default, every filter in
the beam, is always right.

.. autoclass:: xcal.Calibrator

.. automethod:: xcal.Calibrator.add_scan

Running the calibration
-----------------------

One call runs the whole pipeline and returns two things, the
estimated system and the information about how the fit went:

.. code-block:: python

    est_system, fit_info = cal.calibrate()

Internally, calibrate does two things.  It forward projects each
scan's target masks to get the path length of every ray through every
rod.  Then it fits the system parameters so the predicted
transmission matches the measured transmission across all scans at
once.  Candidate materials are tried exhaustively, and the
continuous parameters are fit by gradient descent within their
bounds.

.. automethod:: xcal.Calibrator.calibrate

Reading the result
------------------

``est_system`` is a fully specified :class:`~xcal.System`: the same
kind of object you described the system with, but with every value
filled in.  Ask it for spectra, read its values, or reuse its parts
in a new System:

.. code-block:: python

    R = est_system.effective_spectrum(voltage=80)   # a function of energy
    E = np.linspace(1, 80, 320)                     # keV
    plt.plot(E, R(E))

    est_system.filters[0].thickness                 # a number, in mm

    new_system = xcal.System(          # different filters, same
        source=est_system.source,      # estimated source and detector
        filters=[xcal.Filter('Cu', thickness=0.5)],
        detector=est_system.detector)

``R`` represents a continuous spectral density in units of 1/keV.
It is zero above the source voltage and integrates to one, because
the air scan normalization makes the absolute scale unidentifiable.
The voltage may be any value in the calibrated range, not only the
scanned voltages, because the source model interpolates.

``fit_info`` holds everything about how the fit went.
``fit_info.parameters()`` is the full parameter table with
provenance (given, estimated with bounds, or setting), and
``fit_info.summary()`` prints it.  ``fit_info.transmission_fit(0)``
returns the measured and predicted transmission arrays for the
first scan you added, for judging how well the model fits the data.
``fit_info.candidates`` ranks every material combination by cost.
``fit_info.save(path)`` stores the calibration.

For review there is one display convenience, ``fit_info.show()``:
it prints the parameter table and plots the spectra and the fit.
The target masks are reviewed earlier, at the segmentation step,
before any fitting: if a mask is wrong, every estimate downstream
of it is wrong.

.. autoclass:: xcal.CalibrationResult

.. automethod:: xcal.CalibrationResult.show

.. automethod:: xcal.CalibrationResult.summary

.. autoproperty:: xcal.CalibrationResult.params

.. automethod:: xcal.CalibrationResult.effective_spectrum

.. automethod:: xcal.CalibrationResult.source_spectrum

.. automethod:: xcal.CalibrationResult.filter_response

.. automethod:: xcal.CalibrationResult.detector_response


.. automethod:: xcal.CalibrationResult.transmission_fit

.. automethod:: xcal.CalibrationResult.save

.. automethod:: xcal.CalibrationResult.load
