.. _CalibratorDocs:

==========
Calibrator
==========

The calibrator takes everything you have declared, the system
description, the rods, and the scans, and produces the calibrated
result.

This page has three parts: how to give the calibrator your scans,
what happens when you run it, and how to read the result.

Adding the scans
----------------

xcal does not read scanner files.  Each scan enters as two objects:
the sinogram, and the tomography model that describes the scan
geometry.  mbirtorch preprocessing produces exactly this pair from
the scanner's own files:

.. code-block:: python

    import mbirtorch.preprocess as mtp
    sino, ct_model = mtp.zeiss.get_sino_and_model('scan_080kV.txrm')

The calibrator is constructed from the two things you declared on
the :ref:`System Description <SystemDocs>` page: ``system``, the
:class:`~xcal.System` holding the source, filters, and detector,
and ``rods``, the list of :class:`~xcal.Rod` objects describing the
calibration object.  You then add each scan pair together with the
instrument settings for that scan:

.. code-block:: python

    cal = xcal.Calibrator(system, rods)
    cal.add_scan(sino, ct_model, voltage=80)

By default, a scan is assumed to contain every rod, with every
filter in the beam.  If a scan held only some rods, or only some
filters were in place, say so with the ``rods`` and ``filters``
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

    cal.add_scan(sino_low,  model_low,  filters=[si_filter])
    cal.add_scan(sino_high, model_high, filters=[si_filter, al_filter])

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

One call runs the whole pipeline:

.. code-block:: python

    result = cal.calibrate()

Internally, calibrate does four things.  It reconstructs each scan
geometry.  It segments the rods out of the reconstructions, so the
rod shapes are measured rather than assumed.  It forward projects
the segmented rods to get the path length of every ray through
every rod.  Finally, it fits the system parameters so the predicted
transmission matches the measured transmission across all scans at
once.  Candidate materials are tried exhaustively, and the
continuous parameters are fit by gradient descent within their
bounds.

.. automethod:: xcal.Calibrator.calibrate

Reading the result
------------------

The result returns data and functions; it does not plot.  The
spectral quantities come back as functions of energy, which you
evaluate at any energies in keV and plot with your own tools:

.. code-block:: python

    R = result.effective_spectrum(voltage=80)   # a function of energy
    E = np.linspace(1, 80, 320)                 # keV
    plt.plot(E, R(E))

``R`` represents a continuous spectral density in units of 1/keV.
It is zero above the source voltage and integrates to one, because
the air scan normalization makes the absolute scale unidentifiable.
The voltage may be any value in the calibrated range, not only the
scanned voltages, because the source model interpolates.

The components are available the same way.
``result.source_spectrum(voltage=80)`` returns the source density,
``result.filter_response(al_filter)`` returns one filter's
transmission (values between 0 and 1), and
``result.detector_response()`` returns the detector's relative
response.

The estimated parameters come from ``result.params``, a dictionary
with readable names, or ``result.summary()``, a printable table.

During calibration, xcal reconstructs each scan and segments the
rods.  Those intermediate images come back as numpy arrays, indexed
by scan in the order you added them: ``result.reconstruction(0)``
is the reconstructed volume of the first scan, and
``result.segmentation(0)`` labels which voxels belong to which rod.
``result.transmission_fit(0)`` returns the measured and predicted
transmission arrays for that scan, for judging how well the model
fits the data.

For review there is one display convenience, ``result.show()``: it
opens the slice viewer on the segmented rods, prints the parameter
table, and plots the spectra and the fit.  Look at the segmentation
before you trust any number: if the segmentation missed a rod,
every estimate downstream of it is wrong.

.. autoclass:: xcal.CalibrationResult

.. automethod:: xcal.CalibrationResult.show

.. automethod:: xcal.CalibrationResult.summary

.. autoproperty:: xcal.CalibrationResult.params

.. automethod:: xcal.CalibrationResult.effective_spectrum

.. automethod:: xcal.CalibrationResult.source_spectrum

.. automethod:: xcal.CalibrationResult.filter_response

.. automethod:: xcal.CalibrationResult.detector_response

.. automethod:: xcal.CalibrationResult.reconstruction

.. automethod:: xcal.CalibrationResult.segmentation

.. automethod:: xcal.CalibrationResult.transmission_fit

.. automethod:: xcal.CalibrationResult.save

.. automethod:: xcal.CalibrationResult.load
