.. _UserAPIDocs:

========
User API
========

A calibration is four steps, and the API has one part for each step:

1. **Describe** what you scanned and what the system might be, using
   :class:`~xcal.Rod` and :class:`~xcal.System` (:ref:`SystemDocs`).
2. **Add the scans** to a :class:`~xcal.Calibrator`
   (:ref:`CalibratorDocs`).
3. **Calibrate** with one call to
   :meth:`~xcal.Calibrator.calibrate`.
4. **Review** the images, parameters, and spectra in the returned
   :class:`~xcal.CalibrationResult`.

In outline, every calibration script looks like this:

.. code-block:: python

    # 1. Describe.
    rods = [xcal.Rod(material='Ti', diameter=1.0), ...]
    system = xcal.System(source=..., filters=[...], detector=...)

    # 2. Add the scans, as (sinogram, model) pairs from mbirtorch.
    cal = xcal.Calibrator(system, rods)
    cal.add_scan(sino, ct_model, voltage=80)
    ...

    # 3. Calibrate.
    result = cal.calibrate()

    # 4. Review.
    result.show()
    print(result.summary())

The named materials you can use in step 1, and how to add your own,
are listed by the materials catalog (:ref:`CatalogDocs`).

.. toctree::
   :hidden:
   :maxdepth: 2

   usr_system
   usr_calibrator
   usr_catalog
