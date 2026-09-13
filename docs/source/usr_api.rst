.. _UserAPIDocs:

========
User API
========

A calibration is four steps, and the API has one part for each step:

1. **Describe** what you scanned and what the system might be, using
   :class:`~xcal.Target` and :class:`~xcal.System` (:ref:`SystemDocs`).
2. **Reconstruct and segment** each scan: mbirtorch reconstructs,
   and :func:`~xcal.segment_targets` measures the target masks, which you
   inspect (:ref:`SegmentDocs`).
3. **Calibrate**: add each scan with its masks to a
   :class:`~xcal.Calibrator` and call
   :meth:`~xcal.Calibrator.calibrate` (:ref:`CalibratorDocs`).
4. **Review** the parameters and spectra in the returned
   :class:`~xcal.CalibrationResult`.

In outline, every calibration script looks like this:

.. code-block:: python

    # 1. Describe.
    targets = [xcal.Target(material='Ti', size=1.0), ...]
    system = xcal.System(source=..., filters=[...], detector=...)

    # 2. Reconstruct, segment, and add each scan.
    cal = xcal.Calibrator(system, rods)
    recon, _ = ct_model.recon(sino)
    masks = xcal.segment_targets(recon, targets, ct_model)
    cal.add_scan(sino, ct_model, masks, voltage=80)
    ...

    # 3. Calibrate.
    est_system, fit_info = cal.calibrate()

    # 4. Review.
    fit_info.show()
    print(fit_info.summary())

The named materials you can use in step 1, and how to add your own,
are listed by the materials catalog (:ref:`CatalogDocs`).

.. toctree::
   :hidden:
   :maxdepth: 2

   usr_system
   usr_calibrator
   usr_segment
   usr_catalog
   usr_simulate
