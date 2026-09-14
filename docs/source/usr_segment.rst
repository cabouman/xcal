.. _SegmentDocs:

============
Target Masks
============

.. automodule:: xcal.segment
   :no-index:

Segmenting the targets from a reconstruction is your application's
job, not xcal's: the right method depends on your scan, and you must
see and judge the result, because a wrong mask silently corrupts
every estimate downstream.  mbirtorch provides the standard tools.
The pattern the demos use, for one rod per scan:

.. code-block:: python

    import mbirtorch.preprocess as mtp
    from scipy import ndimage

    recon, _ = ct_model.recon(sino)
    img = recon[:, :, 0]
    threshold = mtp.multi_threshold_otsu(img, classes=2)[0]
    binary = img >= threshold
    cc, n = ndimage.label(binary)
    sizes = ndimage.sum(binary, cc, range(1, n + 1))
    shape = ndimage.binary_fill_holes(cc == int(np.argmax(sizes)) + 1)
    mask = np.zeros(recon.shape, np.float32)
    mask[:, :, :] = shape[:, :, None]

With N targets in one scan, use ``classes=N + 1`` and pair the
intensity classes with the targets in order of increasing
attenuation.  Order the mask list like the targets list —
``masks[k]`` belongs to ``targets[k]`` — and review the result:

.. code-block:: python

    xcal.save_segmentation_plot(recon, targets, masks, 'review.png')

.. autofunction:: xcal.cylinder_masks

.. autofunction:: xcal.save_segmentation_plot
