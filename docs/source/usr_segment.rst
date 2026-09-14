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

With N targets in one scan, the demos threshold at the image
quantile of the declared total target area, split the foreground
into connected regions, and pair regions with targets in order of
increasing material density.  Order the mask list like the targets
list: ``masks[k]`` belongs to ``targets[k]``.  Always review the
masks by overlaying them on the reconstruction; the demos share a
small plotting helper for this in ``demo/demo_utils.py``.

.. autofunction:: xcal.cylinder_masks
