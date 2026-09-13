.. _SegmentDocs:

============
Target Masks
============

.. automodule:: xcal.segment
   :no-index:

Look at the masks before calibrating.  Overlay them on the
reconstruction and check every target's shape:

.. code-block:: python

    recon, _ = ct_model.recon(sino)
    masks = xcal.segment_targets(recon, targets, ct_model)

    import matplotlib.pyplot as plt
    plt.imshow(recon[:, :, 0], origin='lower')
    for m in masks:
        plt.contour(m[:, :, 0] > 0.5, levels=[0.5], colors='r')
    plt.show()

.. autofunction:: xcal.segment_targets

.. autofunction:: xcal.cylinder_masks
