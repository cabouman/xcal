xcal: X-ray CT Spectral Calibration
===================================

xcal estimates the spectral response of an X-ray CT system from
calibration scans of known rods.  The response is modeled as the
product of the source spectrum, the filter responses, and the detector
response, and the physical parameters of each component are estimated
jointly from scans at two or three instrument settings.

xcal is built on `mbirtorch <https://github.com/cabouman/mbirtorch>`_.
Scans enter as a sinogram plus a tomography model, the pair produced by
mbirtorch preprocessing, so xcal works with any scanner and geometry
mbirtorch supports.

For details of the method, see the
`XCal paper <https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-15-30875>`_
in Optics Express.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   overview
   install
   calibration_scan
   quick_start
   usr_api
   credits
