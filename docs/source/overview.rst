========
Overview
========

What xcal does
--------------

A CT detector never records the X-ray spectrum directly.  The recorded
intensity with no object present, called the effective spectrum, is
shaped by the source, by any filters in the beam, and by the detector's
scintillator.  Quantitative CT methods such as beam hardening
correction need this effective spectrum, and it cannot be measured
directly.

xcal estimates the effective spectrum from calibration scans of known
homogeneous rods.  It models the spectrum as the product of three
physical components:

* the source spectrum, from Spekpy or Geant4 lookup tables,
* each filter's transmission, from Beer's law and NIST data,
* the scintillator's response, from NIST absorption data.

Each component has a small number of physical parameters, such as the
takeoff angle, the filter material and thickness, and the scintillator
material and thickness.  xcal estimates these parameters by fitting the
measured transmission of the rods across all scans jointly.  Material
choices are found by exhaustive search over candidates; continuous
parameters are fit by gradient descent in PyTorch.

.. figure:: figs/effective_spectrum.png
   :align: center
   :width: 95%

   The effective spectrum model.  The source spectrum passes through
   the filters and the detector, and each block is controlled by the
   user-adjustable settings (top) and by the parameters xcal
   estimates (bottom).  The output R(E) is the effective spectrum.

Why parameters instead of spectra?
----------------------------------

Because the estimated quantities are physical parameters of the
instrument, they remain valid when the instrument settings change.
After one calibration, the user can change the source voltage or swap
a known filter and compute the new effective spectrum without
recalibrating.  Estimating the spectrum bin by bin, in contrast, must
be redone for every setting.

What the user provides
----------------------

1. Two or three CT scans of the rod target at different source
   voltages or filtrations.
2. The rod materials and nominal diameters.
3. A description of the system: the source type, the possible filters,
   and the possible scintillators.  Facts that are unknown are marked
   as estimated or left as candidate lists.

See :ref:`CalibrationScan` for guidance on the scans, and
:ref:`QuickStart` for a complete script.
