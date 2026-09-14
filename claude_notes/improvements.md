# Improvements list

Things noticed and deliberately deferred.  Not scheduled.

- RESOLVED 2026-09-13: measured rod shapes ran a few percent wide.
  The cause was thresholding all rods at one global value.  The demo
  segmentation now re-measures each rod's boundary with a local
  2-level Otsu threshold (demo/demo_utils.py), and every mask comes
  out at the declared 1.00 mm with 0.99 overlap against ground
  truth.
- Per-scan valid-pixel masks, a per-scan gain parameter, a
  Calibrator preflight check, an Air catalog entry, and uncertainty
  reporting (from the design reviews).
- The saved calibration file and the beam hardening handoff to
  mbirtorch preprocessing are not yet connected.
