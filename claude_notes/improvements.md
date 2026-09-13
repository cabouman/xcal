# Improvements list

Things noticed and deliberately deferred.  Not scheduled.

- Segmentation accuracy: measured rod shapes run a few percent wide
  on simulated polychromatic data, and the fit absorbs the surplus
  into the estimated thicknesses.  Deciding whether and how to
  improve this needs the reconstructions, the segmentations, and the
  data and reconstruction parameters in front of Charlie.
- Per-scan valid-pixel masks, a per-scan gain parameter, a
  Calibrator preflight check, an Air catalog entry, and uncertainty
  reporting (from the design reviews).
- The saved calibration file and the beam hardening handoff to
  mbirtorch preprocessing are not yet connected.
