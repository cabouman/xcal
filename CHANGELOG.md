# Changelog

## v0.2.1 — 2026-09-23

- mbirtorch is now a required dependency and installs automatically with xcal.
- Python 3.11 or newer is required (mbirtorch needs it).
- The automated tests run on Python 3.11 and 3.12 and install Spekpy, so the
  full calibration pipeline test runs, not just the smaller unit tests.
- The install, build, and test scripts in dev_scripts run correctly from any
  directory and set up the conda environment reliably.
- Documentation: full description of Filter and Scintillator materials and
  densities; a preview card shown when an xcal link is shared; citation
  metadata (CITATION.cff) added and kept current automatically.

## v0.2.0 — 2026-09-15

- Rewrite of the xcal core: calibration target catalog, System model, physics,
  spectral fit, and segmentation.
- A simulation module and two runnable demos: demo_1 (multi-voltage simulated
  calibration) and demo_2 (measured ALS synchrotron data).
- System save and load; package data reorganized into physical_params and
  source_models and stored as CSV text, verified against NIST.
- A fast test suite with automated tests, and a documentation overhaul.
