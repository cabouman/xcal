# Plan for xcal 2

Goal: make xcal much easier to use and understand, and move its CT
dependency from mbirjax to mbirtorch.

STATUS (2026-09-13): steps 1 through 6 are done in first form on
branch xcal_lean: the package is implemented, tested (46 tests,
about 11 s), and documented.  Milestone reached today: demo 1 (the
paper's simulated three-voltage experiment) runs end to end both
ways — with ground-truth masks (88 s) and with reconstruction and
segmentation (210 s) — with close results: correct materials in
both; thicknesses within about 4% of truth; spectrum NRMSE
0.012-0.014 (ground-truth masks) versus 0.015-0.016 (segmented).
The line-by-line API refinement walk through demo 1 with Charlie is
ongoing.  Latest decisions: package data reorganized into
xcal/physical_params (universal physics, verified against NIST) and
xcal/source_models (one readable CSV per source model, selected by
a single name argument; users add models by adding files).
Remaining: segmentation mask width bias (see improvements.md),
system-dependent energy band for segmentation matching, demo 2
specification, a measured-data (ALS) demo, CI, and the readthedocs
cutover.

## 1. Preserve the current version

- Tag the current main as v0.1.0.  (Done: tag pushed, GitHub release published.)
- Note: setup.py says version 0.3.0 but xcal/__init__.py says 0.1.0.
  The new version will keep the version number in one place.

## 2. Design the new user API, and agree on it before writing code

- Provide one simple path for the common case: the user supplies the
  normalized radiographs, the sample materials and masks, and lists of
  candidate components.  The package returns the estimated spectrum
  and parameters.
- Generate the source spectrum table inside the package by wrapping
  spekpy.  Today the user writes about 30 lines of loops to build it.
- Compute the forward matrix using mbirtorch projectors directly.
  Today the user must write a projector wrapper class.
- Give every parameter a stable, readable name.  Today names include
  an instance counter (for example Filter_2_material), so the name
  depends on how many objects were created earlier.
- Replace the (initial, lower, upper) tuple convention with a clearer
  way to say fixed versus estimated with bounds.
- Follow the mbirtorch convention: entry points accept numpy arrays
  and return numpy arrays.

## 3. Restructure the package

- Keep the physics core: source, filter, and scintillator models, the
  NIST material constants, and the forward model.
- Simplify estimation: review the multiprocessing pool, the per-process
  logging, and the vendored L-BFGS optimizer, and keep only what earns
  its complexity.
- Decide the fate of the older dictionary-based method (dictSE.py,
  about 1200 lines): keep, archive, or drop.

## 4. Rewrite the documentation

- Short README with a quick start that runs in minutes.
- One basic demo and a small number of advanced tutorials.
- Succinct docstrings that state what each function does and how to
  call it.

## 5. Demos and tests

- One realistic end-to-end demo with mbirtorch whose plots and images
  Charlie reviews.
- A minimal, fast test suite that guards against major bugs.

## 6. Modernize the packaging

- Move to pyproject.toml with a single source for the version number.
- Add CI and keep readthedocs working.

## Order of work

Step 1 is done except for pushing the tag.  Step 2 is a design
discussion with Charlie, and no code is written until he approves it.
Steps 3 through 6 follow.

## Decisions to discuss

- Keep the package name xcal, or rename it.
- Which use cases the simple path must cover: multi-voltage scans,
  multi-filter scans, or both.
- Whether spekpy becomes a required dependency or an optional one.
- What happens to the dictionary-based method and the ALS demo.
