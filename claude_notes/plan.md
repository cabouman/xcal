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

## 1. Preserve the current version — DONE

Tag v0.1.0 pushed and a GitHub release published, pinning the
version that matches the Optics Express 2025 paper.

## 2. Design the new user API — DONE

Designed with Charlie through demo scripts before implementation.
The result: a three-step workflow (reconstruct with mbirtorch,
segment_targets for masks, Calibrator.add_scan then calibrate);
System/Filter/Scintillator/Target objects with plain values for
givens and xcal.estimate(low, high) for unknowns; scans enter as a
sinogram plus an mbirtorch CT model; results return as est_system
(a fully specified System) plus fit_info; every parameter has a
stable readable name.  Refinement continues in the demo 1 walk.

## 3. Restructure the package — DONE

Rewritten from scratch on xcal_lean.  The dictionary-based method,
the vendored optimizer, the multiprocessing pool, and all other
unused v1 material are gone (the old branch keeps everything).
Estimation is a single torch Adam fit with exhaustive search over
discrete material candidates.

## 4. Rewrite the documentation — DONE

Short README with a quick start; sphinx_book_theme docs (overview,
install, calibration scan, quick start, user API pages); docstrings
state valid values for every argument.  Builds with no warnings.
Note: sphinx-build is not on the default PATH; use the mbirtorch
env's sphinx and rebuild from clean after docstring changes.

## 5. Demos and tests — demo 1 DONE, more to come

demo/demo_1_multi_voltage.py reproduces the paper's simulated
experiment and is verified both with ground-truth masks and with
reconstruction plus segmentation.  Tests: 46, about 11 s, guarding
the pipeline rather than exact accuracy.  Remaining: demo 2 (to be
specified after the demo 1 walk finishes) and a measured-data (ALS)
demo; see demos.md.

## 6. Modernize the packaging — mostly DONE

pyproject.toml with the version in one place (xcal/__init__.py);
spekpy and docs/test extras are optional dependencies.  Remaining:
CI and the readthedocs cutover.
