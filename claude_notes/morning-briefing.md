# Morning briefing, 2026-09-13

What happened overnight on branch xcal_2, what is verified, and what
needs your decisions.  Everything is committed locally on xcal_2 and
NOT pushed, except the first commit (the skeleton and docs), which
you approved before sleeping.

## The headline

xcal 2 is implemented and works end to end.  On a simulated
three-voltage calibration with known ground truth (reflection source
at 13 degrees takeoff, 3 mm Al filter, 0.33 mm CsI scintillator,
four rods, Poisson noise), the full pipeline (reconstruct, segment,
compute path lengths, joint fit over 14 material combinations)
recovers:

- takeoff angle 13.2 degrees (truth 13.0)
- Al filter, 3.013 mm (truth Al, 3.0)
- CsI detector, 0.329 mm (truth CsI, 0.33)

The correct material combination wins with a 65 percent cost margin
over the runner-up.  LOOK AT THE IMAGES FIRST:
claude_notes/review_figures/ holds spectra.png (estimate on top of
ground truth at all three voltages), segmentation.png, and fit.png.

The ALS-style case works too: a second simulated calibration with a
known synchrotron source and two filtrations recovers Si 2.03 mm
(truth 2.0), Al 7.99 mm (truth 8.0), and picks LuAG from the seven
scintillators at 0.046 mm (truth 0.05).  So both scan types you
asked for, multi-voltage and multi-filtration, run end to end.

Run them yourself (both verified tonight exactly as committed):
`python demo/demo_1_multi_voltage.py` (about 3 minutes) and
`python demo/demo_simulated_multi_filtration.py` (about 1 minute).
Their figures are also in claude_notes/review_figures/.

## What exists now

- xcal/catalog.py, xcal/_materials.py: the materials catalog
  (shipped YAML plus user files), name and formula resolution with
  plain-language errors.
- xcal/system.py: the description classes, fully validated.
- xcal/_physics.py: coefficient curves with the energy range guard,
  Spekpy reflection tables, the Geant4 transmission table (now
  shipped in xcal/data), the ALS spectrum, voltage interpolation.
- xcal/_segment.py: automatic rod segmentation (matched filter,
  greedy peak peel, per-rod Otsu), scale-invariant rod matching,
  sub-voxel disk masks, loud failure messages.
- xcal/_fit.py: exhaustive discrete search plus Adam on CPU.  No
  multiprocessing, no vendored L-BFGS.
- xcal/calibrator.py: the pipeline and the CalibrationResult with
  function-valued spectra, params, summary, save/load, show.
- tests/: 41 tests, about 13 seconds, all passing, including
  a miniature full-pipeline calibrate() run and per-candidate
  thickness bounds.  CI workflow added (.github/workflows).
- pyproject.toml replaces setup.py; requirements updated; docs build
  with zero warnings; README and install.rst rewritten.
- v1 modules deleted from this branch (preserved at tag v0.1.0).

## Verified against closed-form values

The recon-segment-project chain was checked on a synthetic cylinder:
reconstructed attenuation within 0.6 percent of NIST truth, path
lengths within 2 percent (scratchpad smoke test).  The same chain
was verified on a cone-beam model at magnification 2, confirming the
pipeline is geometry independent.  The fit engine recovers known
parameters on synthetic problems with and without noise, including
the transmission source's off-grid voltage and thickness
interpolation (tests/).  A pip install into a clean environment
ships the catalog and the Geant4 table correctly.

## Bugs found and fixed along the way

- v1's Interp2D clamps indices against the wrong dimensions, so v1
  reflection fits above 13 degrees takeoff used a corrupted response
  surface.  xcal 2 does not use Interp2D; noted for the record.
- v1's voltage interpolation zeroed each spectrum's own cutoff bin.
- v1's out-of-range NIST interpolation silently returns wrong
  coefficients; xcal 2 raises instead.
- New tonight: rod-to-blob matching must be scale invariant (the
  absolute log-ratio cost is permutation-blind); empty rings in the
  radial profile faked half-max crossings; one-voxel erosion was
  removing 20 percent of thin rods.

## Four Opus reviews

Four Opus agents reviewed the API, the physics port, implementation
risk, and the plan documents before implementation.  Their findings
drove the design (the exact mbirtorch units recipe, the segmentation
algorithm, dropping the process pool, the save/load identity fix).
Findings I applied are in the commits; the rest are the decisions
below.

## Decisions that are yours (in rough priority order)

1. DECIDED (2026-09-13): rod masks are the measured shapes, from
   Wenrui's segmentation approach with automatic value ranges.  The
   accuracy note moved to claude_notes/improvements.md.
2. Loss and weights are fixed to v1's transmission loss with
   1/transmission weights.  v1's ALS demo used unweighted least
   squares; if we reproduce ALS-style data, this choice matters.
3. The session metadata YAML file and the create/check-metadata
   command-line tools from metadata.md are not implemented.  Keep as
   a later layer, or drop?
4. The demo dataset question is still open: the primary demo is now
   simulated.  The ALS files could become a measured-data example
   through the generic path; raw Versa files would need permission.
5. What happens to the v1 folders still on this branch: demo/ (two
   broken v1 demos plus the new working one), examples/ (four
   notebooks referencing deleted modules), sim_data/, data/,
   dev_scripts/ (v1 install scripts).
6. result.params keys are readable sentences ('filter 1 (Si)
   thickness (mm)').  One reviewer argued for machine-friendly keys
   with units in summary() only.  I kept your approved format.
7. Smaller API gaps flagged by review, not implemented: per-scan
   valid-pixel masks beyond the automatic exclusions, a per-scan
   gain parameter for imperfect air scans, a Calibrator.check()
   preflight, an Air entry in the catalog, uncertainty reporting.

## Known limitations (documented, not hidden)

- Transmission sources interpolate a 3-voltage (40/80/150 kV) Geant4
  table; scan voltages outside that range raise.
- The reflection anode is tungsten (Spekpy).
- The Si-thickness versus scintillator-thickness direction is
  ill-conditioned and converges slowly; defaults now run 5000 Adam
  iterations per combination (roughly 5 to 10 seconds each on this
  Mac).
- ReflectionSource table generation calls Spekpy at calibrate time
  (a few seconds per scan).

## Housekeeping

- I installed two small packages into the mbirtorch conda env:
  chemparse and spekpy (both pure Python, needed by xcal).
- Untracked outputs for your review: claude_notes/review_figures/*.png
  (not committed, per the no-binaries rule).
- The scratchpad test scripts live outside the repo and vanish with
  the session.
