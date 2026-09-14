# Demos planned for xcal 2.0

Two demos, matching the paper's two experiments: the simulated
multi-voltage study (demo 1) and the measured ALS multi-filtration
data (demo 2).  Demo 2 is parked until the simulations are right
and reconstruction from raw scans is implemented and debugged as
its own step.  A simulated multi-filtration demo existed briefly as
a stand-in for demo 2 and was deleted: it demonstrated nothing the
ALS demo will not show better with real data, and the multi-filter
capability is guarded by tests/test_fit.py.

## Demo 1: multi-voltage (demo_1_multi_voltage.py)

Follows the simulated experiment of the paper (Table 2).  A
reflection tube scans four rods at three voltages.  This is the most
important demo and the quick start example.

Ground truth to recover (fixed, inside the paper's ranges):
- Source: reflection tube, tungsten anode, takeoff angle 20 degrees.
- Filter: Al, 5.0 mm.
- Scintillator: CsI, 0.25 mm.

Scans:
- Voltages 50, 100, 150 kV; one full scan each.
- 360 views over 180 degrees (the paper used 15 measurement views;
  a real user takes a full scan, and the fit uses a small subset).
- One detector row, 1024 channels, 0.005 mm pixels.
- 40,000 air photons per detector element, Poisson noise.

Calibration targets (all in every scan): cylinders of V, Ti, Al,
Mg, each 1.0 mm diameter, placed on a circle inside the field of
view.

Target masks (the third calibration input), two modes:
- Ground truth masks (default): the ideal masks the simulation used.
  Tests the fit alone, as the paper's simulated study did.
- Segmented masks: reconstruct each scan and segment the rods.
  Tests the whole measurement pipeline.
  KNOWN OPEN ISSUE (2026-09-13): the segmented mode's masks verify
  well against ground truth (0.97 to 1.04 mm, correct material
  pairing), yet the calibration from them picks the wrong filter
  material.  The contradiction is not yet diagnosed.

Estimated by the calibration (bounds from the catalog):
- Takeoff angle, 5 to 45 degrees.
- Filter material from {Al, Cu}; thickness Al 0 to 10 mm,
  Cu 0 to 1 mm.
- Scintillator from the 7 catalog candidates, 0.001 to 0.5 mm.
- 14 material combinations searched.

Quality target (paper Table 3, spectrum NRMSE): about 0.0017 at low,
0.0010 at mid, 0.0008 at high voltage.

## Demo 2: ALS measured data (demo_2_als_measured.py)

The paper's real experiment: measured scans from ALS beamline
8.3.2.  A synchrotron has no voltage knob, so the scans differ by
filtration.  Same structure and standard as demo 1; nothing is
carried over from the xcal 1 demo without earning its place.

Data (1.1 GB, auto-downloaded to data/, already on this machine):
- 8 scans: rods of V, Ti, Al, Mg, each scanned alone under low
  filtration (Si only) and high filtration (Si plus Al).
- Each scan is a full CT scan: 2625 views over 360 degrees, one
  detector row, 2560 channels, 0.65 um pixels (1.66 mm field of
  view; the 1 mm rod fills most of it).
- Files hold normalized transmission; the sinogram is its negative
  log.  Each file also carries Wenrui's reconstruction, used only
  as a reference for checking ours.

Known: the source (SynchrotronSource, the measured als_bm832
spectrum, no parameters, no per-scan voltage).

No ground truth exists.  Nominal values: Si 2.0 mm, Al 8.0 mm,
LuAG 50 um.  Quality target, the paper's estimates (Table 9):
Si 2.557 mm, Al 9.494 mm, LuAG 50.6 um.

Feasible system:
- Filter 1: Si, thickness estimated (in every scan).
- Filter 2: Al, thickness estimated (in the high-filtration scans
  only, via the per-scan filters argument of add_scan).
- Detector: scintillator searched over the catalog candidates,
  thickness estimated.

Masks: reconstruct each scan with mbirtorch (float32 on Metal;
xcal's fit stays float64 on CPU), segment with
segment_targets(..., system=feasible_system) so the matching band
comes from the synchrotron spectrum, one rod per scan, review
images from save_segmentation_plot.

Output: cal_result.save() directory, same layout as demo 1.

Open questions to settle during the build:
- Detector center offset: the xcal 1 demo hardwired one measured
  offset per scan; decide whether mbirtorch preprocessing can
  determine it instead.
- Downsampling: the draft averaged 4x over channels and views for
  a 10 minute runtime; decide full resolution or not.
- Wenrui's radiograph cleanup (outlier masking, center-window
  masking in utils.py): keep only if the fit needs it.
