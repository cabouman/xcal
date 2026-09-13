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

Estimated by the calibration (bounds from the catalog):
- Takeoff angle, 5 to 45 degrees.
- Filter material from {Al, Cu}; thickness Al 0 to 10 mm,
  Cu 0 to 1 mm.
- Scintillator from the 7 catalog candidates, 0.001 to 0.5 mm.
- 14 material combinations searched.

Quality target (paper Table 3, spectrum NRMSE): about 0.0017 at low,
0.0010 at mid, 0.0008 at high voltage.

## Demo 2: ALS measured data (demo_als_measured.py)

The paper's real experiment: measured scans from ALS beamline
8.3.2, a synchrotron source with no voltage knob, so the scans
differ by filtration.  To be specified in detail after the demo 1
walk finishes; the existing script is a stale draft.
