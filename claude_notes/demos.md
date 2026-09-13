# Demos planned for xcal 2.0

Two simulation demos.  Real data demos are parked until the
simulations are right and reconstruction from raw scans is
implemented and debugged as its own step.

## Demo 1: multi-voltage (demo_simulated_multi_voltage.py)

A laboratory scanner with a reflection tube scans four rods at three
voltages.  This is the most important demo: the common laboratory
case, the paper's headline experiment, and the quick start example.

Ground truth to recover:
- Source: reflection tube, tungsten anode, takeoff angle 13 degrees.
- Filter: Al, 3.0 mm.
- Scintillator: CsI, 0.33 mm.

Scans:
- Voltages 80, 130, 180 kV; one scan each.
- 40,000 air photons per detector element, Poisson noise.

Rods (all in every scan):
- V 0.5 mm, Ti 0.5 mm, Al 1.0 mm, Mg 1.0 mm diameter.
- Placed automatically on a circle inside the field of view.

Geometry:
- Parallel beam, 96 views over 180 degrees.
- 4 detector rows, 384 channels, 0.025 mm pixels.
- Reconstruction grid 385 x 385, 0.025 mm voxels, so the rod radii
  are 10 to 20 voxels.

Estimated by the calibration:
- Takeoff angle, bounds 5 to 45 degrees.
- Filter material from {Al, Cu}, thickness bounds 0 to 10 mm.
- Scintillator material from the 7 catalog candidates, thickness
  bounds 0.001 to 0.5 mm.
- 2 x 7 = 14 material combinations searched.

Fit settings:
- Energy grid 1.5 to 179.5 keV in 1 keV bins.
- Weights 1/transmission; Adam, rate 0.02, up to 5000 iterations.

## Demo 2: multi-filtration (demo_simulated_multi_filtration.py)

To be specified next.
