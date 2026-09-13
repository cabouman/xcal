# Metadata specification for xcal 2 (draft for discussion)

A calibration session is described by one metadata file.  The file
names the scan data files and states everything xcal cannot learn
from them: what the source, filters, and detector might be, and what
the calibration object is made of.  This document lists the full set
of options, proposes the file format, and says how users create and
validate these files.

## Where each kind of information lives

Three layers, from most fixed to most user-owned:

1. Physics tables.  Package data that nobody edits: the NIST
   element tables (mu_en.h5, elements 1 to 92 plus air), element
   densities and atomic weights, the ALS beamline spectrum, and the
   precomputed tungsten transmission-source tables.
2. The materials catalog.  One file shipped with the package
   listing the named materials a user can select from, with every
   parameter the software needs: name, chemical formula, density,
   and default thickness range.  It ships with the v1 and paper
   content: filter materials Al, Cu, Si; target materials V, Ti,
   Al, Mg; the seven scintillators with densities.  It also defines
   the default candidate list for each component type.  A user
   extends it with their own catalog file, which xcal reads on top
   of the shipped one.  The shipped file is never modified.
   The catalog also carries the other xcal defaults: thickness
   bounds per material, the takeoff-angle range, the energy grid
   rule, and the Spekpy lookup-table settings.
3. The user metadata.  One per calibration session, entirely
   user-written: source type, per-scan voltage, filters, detector,
   rods, scan files.  Every material name in it must resolve
   against the catalog.

The schema is the fourth piece: not data but the definition of what
fields exist, their units, and the value forms below.  It lives in
the xcal code.  The validator checks any user metadata against the
schema and the catalog before computation, and prints plain-language
errors naming the field.  The documentation page listing valid
options is generated from the same schema and catalog.

## Value conventions

Every physical quantity is written in one of three forms:

- A plain value means the fact is known and fixed: `voltage: 80`,
  `material: Al`.
- A range means xcal estimates it: `thickness: {estimate: [0, 10]}`.
- A list means it is one of several candidates and xcal searches
  over them: `material: [Al, Cu]`.

Materials have a fourth case: omitting the material entirely means
xcal searches the full default candidate list for that component
type, as defined in the materials catalog.  This matches how the
algorithm already works: a material is always found by exhaustive
search over candidates, fitting the continuous parameters once per
candidate and keeping the lowest cost.

Units are fixed by the schema: keV for voltage, mm for thickness and
diameter, degrees for angles.  The schema rejects a file that omits
a required field or uses an unknown one.

## The full option space

This list comes from the current code, checked against the Optics
Express 2025 paper (Tables 1, 2, 6, 10, 14).

### Supported materials (what the code can actually compute)

The material data lives in the package file `mu_en.h5`, which holds
the NIST mass attenuation and mass energy-absorption tables for all
92 elements from hydrogen through uranium, plus air.  Each table
covers 1 keV to 20 MeV.

- Any material is specified as a chemical formula plus a density in
  g/cm^3.  A compound formula such as `Gd2O2S` is parsed into its
  elements (the `chemparse` package), and its coefficients are the
  mass-weighted sum of the element tables.  So filters, rods, and
  scintillators all support any compound of elements 1 through 92.
- Built-in densities exist for the 92 elements and air only.  The
  code has no density table for compounds.  The seven scintillator
  densities used in the paper are typed into the demo scripts, not
  stored in the package.  xcal 2 should ship a compound density
  table covering at least the curated scintillator list.
- Outside the 1 keV to 20 MeV table range the current interpolation
  silently returns a wrong value (the mass coefficient becomes 1)
  instead of raising an error.  xcal 2 should validate the energy
  grid against the table range.
- The paper's material choices were a subset of this: filters Al,
  Cu, Si; rods V, Ti, Al, Mg; the seven scintillators of Table 14.
  Those are sensible curated defaults, not software limits.

### Source (choose one type)

1. `reflection`: an X-ray tube with a thick angled anode.
   - voltage: known per scan (user adjustable on the instrument).
   - takeoff_angle: usually estimated, range 5 to 45 degrees.
   - target material: tungsten only.  The analytical takeoff-angle
     correction in the code hard-wires tungsten (Z=74), and Spekpy
     models tungsten anodes.
   - Reference spectra come from a Spekpy lookup table over voltage
     and takeoff angle.  xcal generates this table internally.
2. `transmission`: an X-ray tube with a thin target the beam passes
   through (the Zeiss Versa case).
   - voltage: known per scan.
   - target_thickness: estimated, for example 1 to 7 micrometers.
   - Reference spectra come from a Geant4 lookup table over voltage
     and target thickness.  Geant4 cannot run at install time, so
     xcal ships precomputed tungsten tables.
3. `synchrotron`: the spectrum is known from a table and no source
   parameter is estimated.  xcal ships the ALS beamline 8.3.2
   spectrum; users can supply their own table.

### Filters (zero or more)

Each filter has:
- material: a chemical formula, a candidate list, or a curated list
  name.  Any compound of elements 1 through 92 is legal (see
  Supported materials above).  Element densities are built in;
  compound densities must be supplied until xcal 2 ships a compound
  table.
- thickness: known, or estimated with bounds.

### Detector

An energy-integrating scintillator:
- material: one formula or a candidate list.  Any compound of
  elements 1 through 92 is legal, since the response model needs
  only the attenuation and energy-absorption tables.  The curated
  list `common_scintillators` holds the seven materials from the
  paper: CsI, Gd3Al2Ga3O12, Lu3Al5O12, CdWO4, Y3Al5O12, Bi4Ge3O12,
  Gd2O2S, with their densities.
- thickness: estimated, typically 0.001 to 0.5 mm.
- An alternative tabulated detector model (the MCNP-style lookup in
  the current code) may be kept for users with simulated detector
  response tables.

### Calibration object

A list of rods, each with a material formula and a nominal diameter.
Any compound of elements 1 through 92 is legal; the demos and paper
used elemental rods (V, Ti, Al, Mg), whose densities are built in.
The diameters guide segmentation; the actual shapes are measured
from the reconstruction, not trusted from the file.

### Scans

Scan data enters xcal as a sinogram plus a tomography model, the
pair produced by mbirtorch preprocessing
(`mtp.zeiss.get_sino_and_model(...)` and its siblings).  The
sinogram is in the usual log domain; xcal recovers the transmission
internally as exp(-sino).  The documentation and the add_scan
docstring warn that preprocessing corrections (stripe or offset
removal) carry into the recovered transmission, which is normally
desirable.  File paths
and scanner formats live in the user's script at the preprocessing
call, never inside xcal.  Each scan added to the calibrator also
states its per-scan settings: the voltage, and when relevant which
rods and which filters were in the beam.  Two or three scans at
different settings are expected; that diversity is what makes the
estimation well posed.

## Proposed format: one YAML file

Example for a Zeiss Versa multi-voltage calibration:

```yaml
xcal_metadata: 1                  # format version

source:
  type: transmission
  target_thickness: {estimate: [0.001, 0.007]}   # mm

filters:
  - material: [Al, Cu]
    thickness: {estimate: [0, 10]}               # mm

detector:
  material: common_scintillators
  thickness: {estimate: [0.001, 0.5]}            # mm

calibration_object:
  rods:
    - {material: Ti, diameter: 1.0}              # mm
    - {material: Al, diameter: 0.5}
    - {material: Mg, diameter: 0.5}

scans:
  - {file: scan_040kV.txrm, format: zeiss, voltage: 40}
  - {file: scan_080kV.txrm, format: zeiss, voltage: 80}
  - {file: scan_150kV.txrm, format: zeiss, voltage: 150}
```

For supported scanner formats, the geometry, pixel size, and view
angles come from the scan files themselves through the mbirtorch
loaders, never from this file.  Only `format: arrays` adds a
geometry block (object scan, air scan, angles, pixel size in mm),
for scanners mbirtorch cannot read.  This closes the unit problem in
the current code, where pixel sizes are typed into demo scripts by
hand: attenuation coefficients are in per-millimeter units, so a
wrong pixel size corrupts every path length.

The Python API and this file share one schema.  `xcal.System` is the
in-memory form; the YAML file is the on-disk form.  Loading a file
produces the same object a user could build directly in Python.

## How users create and validate the file

1. Templates.  The documentation provides one annotated template per
   source type.  Most users copy the closest template and edit a few
   lines.
2. Assistant.  `xcal create-metadata` asks questions on the command
   line (source type, voltages, candidate filters, rods) and writes
   a valid file.  This works over ssh and needs no display.
3. Validator.  `xcal check-metadata file.yaml` checks the file
   against the schema and prints plain-language errors naming the
   exact field.  The estimator runs the same check before fitting.

A graphical form could be added later on top of the same schema.

## Where the valid configurations are defined

In one schema inside the package.  The validator, the assistant, and
a generated documentation page all read that schema, so they cannot
disagree.  The curated material lists and their densities are
package data next to the schema.

## How the scans are used

The full set of views in one scan per geometry is used to
reconstruct and segment the rods.  The spectral fit then uses a
small subset of views and slices (the paper used 16 views and the
center 5 slices for ALS, 33 views for the sparse Versa scans).  xcal
chooses the subset automatically, with an option to override.

## Open questions

- Whether the rods may be scanned separately (ALS style, one rod per
  scan) as well as together (Versa style).  Supporting both changes
  the scans section: each scan entry may need to say which rods it
  contains.
- Whether voltage is ever estimated rather than known.  The current
  code allows it; the paper always treats it as known.
- Whether to ship Geant4 transmission-source tables for materials
  other than tungsten.
- The exact name and layout of the generic `arrays` format.
