# Concept of operations for xcal 2

xcal estimates the spectral response of an X-ray CT system from
calibration scans of a known object.  The response is the product of
the source spectrum, the filter response, and the scintillator
response.  xcal estimates the physical parameters of each component
and returns the resulting spectrum.

## What the user provides

The user provides three things:

1. The calibration scans.  The user's script loads each scan with
   mbirtorch preprocessing (Zeiss, NSI, pymbir), which returns a
   sinogram and a tomography model; those pairs are what xcal takes.
   For unsupported scanners the script builds the pair from the
   object scan, the air scan, and the geometry, in a few lines.
2. The composition of the calibration object: which material each
   rod is made of.
3. A description of the system components: which facts are known,
   which are to be estimated, and which are one of several
   candidates.

## The calibration scan

The calibration object is a set of homogeneous rods of known pure
materials spanning a range of attenuation, for example Mg, Al, Ti,
and V.  The user scans this object two or three times with different
source voltages, or with different filters, keeping everything else
fixed.  The repeated scans with different settings are what make the
estimation problem well posed.  The documentation will include a page
that tells the user what target to build or buy, which materials suit
their voltage range, and how many scans to take.

## Role of mbirtorch

The design follows the mbirtorch composition pattern: preprocessing
produces a sinogram and a tomography model, and higher-level
algorithms are built from those objects (as MACE4DModel in mbirjax
is built from a ct_model).

- Scanner-specific loading stays in mbirtorch.preprocess.  In the
  user's script, one call per scan such as
  `mtp.zeiss.get_sino_and_model(file)` returns the sinogram and a
  fully configured tomography model carrying the real geometry and
  pixel size.
- xcal accepts (sinogram, model) pairs and never touches scanner
  file formats.  It works with any geometry mbirtorch supports,
  because it uses only the model's recon and forward projection
  methods.
- Inside xcal: reconstruct each scan, segment the rods, forward
  project the masks for path lengths, and fit.  The rod geometry is
  measured from the data rather than trusted from the user.  The
  user never writes a projector wrapper and never builds a forward
  matrix.
- The estimated spectrum feeds back into mbirtorch preprocessing,
  for example beam hardening correction.

For scanners without an mbirtorch loader (such as the ALS
data-exchange files today), the user's script builds the pair
directly, a few lines in the style of the nersc application script:
read the arrays, call compute_sino_transmission, and construct a
ParallelBeamModel.

## How the user describes the system

The description is a short block of plain Python at the top of the
user's script.  Each fact is stated in one of four forms:

- A plain value means the fact is known: `thickness=2.0`.
- `xcal.estimate(low, high)` means the value is estimated within
  bounds.
- A list means the value is one of several candidates, and xcal
  picks the best: `material=['Al', 'Cu']`.
- An omitted material means xcal searches the standard candidate
  list from the materials catalog.

For example:

```python
system = xcal.System(
    source=xcal.ReflectionSource(takeoff_angle=xcal.estimate(5, 45)),
    filters=[xcal.Filter(material=['Al', 'Cu'],
                         thickness=xcal.estimate(0, 10))],
    detector=xcal.Scintillator(thickness=xcal.estimate(0.001, 0.5)),
)
```

The per-scan voltage goes to `Calibrator.add_scan`, not into the
source object.  A material is a chemical formula plus a density, for
example `'Gd2O2S'`.  Any formula of elements 1 through 92 is
allowed.  The materials catalog carries densities for the elements
and the common scintillators, and defines the default candidate
lists used when a material is omitted.

## Workflow

1. The user reads the calibration scan page, builds or buys the rod
   target, and does the scans.
2. The user loads each scan with mbirtorch preprocessing and writes
   the system description and the rod compositions.
3. The user calls the estimator.
4. The user reviews the outputs: the estimated spectrum, the
   estimated parameters, and plots of measured versus predicted
   transmission.

## Outputs

The result returns the estimated parameters as a dictionary with
readable names, and the spectral quantities as functions of energy:
the user evaluates them at any energies in keV and plots them with
their own tools.  The effective spectrum is a density in 1/keV that
integrates to one.  Array-valued inputs and outputs are numpy.

## Open questions

- Whether the rod masks should use the declared diameter (current
  choice: the measurement locates the center and validates the
  diameter) or the measured radius.
- Which candidate lists to curate and what goes in them.
- What the demo dataset is.  The current measured demo uses ALS
  synchrotron files that were normalized and reconstructed offline.
  A demo matching this conops needs either raw scanner files we can
  redistribute or a simulated scan written in a supported format.
