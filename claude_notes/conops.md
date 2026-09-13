# Concept of operations for xcal 2

xcal estimates the spectral response of an X-ray CT system from
calibration scans of a known object.  The response is the product of
the source spectrum, the filter response, and the scintillator
response.  xcal estimates the physical parameters of each component
and returns the resulting spectrum.

## What the user provides

The user provides three things:

1. The calibration scans, in the format the scanner wrote them.
   xcal reads any format mbirtorch can read (Zeiss, NSI, pymbir).
   For unsupported scanners there is a generic path: the user
   supplies the object scan, the air scan, and the geometry as
   plain arrays.
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

mbirtorch is a dependency of xcal, not a separate step the user
runs.  xcal calls mbirtorch internally to read the scanner files,
normalize the data, reconstruct, and compute the path length of
each ray through each rod.  xcal also reconstructs and segments the
rods itself, so the rod geometry is measured from the data rather
than trusted from the user.  The user never converts data, never
writes a projector wrapper, and never builds a forward matrix.  The
estimated spectrum can then feed back into mbirtorch preprocessing,
for example beam hardening correction.

## How the user describes the system

The description is a short block of plain Python at the top of the
user's script.  Each fact is stated in one of three forms:

- A plain value means the fact is known: `voltage=80`.
- A range means the value is estimated:
  `thickness=xcal.estimate(5, low=0, high=10)`.
- A list means the value is one of several candidates, and xcal
  picks the best: `material=['Al', 'Cu']`.

For example:

```python
system = xcal.System(
    source = xcal.Source(voltage=80,
                         takeoff_angle=xcal.estimate(25, low=5, high=45)),
    filter = xcal.Filter(material=['Al', 'Cu'],
                         thickness=xcal.estimate(5, low=0, high=10)),
    scintillator = xcal.Scintillator(material=xcal.common_scintillators,
                                     thickness=xcal.estimate(0.25, low=0.01, high=0.5)),
)
```

A material is a chemical formula plus a density, for example
`'Gd2O2S'`.  Any formula covered by the NIST tables is allowed.  The
package carries a built-in density table for elements and common
compounds, and ships curated candidate lists such as
`xcal.common_scintillators` for users who do not know what is inside
their detector.

## Workflow

1. The user reads the calibration scan page, builds or buys the rod
   target, and does the scans.
2. The user points xcal at the scanner files and writes the system
   description and the rod compositions.
3. The user calls the estimator.
4. The user reviews the outputs: the estimated spectrum, the
   estimated parameters, and plots of measured versus predicted
   transmission.

## Outputs

The estimator returns the estimated spectrum as a numpy array over
energy, a dictionary of estimated parameters with stable readable
names, and a fit report the user can plot.  Entry points accept
numpy arrays and return numpy arrays.

## Open questions

- Whether mbirtorch's loaders need any additions, for example an
  option to return the transmission data before the log is taken,
  which is what the estimator uses.
- Whether the segmentation step needs any user input, such as
  approximate rod diameters, or can run fully automatically.
- Which candidate lists to curate and what goes in them.
- What the demo dataset is.  The current measured demo uses ALS
  synchrotron files that were normalized and reconstructed offline.
  A demo matching this conops needs either raw scanner files we can
  redistribute or a simulated scan written in a supported format.
