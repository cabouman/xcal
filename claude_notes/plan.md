# Plan for xcal 2

Goal: make xcal much easier to use and understand, and move its CT
dependency from mbirjax to mbirtorch.

## 1. Preserve the current version

- Tag the current main as v0.1.0.  (Tag created locally, not yet pushed.)
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
