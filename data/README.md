# Data folder

Downloaded measurement data goes here.  Nothing in this folder is
tracked by git.

What we currently have:

- `demo_xcal_data/` (about 1.1 GB after download): the eight reduced
  ALS Beamline 8.3.2 files from the XCal paper, one per rod (V, Ti,
  Al, Mg) per filtration (low, high).  Each HDF5 file holds
  `data_norm`, the normalized transmission radiograph
  (2625 views, 1 row, 2560 columns), and `recon`, a precomputed
  reconstruction.  The files carry no geometry; the pixel size is
  0.00065 mm and the views span a full rotation.

  Canonical copy:
  https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xcal_data.tgz
