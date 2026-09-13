# Demos xcal 2 will support initially

Two sets: simulation demos, which generate their own data and verify
against known ground truth, and real data demos, which need
measurement files.

## Simulation demos

| Demo | Source | Scans | Status |
|---|---|---|---|
| demo_simulated_multi_voltage.py | Reflection tube (Spekpy) | 3 voltages, 4 rods in one scan | Working; recovers ground truth |
| demo_simulated_multi_filtration.py | Synchrotron (ALS spectrum) | 2 filtrations, 2 rods | Working; recovers ground truth |
| Simulated transmission source (Versa style) | Transmission tube (Geant4 table) | 3 voltages | Planned; the fit path is covered by tests, no demo script yet |

Both working demos run on a laptop CPU (about 3 minutes and 1
minute) and save figures comparing the estimated effective spectrum
to the truth.

## Real data demos

| Demo | Data needed | Data in hand? | Blocking decisions |
|---|---|---|---|
| ALS Beamline 8.3.2, two filtrations | The eight reduced HDF5 files from the paper | Yes: data/demo_xcal_data/ (1.1 GB local; canonical copy on the Purdue data depot) | Loss and weight choice (v1 fit this data with unweighted least squares); per-scan detector center offsets, which v1 hard-coded |
| Zeiss Versa, three voltages | Raw .txrm scans of a rod target | Unknown; the paper says Versa data is available from the authors on request. Does Charlie have the scans? | None beyond the data itself |
| Raw ALS data-exchange files | Original beamline files with dark and white scans | Unknown; only the reduced files are known to survive | None; would replace the reduced-file demo if found |

## Where the data lives

1. Simulated data: generated inside the demos at run time; no files.
2. Measured ALS (reduced): data/demo_xcal_data/*.h5 on this Mac,
   ignored by git.  Canonical copy:
   https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xcal_data.tgz
3. Package data shipped inside xcal itself: the materials catalog
   (xcal/data/materials.yaml), the Geant4 transmission source table
   (xcal/data/*.csv), and the NIST and ALS spectrum tables
   (xcal/chem_consts/*.h5).
