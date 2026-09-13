# DRAFT: proposed user script for xcal 2.  Nothing here is implemented.
# This is the API proposal for discussion.
#
# Calibrate the ALS beamline 8.3.2 detector from scans of four rods
# under two different filtrations.  The source is a synchrotron with
# a known spectrum, so only the filters and detector are estimated.
#
# The ALS files are data-exchange HDF5, read the same way as in
# mbirtorch_applications/nersc/demo_nersc.py: load the object, white,
# and dark scans, compute the sinogram, and build a parallel-beam
# model.  That loading stays outside xcal.

import numpy as np
import h5py
import mbirtorch as mt
import mbirtorch.preprocess as mtp
import xcal

def load_als_scan(filename):
    """Read one ALS data-exchange file and return (sino, ct_model)."""
    with h5py.File(filename, 'r') as f:
        pixel_size = f['/measurement/instrument/detector/pixel_size'][0]  # mm
        angles = -np.deg2rad(f['exchange/theta'])
        obj_scan = f['exchange/data'][:]
        blank_scan = f['exchange/data_white'][:]
        dark_scan = f['exchange/data_dark'][:]
    sino = mtp.compute_sino_transmission(obj_scan, blank_scan, dark_scan)
    ct_model = mt.ParallelBeamModel(sinogram_shape=sino.shape, angles=angles)
    ct_model.set_params(delta_det_channel=pixel_size)
    return sino, ct_model

# ---------- The system description ----------
# Filters are named Python variables so that each scan can say which
# filters were in the beam.  No name counters, no hidden prefixes.
si_filter = xcal.Filter(material='Si', thickness=xcal.estimate(0, 5))    # mm
al_filter = xcal.Filter(material='Al', thickness=xcal.estimate(0, 10))   # mm

system = xcal.System(
    source=xcal.SynchrotronSource(spectrum='als_bm832'),   # built-in table
    filters=[si_filter, al_filter],
    detector=xcal.Scintillator(thickness=xcal.estimate(0.01, 0.5)),  # mm
)

# ---------- The calibration object ----------
rods = {name: xcal.Rod(material=name, diameter=1.0)   # mm
        for name in ['V', 'Ti', 'Al', 'Mg']}

# ---------- Build the calibrator and add the scans ----------
# At ALS each scan holds one rod, and the two filtrations differ in
# which filters were in the beam.  Each add_scan call states both.
cal = xcal.Calibrator(system, rods=list(rods.values()))
for name, rod in rods.items():
    sino, ct_model = load_als_scan(f'low_fltr_{name}.h5')
    cal.add_scan(sino, ct_model, rods=[rod], filters=[si_filter])

    sino, ct_model = load_als_scan(f'high_fltr_{name}.h5')
    cal.add_scan(sino, ct_model, rods=[rod], filters=[si_filter, al_filter])

# ---------- Run the calibration ----------
result = cal.calibrate()

# ---------- Review the results ----------
result.show()

result.save('als_calibration.h5')
