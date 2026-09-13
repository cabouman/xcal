# DRAFT: proposed user script for xcal 2.  Nothing here is implemented.
# This is the API proposal for discussion.
#
# Calibrate a Zeiss Versa from three scans of a rod target at three
# source voltages.
#
# Modularity: scanner-specific loading stays in mbirtorch.preprocess,
# which returns a sinogram and a tomography model for each scan.
# xcal accepts those pairs and never touches scanner file formats.
# Any geometry mbirtorch supports (parallel, cone, ...) works, since
# xcal only uses the model's recon and forward projection methods.

import numpy as np
import matplotlib.pyplot as plt
import mbirtorch.preprocess as mtp
import xcal

# ---------- Preprocess each scan: scanner-specific ----------
# get_sino_and_model reads the scanner file and returns the sinogram
# and a fully configured cone-beam model with the real geometry.
sino_40,  model_40  = mtp.zeiss.get_sino_and_model('scan_040kV.txrm')
sino_80,  model_80  = mtp.zeiss.get_sino_and_model('scan_080kV.txrm')
sino_150, model_150 = mtp.zeiss.get_sino_and_model('scan_150kV.txrm')

# ---------- The calibration object ----------
# Rods of known pure materials.  Materials come from the catalog.
# Diameters are nominal; xcal measures the true shapes by
# reconstructing and segmenting.
rods = [
    xcal.Rod(material='Ti', diameter=1.0),   # mm
    xcal.Rod(material='Al', diameter=0.5),
    xcal.Rod(material='Mg', diameter=0.5),
]

# ---------- The system description ----------
# Three forms for every fact:
#   a plain value        -> known, fixed
#   xcal.estimate(lo,hi) -> estimated within bounds
#   a list               -> candidates, xcal picks the best
#   omitted material     -> xcal searches the catalog's standard list
system = xcal.System(
    source=xcal.TransmissionSource(
        target_thickness=xcal.estimate(0.001, 0.007)),   # mm
    filters=[
        xcal.Filter(material=['Al', 'Cu'],
                    thickness=xcal.estimate(0, 10)),      # mm
    ],
    detector=xcal.Scintillator(
        thickness=xcal.estimate(0.001, 0.5)),             # mm
    # detector material omitted: xcal searches the seven standard
    # scintillators in the catalog.
)

# ---------- Build the calibrator and add the scans ----------
# Each scan is a (sinogram, model) pair plus its instrument setting.
# xcal recovers the transmission as exp(-sino).  Note: if the
# preprocessing applied corrections (stripe or offset removal), the
# transmission reflects the corrected data, not the raw ratio of
# object scan to air scan.  That is usually what you want.
cal = xcal.Calibrator(system, rods)
cal.add_scan(sino_40,  model_40,  voltage=40)    # kV
cal.add_scan(sino_80,  model_80,  voltage=80)
cal.add_scan(sino_150, model_150, voltage=150)

# ---------- Run the calibration ----------
# Internally: reconstruct with each model, segment the rods, forward
# project the masks for path lengths, then jointly fit all scans.
result = cal.calibrate()

# ---------- Review the results ----------
# Opens the slice viewer on the segmentation, prints the parameter
# table, and opens the spectrum and fit plots.
result.show()

# ---------- Save and use ----------
result.save('versa_calibration.h5')

# The estimated responses come back as functions of energy in keV.
# R is the effective spectral density: zero above the voltage,
# integrates to one.  Evaluate it anywhere and plot as you choose.
R = result.effective_spectrum(voltage=80)
E = np.linspace(1, 80, 320)
plt.plot(E, R(E))
