# xcal

xcal estimates the spectral response of an X-ray CT system from
calibration scans of known metal rods.  The response is modeled as the
product of the source spectrum, the filter responses, and the detector
response, and the physical parameters of each component are estimated
jointly from scans at two or three instrument settings.  Because the
estimates are physical parameters, they stay valid when the source
voltage or the filters change.

xcal is built on [mbirtorch](https://github.com/cabouman/mbirtorch).
Scans enter as a sinogram plus a tomography model, the pair produced
by mbirtorch preprocessing, so xcal works with any scanner and
geometry mbirtorch supports.

Full documentation: [xcal.readthedocs.io](https://xcal.readthedocs.io).
For the method, see the paper
[XCal: A Model-Based Approach to X-ray CT Spectral Calibration](https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-15-30875),
Optics Express, 2025.  The software release matching the paper is tag
[v0.1.0](https://github.com/cabouman/xcal/releases/tag/v0.1.0).

## Install

```bash
git clone git@github.com:cabouman/xcal.git
cd xcal
pip install .
```

This installs mbirtorch and the other dependencies automatically.
Reflection tube sources also need Spekpy: `pip install spekpy`.

## Quick look

```python
import mbirtorch.preprocess as mtp
import xcal

sino_80, model_80 = mtp.zeiss.get_sino_and_model('scan_080kV.txrm')

targets = [xcal.Target('Ti'), xcal.Target('Al')]
system = xcal.System(
    source=xcal.TransmissionSource(target_thickness=xcal.estimate(0.001, 0.007)),
    filters=[xcal.Filter(material=['Al', 'Cu'], thickness=xcal.estimate(0, 10))],
    detector=xcal.Scintillator(),
)

recon, _ = model_80.recon(sino_80)
masks = segment(recon)   # your segmentation; see the demos

cal = xcal.Calibrator(system, targets)
cal.add_scan(sino_80, model_80, masks, voltage=80)
cal_result = cal.calibrate()

cal_result.show()
est_system = cal_result.est_system
R = est_system.effective_spectrum(voltage=80)   # a function of energy in keV
```

See the [Quick Start](https://xcal.readthedocs.io) for the complete
workflow, and `demo/demo_1_multi_voltage.py` for a runnable
simulated calibration with known ground truth.

## Citation

Please cite the paper when referencing the method.

```bibtex
@article{li2025xcal,
  title = {{XCal}: model-based approach to {X}-ray {CT} spectral calibration},
  author = {Wenrui Li and K. Aditya Mohan and Venkatesh Sridhar and Xin Liu and Jean-Baptiste Forien and Joseph Bendahan and Saransh Singh and Gregery T. Buzzard and Charles A. Bouman},
  journal = {Optics Express},
  volume = {33},
  number = {15},
  pages = {30875--30896},
  year = {2025},
  doi = {10.1364/OE.566319}
}
```

Please cite the software itself when referencing this package.

```bibtex
@misc{xcal,
  title = {{X}-ray {S}pectrum {C}alibration},
  author = {Wenrui Li and K. Aditya Mohan and Venkatesh Sridhar and Xin Liu and Jean-Baptiste Forien and Joseph Bendahan and Saransh Singh and Gregery T. Buzzard and Charles A. Bouman},
  howpublished = {Software library available from \url{https://github.com/cabouman/xcal}},
  note = {Version 0.2.1},
  year = 2026
}
```

GitHub's "Cite this repository" button on the repository page generates the
paper citation from `CITATION.cff`.
