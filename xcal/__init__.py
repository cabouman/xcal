"""xcal: X-ray CT spectral calibration.

xcal estimates the spectral response of an X-ray CT system from
calibration scans of known rods.  Scans enter as (sinogram, model)
pairs produced by mbirtorch preprocessing; xcal reconstructs, segments
the rods, computes path lengths, and jointly fits the source, filter,
and detector parameters across all scans.
"""

__version__ = '0.2.1'

from .system import (estimate, Target, Filter, Scintillator,
                     ReflectionSource, TransmissionSource,
                     SynchrotronSource, System,
                     load_system)
from .calibrator import Calibrator, CalibrationResult
from .catalog import list_materials, add_materials
from .simulate import simulate_scan
from .segment import cylinder_masks

__all__ = [
    'estimate', 'Target', 'Filter', 'Scintillator',
    'ReflectionSource', 'TransmissionSource', 'SynchrotronSource',
    'System', 'load_system', 'Calibrator', 'CalibrationResult',
    'list_materials', 'add_materials', 'simulate_scan',
    'cylinder_masks',
]
