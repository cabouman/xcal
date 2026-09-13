"""Classes that describe the X-ray system and the calibration object.

A user builds a :class:`System` from a source, a list of filters, and a
detector, and describes the calibration object as a list of :class:`Rod`
objects.  Every physical fact is stated in one of three forms:

* a plain value means the fact is known and fixed,
* :class:`estimate` means xcal estimates it within bounds,
* a list of names means xcal searches the candidates and picks the best.

A material argument may also be omitted, in which case xcal searches the
standard candidate list from the materials catalog.
"""

import numpy as np

from . import catalog
from . import _materials

__all__ = ['estimate', 'Rod', 'Filter', 'Scintillator', 'ReflectionSource',
           'TransmissionSource', 'SynchrotronSource', 'System']


class estimate:
    """Mark a parameter as estimated within bounds.

    Args:
        low (float): Lower bound.
        high (float): Upper bound.
        initial (float, optional): Starting value for the optimization.
            Defaults to the midpoint of the bounds.

    Example:
        >>> xcal.Filter(material='Al', thickness=xcal.estimate(0, 10))
    """

    def __init__(self, low, high, initial=None):
        low = float(low)
        high = float(high)
        if not low < high:
            raise ValueError(
                f"estimate bounds must satisfy low < high, got "
                f"low={low}, high={high}.")
        if initial is None:
            initial = 0.5 * (low + high)
        initial = float(initial)
        if not low <= initial <= high:
            raise ValueError(
                f"estimate initial value {initial} lies outside the "
                f"bounds [{low}, {high}].")
        self.low = low
        self.high = high
        self.initial = initial

    def __repr__(self):
        return f"estimate({self.low}, {self.high}, initial={self.initial})"


def _check_scalar_or_estimate(value, what):
    """Validate a continuous parameter: a number or an estimate."""
    if isinstance(value, estimate):
        return value
    if isinstance(value, (int, float)):
        value = float(value)
        if value <= 0:
            raise ValueError(f"{what} must be positive, got {value}.")
        return value
    raise TypeError(
        f"{what} must be a number or xcal.estimate(low, high), got "
        f"{value!r}.")


def _resolve_candidates(material, kind, density, context):
    """Turn a material argument (None, str, or list) into a list of
    Material candidates."""
    if material is None:
        entries = catalog._default_candidates(kind)
        if not entries:
            raise ValueError(
                f"{context}: the materials catalog has no default "
                f"{kind} materials to search; name a material instead.")
        return [_materials.Material(e['name'], e['formula'],
                                    float(e['density'])) for e in entries]
    if isinstance(material, str):
        return [_materials.resolve(material, kind, density, context)]
    if isinstance(material, (list, tuple)):
        if len(material) == 0:
            raise ValueError(f"{context}: the material candidate list is "
                             f"empty.")
        return [_materials.resolve(m, kind, None, context)
                for m in material]
    raise TypeError(
        f"{context}: material must be a name, a list of names, or None, "
        f"got {material!r}.")


def _default_thickness_estimate(kind, materials):
    """Build a thickness estimate from the catalog's ranges for these
    candidate materials."""
    lows, highs = [], []
    for mat in materials:
        entry = catalog._find(kind, mat.name)
        rng = (entry or {}).get('thickness_range')
        if rng is not None:
            lows.append(float(rng[0]))
            highs.append(float(rng[1]))
    if not lows:
        raise ValueError(
            f"thickness was omitted and the catalog has no default "
            f"thickness range for {[m.name for m in materials]}; pass "
            f"thickness= as a number or xcal.estimate(low, high).")
    return estimate(min(lows), max(highs))


class Rod:
    """One homogeneous rod in the calibration object.

    Args:
        material (str): The rod material: a catalog name or any
            chemical formula of elements 1 through 92, e.g. 'Ti'.
        diameter (float): Rod diameter in mm, a manufactured dimension
            the user knows.  The segmentation locates each rod,
            validates this diameter against the reconstruction, and
            uses it for the path length masks.
        density (float, optional): Density in g/cm^3.  Required only for
            compound formulas whose density is not in the catalog.
    """

    def __init__(self, material, diameter, density=None):
        self.material = _materials.resolve(material, 'rod', density,
                                           context='Rod')
        diameter = float(diameter)
        if diameter <= 0:
            raise ValueError(f"Rod diameter must be positive mm, got "
                             f"{diameter}.")
        self.diameter = diameter

    def __repr__(self):
        return (f"Rod(material='{self.material.name}', "
                f"diameter={self.diameter})")


class Filter:
    """A beam filter modeled by Beer's law.

    Args:
        material (str or list, optional): Chemical formula, a list of
            candidate formulas, or omitted to search the catalog's
            standard filter materials.
        thickness (float or estimate, optional): Thickness in mm, known
            or estimated.  Omitted, the catalog's default thickness
            range for the candidate materials is used.
        name (str, optional): Label used in results, for example in
            the keys of ``result.params``.  Defaults to the filter's
            position, 'filter 1', 'filter 2', and so on.
        density (float, optional): Density in g/cm^3, needed only when
            the material is a compound formula that is not in the
            catalog.

    Example:
        >>> al = xcal.Filter(material=['Al', 'Cu'], thickness=xcal.estimate(0, 10))
    """

    _kind = 'filter'

    def __init__(self, material=None, thickness=None, name=None,
                 density=None):
        self.materials = _resolve_candidates(material, self._kind, density,
                                             context=type(self).__name__)
        if thickness is None:
            thickness = _default_thickness_estimate(self._kind,
                                                    self.materials)
        self.thickness = _check_scalar_or_estimate(thickness, 'thickness')
        self.name = name

    def __repr__(self):
        label = f", name='{self.name}'" if self.name else ""
        return (f"{type(self).__name__}(materials="
                f"{[m.name for m in self.materials]}, "
                f"thickness={self.thickness}{label})")


class Scintillator(Filter):
    """An energy-integrating scintillated detector.

    The response is the scintillator's absorption efficiency times the
    deposited photon energy, computed from the NIST attenuation and
    energy-absorption tables.

    Args:
        material (str or list, optional): Chemical formula, a list of
            candidates, or omitted to search the catalog's standard
            scintillators.
        thickness (float or estimate, optional): Thickness in mm, known
            or estimated.  Omitted, the catalog's default thickness
            range is used.
        name (str, optional): Label used in results.  Defaults to
            'detector'.
    """

    _kind = 'scintillator'


class ReflectionSource:
    """An X-ray tube with a thick angled tungsten anode.

    Spectra come from a Spekpy lookup table over voltage and takeoff
    angle that xcal generates internally.  The per-scan voltage is given
    to :meth:`Calibrator.add_scan`, not here.

    Args:
        takeoff_angle (float or estimate, optional): Anode takeoff angle
            in degrees, known or estimated.  Omitted, the catalog's
            default range (5 to 45 degrees) is estimated.
    """

    def __init__(self, takeoff_angle=None):
        if takeoff_angle is None:
            rng = catalog._defaults().get('takeoff_angle_range', [5, 45])
            takeoff_angle = estimate(rng[0], rng[1])
        self.takeoff_angle = _check_scalar_or_estimate(takeoff_angle,
                                                       'takeoff_angle')

    def __repr__(self):
        return f"ReflectionSource(takeoff_angle={self.takeoff_angle})"


class TransmissionSource:
    """An X-ray tube with a thin tungsten transmission target.

    Spectra come from a lookup table over voltage and target thickness.
    The per-scan voltage is given to :meth:`Calibrator.add_scan`, not
    here.

    Args:
        target_thickness (float or estimate): Target thickness in mm,
            known or estimated.
        spectra_table (str, optional): Path to an HDF5 lookup table of
            simulated source spectra over voltage and target thickness.
            xcal does not ship transmission tables yet, so one must be
            provided.
    """

    def __init__(self, target_thickness, spectra_table=None):
        self.target_thickness = _check_scalar_or_estimate(
            target_thickness, 'target_thickness')
        self.spectra_table = spectra_table

    def __repr__(self):
        return (f"TransmissionSource(target_thickness="
                f"{self.target_thickness})")


class SynchrotronSource:
    """A source with a known spectrum, such as a synchrotron beamline.

    No source parameter is estimated.

    Args:
        spectrum (str or tuple): The name of a built-in spectrum table
            ('als_bm832' for ALS beamline 8.3.2), or a tuple
            (energies, counts) of numpy arrays with energies in keV.
    """

    _builtin = ('als_bm832',)

    def __init__(self, spectrum):
        if isinstance(spectrum, str):
            if spectrum not in self._builtin:
                raise ValueError(
                    f"Unknown built-in spectrum '{spectrum}'; available: "
                    f"{list(self._builtin)}.")
            self.spectrum = spectrum
        else:
            try:
                energies, counts = spectrum
                energies = np.asarray(energies, dtype=float)
                counts = np.asarray(counts, dtype=float)
            except (TypeError, ValueError):
                raise ValueError(
                    "spectrum must be a built-in name or a tuple "
                    "(energies, counts) of equal-length arrays.")
            if energies.shape != counts.shape or energies.ndim != 1:
                raise ValueError(
                    "spectrum arrays must be 1D and of equal length, got "
                    f"shapes {energies.shape} and {counts.shape}.")
            self.spectrum = (energies, counts)

    def __repr__(self):
        label = self.spectrum if isinstance(self.spectrum, str) else 'custom'
        return f"SynchrotronSource(spectrum='{label}')"


class System:
    """The complete description of the X-ray system to calibrate.

    Args:
        source: One of :class:`ReflectionSource`,
            :class:`TransmissionSource`, or :class:`SynchrotronSource`.
        filters (list): The :class:`Filter` objects that may be in the
            beam.  Scans state which of them were present.
        detector: A :class:`Scintillator`.

    Example:
        >>> system = xcal.System(
        ...     source=xcal.ReflectionSource(),
        ...     filters=[xcal.Filter(material=['Al', 'Cu'], thickness=xcal.estimate(0, 10))],
        ...     detector=xcal.Scintillator(thickness=xcal.estimate(0.001, 0.5)))
    """

    def __init__(self, source, filters=(), detector=None):
        if not isinstance(source, (ReflectionSource, TransmissionSource,
                                   SynchrotronSource)):
            raise TypeError(
                f"source must be a ReflectionSource, TransmissionSource, "
                f"or SynchrotronSource, got {source!r}.")
        if isinstance(filters, Filter):
            filters = [filters]
        filters = list(filters)
        for f in filters:
            if not isinstance(f, Filter) or isinstance(f, Scintillator):
                raise TypeError(f"filters must contain Filter objects, "
                                f"got {f!r}.")
        if len(set(map(id, filters))) != len(filters):
            raise ValueError("filters contains the same Filter object "
                             "twice; create one object per physical "
                             "filter.")
        if not isinstance(detector, Scintillator):
            raise TypeError(f"detector must be a Scintillator, got "
                            f"{detector!r}.")
        self.source = source
        self.filters = filters
        self.detector = detector

    def filter_label(self, filt):
        """Return the display label of one filter, for example
        'filter 1 (Si)'."""
        index = [id(f) for f in self.filters].index(id(filt))
        label = f"filter {index + 1}"
        if filt.name:
            label += f" ({filt.name})"
        elif len(filt.materials) == 1:
            label += f" ({filt.materials[0].name})"
        return label
