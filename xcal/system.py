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
        raise NotImplementedError("xcal 2 skeleton")


class Rod:
    """One homogeneous rod in the calibration object.

    Args:
        material (str): Chemical formula of the rod material, e.g. 'Ti'.
            Must resolve against the materials catalog or be an element.
        diameter (float): Nominal rod diameter in mm.  Used to guide
            segmentation; the actual shape is measured from the
            reconstruction.
        density (float, optional): Density in g/cm^3.  Required only for
            materials whose density is not in the catalog.
    """

    def __init__(self, material, diameter, density=None):
        raise NotImplementedError("xcal 2 skeleton")


class Filter:
    """A beam filter modeled by Beer's law.

    Args:
        material (str or list, optional): Chemical formula, a list of
            candidate formulas, or omitted to search the catalog's
            standard filter materials.
        thickness (float or estimate): Thickness in mm, known or
            estimated.
        name (str, optional): Label used in results, for example in
            the keys of ``result.params``.  Defaults to the filter's
            position, 'filter 1', 'filter 2', and so on.

    Example:
        >>> al = xcal.Filter(material=['Al', 'Cu'], thickness=xcal.estimate(0, 10))
    """

    def __init__(self, material=None, thickness=None, name=None):
        raise NotImplementedError("xcal 2 skeleton")


class Scintillator:
    """An energy-integrating scintillated detector.

    The response is the scintillator's absorption efficiency times the
    deposited photon energy, computed from the NIST attenuation and
    energy-absorption tables.

    Args:
        material (str or list, optional): Chemical formula, a list of
            candidates, or omitted to search the catalog's standard
            scintillators.
        thickness (float or estimate): Thickness in mm, known or
            estimated.
    """

    def __init__(self, material=None, thickness=None):
        raise NotImplementedError("xcal 2 skeleton")


class ReflectionSource:
    """An X-ray tube with a thick angled tungsten anode.

    Spectra come from a Spekpy lookup table over voltage and takeoff
    angle that xcal generates internally.  The per-scan voltage is given
    to :meth:`Calibrator.add_scan`, not here.

    Args:
        takeoff_angle (float or estimate): Anode takeoff angle in
            degrees, known or estimated.  Typical range is 5 to 45.
    """

    def __init__(self, takeoff_angle=None):
        raise NotImplementedError("xcal 2 skeleton")


class TransmissionSource:
    """An X-ray tube with a thin tungsten transmission target.

    Spectra come from a precomputed Geant4 lookup table over voltage and
    target thickness that ships with xcal.  The per-scan voltage is
    given to :meth:`Calibrator.add_scan`, not here.

    Args:
        target_thickness (float or estimate): Target thickness in mm,
            known or estimated.
    """

    def __init__(self, target_thickness=None):
        raise NotImplementedError("xcal 2 skeleton")


class SynchrotronSource:
    """A source with a known spectrum, such as a synchrotron beamline.

    No source parameter is estimated.

    Args:
        spectrum (str or tuple): The name of a built-in spectrum table
            ('als_bm832' for ALS beamline 8.3.2), or a tuple
            (energies, counts) of numpy arrays with energies in keV.
    """

    def __init__(self, spectrum):
        raise NotImplementedError("xcal 2 skeleton")


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
        ...     source=xcal.TransmissionSource(target_thickness=xcal.estimate(0.001, 0.007)),
        ...     filters=[xcal.Filter(material=['Al', 'Cu'], thickness=xcal.estimate(0, 10))],
        ...     detector=xcal.Scintillator(thickness=xcal.estimate(0.001, 0.5)))
    """

    def __init__(self, source, filters, detector):
        raise NotImplementedError("xcal 2 skeleton")
