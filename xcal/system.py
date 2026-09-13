"""Classes that describe the X-ray system and the calibration object.

A user builds a :class:`System` from a source, a list of filters, and a
detector, and describes the calibration object as a list of
:class:`Target` objects.  Every physical fact is stated in one of three forms:

* a plain value means the fact is known and fixed,
* :class:`estimate` means xcal estimates it within bounds,
* a list of names means xcal searches the candidates and picks the best.

A material argument may also be omitted, in which case xcal searches the
standard candidate list from the materials catalog.
"""

import numpy as np

from . import catalog
from . import _materials
from . import _physics

__all__ = ['estimate', 'Target', 'Filter', 'Scintillator', 'ReflectionSource',
           'TransmissionSource', 'SynchrotronSource', 'System',
           'load_system']


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


class Target:
    """One calibration target: a homogeneous object of one material.

    The calibration assumes only that the target is made of a single
    known material.  Its shape is whatever its mask says; the shape
    is measured by segmentation or, in simulation, built by a mask
    builder such as :func:`~xcal.cylinder_masks`.

    Args:
        material (str): The target material: a catalog name or any
            chemical formula of elements 1 through 92, e.g. 'Ti'.
        size (float): Approximate width of the target in mm.  Used
            only to scale the segmentation search and validate its
            result.
        density (float, optional): Density in g/cm^3.  Required only for
            compound formulas whose density is not in the catalog.
    """

    def __init__(self, material, size, density=None):
        self.material = _materials.resolve(material, 'target', density,
                                           context='Target')
        size = float(size)
        if size <= 0:
            raise ValueError(f"Target size must be positive mm, got "
                             f"{size}.")
        self.size = size

    def __repr__(self):
        return (f"Target(material='{self.material.name}', "
                f"size={self.size})")


class Filter:
    """A beam filter modeled by Beer's law.

    Args:
        material (str or list, optional): A catalog name or any
            chemical formula of elements 1 through 92, a list of
            candidates, or omitted to search the catalog's standard
            filter materials.
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
            # With thickness omitted, each candidate material gets its
            # own catalog range (copper's sensible range is not
            # aluminum's); self.thickness keeps the envelope for
            # display.  Candidates without a catalog range use the
            # envelope.
            envelope = _default_thickness_estimate(self._kind,
                                                   self.materials)
            per = []
            for m in self.materials:
                try:
                    per.append(_default_thickness_estimate(self._kind,
                                                           [m]))
                except ValueError:
                    per.append(envelope)
            self.thickness_per_candidate = per
            thickness = envelope
        else:
            self.thickness_per_candidate = None
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
        material (str or list, optional): A catalog name or any
            chemical formula of elements 1 through 92, a list of
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
    """An X-ray tube with a thick angled tungsten anode.  Spectra are
    generated at run time by Spekpy; the anode is tungsten only.
    The per-scan voltage is given to :meth:`Calibrator.add_scan`.

    Args:
        takeoff_angle (float or estimate, optional): Anode takeoff
            angle in degrees.  Valid: 0 to 90.  Omitted: estimated
            over the catalog default, 5 to 45.
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
    Spectra come from lookup tables, one CSV file per physics
    model in xcal/source_models (files transmission_<model>.csv).
    The shipped files (Geant4, tungsten target on a 250 um diamond
    substrate) define the valid values below.  To add a model, put
    a file in the same layout in that folder; its name becomes a
    valid physics_model and its contents define the other valid
    values.  Invalid choices are refused with the options listed.
    The per-scan voltage is given to
    :meth:`Calibrator.add_scan`; valid: 40 to 150 kV.

    Args:
        target_thickness (float or estimate): Target thickness in
            mm.  Valid: 0.001 to 0.007.
        apex_angle (float, optional): Anode apex angle in degrees.
            Valid: 1.7, 5, or 10.  Default 10.
        physics_model (str, optional): Geant4 physics model.  Valid:
            'G4EmPenelopePhysics', 'G4EmLivermorePhysics',
            'G4EmStandardPhysics', 'G4EmStandardPhysics-option4'.
            Default 'G4EmLivermorePhysics'.
    """

    def __init__(self, target_thickness, apex_angle=10.0,
                 physics_model='G4EmLivermorePhysics'):
        self.target_thickness = _check_scalar_or_estimate(
            target_thickness, 'target_thickness')
        self.apex_angle = float(apex_angle)
        available = _physics.available_transmission_models()
        if physics_model not in available:
            raise ValueError(
                f"Unknown physics model '{physics_model}'; "
                f"available: {available}.")
        self.physics_model = str(physics_model)

    def __repr__(self):
        return (f"TransmissionSource(target_thickness="
                f"{self.target_thickness}, "
                f"apex_angle={self.apex_angle})")


class SynchrotronSource:
    """A source with a known, exact spectrum and no parameters:
    nothing about it is estimated, and there is no per-scan voltage.
    Spectra come from CSV files, one per spectrum, in
    xcal/source_models (files synchrotron_<name>.csv).  To add a
    spectrum, put a file in the same layout in that folder; its
    name becomes a valid choice.  Invalid names are refused with
    the options listed.

    Args:
        spectrum (str, optional): Spectrum name.  Valid:
            'als_bm832' (the measured ALS Beamline 8.3.2 spectrum,
            0.5 to 99.5 keV).  Default 'als_bm832'.
    """

    def __init__(self, spectrum='als_bm832'):
        available = _physics.available_synchrotron_spectra()
        if spectrum not in available:
            raise ValueError(
                f"Unknown spectrum '{spectrum}'; available: "
                f"{available}.")
        self.spectrum = spectrum

    def table(self):
        """Return the spectrum as an (energies, counts) pair of
        arrays, energies in keV."""
        return _physics.synchrotron_source_table(self.spectrum)

    def __repr__(self):
        return f"SynchrotronSource(spectrum='{self.spectrum}')"


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



    def save(self, filename):
        """Save this system to a small readable YAML file.

        The file uses the same value notation as the API: a plain
        number is a given value, an ``estimate:`` entry carries
        bounds, a list of materials is a candidate set, and a null
        thickness means the catalog defaults.  Both fully specified
        systems (such as a ground truth or an estimate) and feasible
        systems save and load without loss.

        Args:
            filename (str): Output path, conventionally .yaml.
        """
        import yaml

        def value(spec):
            if isinstance(spec, estimate):
                out = {'estimate': [spec.low, spec.high]}
                if spec.initial != 0.5 * (spec.low + spec.high):
                    out['initial'] = spec.initial
                return out
            return spec

        def component(part):
            out = {}
            if len(part.materials) == 1:
                m = part.materials[0]
                out['material'] = {'name': m.name, 'formula': m.formula,
                                   'density': m.density}
            else:
                out['material'] = [m.name for m in part.materials]
            out['thickness'] = (None if part.thickness_per_candidate
                                is not None else value(part.thickness))
            if part.name:
                out['name'] = part.name
            return out

        source = self.source
        if isinstance(source, ReflectionSource):
            src = {'type': 'reflection',
                   'takeoff_angle': value(source.takeoff_angle)}
        elif isinstance(source, TransmissionSource):
            src = {'type': 'transmission',
                   'target_thickness': value(source.target_thickness),
                   'apex_angle': source.apex_angle,
                   'physics_model': source.physics_model}
        else:
            src = {'type': 'synchrotron',
                   'spectrum': source.spectrum}

        data = {'xcal_system': 1,
                'source': src,
                'filters': [component(f) for f in self.filters],
                'detector': component(self.detector)}
        with open(filename, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def _require_fully_specified(self, what):
        parts = [('source', self.source)] + \
            [(self.filter_label(f), f) for f in self.filters] + \
            [('detector', self.detector)]
        for label, part in parts:
            for attr in ('takeoff_angle', 'target_thickness',
                         'thickness'):
                if isinstance(getattr(part, attr, None), estimate):
                    raise ValueError(
                        f"{what} needs a fully specified System, but "
                        f"the {label} {attr} is an estimate.  Give it "
                        f"a plain value.")
            if getattr(part, 'materials', None) is not None \
                    and len(part.materials) > 1:
                raise ValueError(
                    f"{what} needs a fully specified System, but the "
                    f"{label} has {len(part.materials)} candidate "
                    f"materials.  Name one material.")

    def effective_spectrum(self, voltage=None, filters=None):
        """Return this system's effective spectrum as a function of
        energy, for a fully specified System (no estimates, no
        candidate lists).  The form matches
        CalibrationResult.effective_spectrum, so a simulated truth
        and a calibration result answer the same question the same
        way.

        Args:
            voltage (float, optional): Source voltage in kV.  Required
                for tube sources; ignored for synchrotron sources.
            filters (list of Filter, optional): The filters in the
                beam.  Defaults to all filters in the system.

        Returns:
            callable: A function R with R(energies) -> density in
            1/keV, normalized to integrate to one.
        """
        self._require_fully_specified('effective_spectrum')
        filts = list(filters) if filters is not None else \
            list(self.filters)
        for f in filts:
            if not any(f is ff for ff in self.filters):
                raise ValueError(f"{f!r} is not one of this System's "
                                 f"filters.")

        if isinstance(self.source, SynchrotronSource):
            e_tab, counts = self.source.table()
            grid = np.linspace(1.0, float(e_tab.max()),
                               max(int(e_tab.max()) * 4, 64))
            values = np.interp(grid, e_tab, counts, left=0.0, right=0.0)
        else:
            if voltage is None:
                raise ValueError("voltage is required for tube "
                                 "sources.")
            grid = np.linspace(1.0, float(voltage),
                               max(int(voltage) * 4, 64))
            if isinstance(self.source, ReflectionSource):
                values = _physics.reflection_source_table(
                    voltage, [self.source.takeoff_angle], grid)[0]
            else:
                voltages, th_mm, e_tab, spectra = \
                    _physics.transmission_source_table(
                        apex_angle=self.source.apex_angle,
                        physics_model=self.source.physics_model)
                per_th = []
                for ti in range(len(th_mm)):
                    ext = _physics.prepare_for_interpolation(
                        spectra[:, ti])
                    row = _physics.interpolate_rows(voltages, ext,
                                                    voltage)
                    per_th.append(np.clip(row, 0.0, None))
                row = _physics.interpolate_rows(
                    th_mm, np.stack(per_th),
                    self.source.target_thickness)
                values = np.interp(grid, e_tab, row, left=0.0,
                                   right=0.0)

        for f in filts:
            values = values * _physics.filter_transmission(
                f.materials[0], f.thickness, grid)
        values = values * _physics.scintillator_response(
            self.detector.materials[0], self.detector.thickness, grid)
        area = np.trapezoid(values, grid)
        if area <= 0:
            raise ValueError("the effective spectrum is zero "
                             "everywhere; check the voltage and "
                             "filters.")
        return _physics.SpectralFunction(grid, values / area)

    def _filters_note(self):
        """Short description of the filtration, e.g. 'Al 5 mm'."""
        if not self.filters:
            return 'no filter'
        return ', '.join(f'{f.materials[0].name} {f.thickness:.3g} mm'
                         for f in self.filters)

    def save_plot(self, filename, voltage=None, compare_to=None):
        """Write a plot of this system's effective spectrum.

        One curve per voltage.  With ``compare_to``, that system's
        spectrum is drawn dashed at the same voltages; the legend
        names each system by its filtration.

        Args:
            filename (str): Output image path.
            voltage (float or list of float, optional): Source
                voltage(s) in kV.  Omit for a synchrotron source.
            compare_to (System, optional): A second fully specified
                system to draw for comparison.
        """
        import matplotlib.pyplot as plt
        self._require_fully_specified('save_plot')
        if voltage is None:
            voltages = [None]
        else:
            voltages = list(np.atleast_1d(voltage))
        fig, ax = plt.subplots(figsize=(6, 4))
        for v in voltages:
            if v is not None:
                grid = _physics.default_energy_grid(float(v))
            else:
                e_tab, _ = self.source.table()
                grid = _physics.default_energy_grid(
                    float(np.max(e_tab)))
            setting = f'{v:g} kV, ' if v is not None else ''
            ax.plot(grid, self.effective_spectrum(voltage=v)(grid),
                    label=f'{setting}{self._filters_note()}')
            if compare_to is not None:
                ax.plot(grid,
                        compare_to.effective_spectrum(voltage=v)(grid),
                        '--',
                        label=f'{setting}'
                              f'{compare_to._filters_note()}')
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Effective spectrum (1/keV)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(filename, dpi=130)
        plt.close(fig)

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


def load_system(filename):
    """Read a system saved by :meth:`System.save`.

    Args:
        filename (str): Path to a system YAML file.

    Returns:
        System: The system, with given values, estimates, and
        candidate lists restored.
    """
    import yaml
    with open(filename) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or 'xcal_system' not in data:
        raise ValueError(f"{filename} is not an xcal system file.")

    def value(spec):
        if isinstance(spec, dict) and 'estimate' in spec:
            low, high = spec['estimate']
            return estimate(low, high, initial=spec.get('initial'))
        return spec

    def component(cls, entry, kind_word):
        mat = entry['material']
        if isinstance(mat, dict):
            return cls(material=mat['formula'],
                       density=mat['density'],
                       thickness=value(entry.get('thickness')),
                       name=entry.get('name'))
        return cls(material=list(mat),
                   thickness=value(entry.get('thickness')),
                   name=entry.get('name'))

    src = data['source']
    if src['type'] == 'reflection':
        source = ReflectionSource(
            takeoff_angle=value(src['takeoff_angle']))
    elif src['type'] == 'transmission':
        source = TransmissionSource(
            target_thickness=value(src['target_thickness']),
            apex_angle=src.get('apex_angle', 10.0),
            physics_model=src.get('physics_model',
                                  'G4EmLivermorePhysics'))
    elif src['type'] == 'synchrotron':
        source = SynchrotronSource(src['spectrum'])
    else:
        raise ValueError(f"unknown source type {src['type']!r} in "
                         f"{filename}.")

    return System(
        source=source,
        filters=[component(Filter, e, 'filter')
                 for e in data.get('filters', [])],
        detector=component(Scintillator, data['detector'],
                           'scintillator'))
