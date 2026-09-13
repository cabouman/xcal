"""The calibrator and its result.

The :class:`Calibrator` is built from a :class:`~xcal.System` and a list
of :class:`~xcal.Target` objects.  Scans are added as (sinogram, model)
pairs produced by mbirtorch preprocessing, and :meth:`Calibrator.calibrate`
returns a :class:`CalibrationResult`.
"""

import numpy as np

from . import _physics
from . import _segment
from .system import (estimate, Filter, Scintillator, ReflectionSource,
                     TransmissionSource, SynchrotronSource, System,
                     Target)

__all__ = ['Calibrator', 'CalibrationResult']


def _as_numpy(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(a)


class Calibrator:
    """Estimates the system spectral response from calibration scans.

    Args:
        system (System): The X-ray system description.
        targets (list of Target): The calibration targets.

    Example:
        >>> recon, _ = ct_model.recon(sino)
        >>> masks = xcal.segment_targets(recon, targets, ct_model)
        >>> cal = xcal.Calibrator(system, targets)
        >>> cal.add_scan(sino, ct_model, masks, voltage=80)
        >>> result = cal.calibrate()
    """

    def __init__(self, system, targets):
        if not isinstance(system, System):
            raise TypeError(f"system must be an xcal.System, got "
                            f"{system!r}.")
        targets = list(targets)
        if not targets:
            raise ValueError("targets must contain at least one "
                             "Target.")
        self.system = system
        self.targets = targets
        self.scans = []

    def add_scan(self, sinogram, ct_model, target_masks,
                 voltage=None, targets=None, filters=None,
                 weights=None):
        """Add one calibration scan.

        The sinogram and model are the pair returned by mbirtorch
        preprocessing, for example ``mtp.zeiss.get_sino_and_model(...)``.
        The target masks come from :func:`~xcal.segment_targets`
        applied to a reconstruction, or from
        :func:`~xcal.cylinder_masks` when the target geometry is
        trusted.
        xcal recovers the transmission internally as exp(-sinogram).
        Two cautions.  Preprocessing corrections such as stripe or
        offset removal carry into the recovered transmission, which is
        normally desirable.  Beam hardening correction, however, must
        be OFF in preprocessing (the pymbir loader applies it by
        default): a corrected sinogram is not a physical transmission,
        and calibrating from it is meaningless.

        The model's geometry must be in millimeter units, which is the
        default (``alu_unit='mm'``) in the mbirtorch loaders.

        Args:
            sinogram (numpy.ndarray or torch.Tensor): Log-domain
                sinogram with shape (views, detector rows, detector
                channels).  Non-finite entries are excluded from the
                fit.
            ct_model (TomographyModel): The mbirtorch geometry model for
                this scan.  Any supported geometry works; xcal uses
                only its forward projection method.
            target_masks (list of numpy.ndarray): One float32 volume
                per target of this scan, in target order, values in
                [0, 1] meaning the fraction of each voxel the target
                occupies.
            voltage (float, optional): Source voltage for this scan in
                kV.  Required for tube sources, ignored for synchrotron
                sources.
            targets (list of Target, optional): The targets present
                in this scan.  Defaults to all targets given to the
                constructor.
            filters (list of Filter, optional): The filters in the beam
                for this scan.  Defaults to all filters in the system.
            weights (numpy.ndarray, optional): Per-ray fit weights,
                shaped like the sinogram.  Defaults to 1/transmission,
                which approximates photon counting noise.
        """
        sinogram = _as_numpy(sinogram).astype(float)
        if sinogram.ndim != 3:
            raise ValueError(
                f"sinogram must have shape (views, rows, channels), got "
                f"shape {sinogram.shape}.")
        if isinstance(self.system.source, (ReflectionSource,
                                           TransmissionSource)):
            if voltage is None:
                raise ValueError("voltage is required for tube sources.")
            voltage = float(voltage)
        scan_targets = (list(targets) if targets is not None
                        else list(self.targets))
        for tg in scan_targets:
            if not any(tg is tt for tt in self.targets):
                raise ValueError(
                    f"{tg!r} was not in the targets list given to "
                    f"the Calibrator.")
        scan_filters = (list(filters) if filters is not None
                        else list(self.system.filters))
        for f in scan_filters:
            if not any(f is ff for ff in self.system.filters):
                raise ValueError(
                    f"{f!r} is not one of the System's filters.")
        if weights is not None:
            weights = _as_numpy(weights).astype(float)
            if weights.shape != sinogram.shape:
                raise ValueError(
                    f"weights shape {weights.shape} does not match the "
                    f"sinogram shape {sinogram.shape}.")
        target_masks = [np.asarray(m, dtype=np.float32)
                        for m in target_masks]
        if len(target_masks) != len(scan_targets):
            raise ValueError(
                f"target_masks has {len(target_masks)} entries for "
                f"{len(scan_targets)} targets in this scan.")
        shape = tuple(ct_model.get_params('recon_shape'))
        for k, m in enumerate(target_masks):
            if m.shape != shape:
                raise ValueError(
                    f"target_masks[{k}] has shape {m.shape}; the "
                    f"model's reconstruction shape is {shape}.")
            if m.min() < 0 or m.max() > 1.001:
                raise ValueError(
                    f"target_masks[{k}] has values outside [0, 1].")
        self._check_masks_inside_ror(target_masks, scan_targets)
        self.scans.append({
            'sinogram': sinogram,
            'ct_model': ct_model,
            'voltage': voltage,
            'targets': scan_targets,
            'filters': scan_filters,
            'weights': weights,
            'target_masks': target_masks,
        })

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _check_masks_inside_ror(target_masks, targets):
        """A mask outside the projector's circular region of
        reconstruction would be silently truncated by the forward
        projection; refuse it."""
        rows, cols, _ = target_masks[0].shape
        yy, xx = np.ogrid[:rows, :cols]
        ror = ((yy - (rows - 1) / 2) ** 2 / ((rows / 2 - 1) ** 2)
               + (xx - (cols - 1) / 2) ** 2
               / ((cols / 2 - 1) ** 2)) <= 1.0
        for tg, m in zip(targets, target_masks):
            mid = m[:, :, m.shape[2] // 2] > 0.5
            if (mid & ~ror).any():
                raise ValueError(
                    f"the {tg.material.name} target's mask extends "
                    f"outside the circular region of reconstruction, "
                    f"so its forward projection would be silently "
                    f"truncated.  Enlarge the grid with "
                    f"ct_model.scale_recon_shape(...).")

    def _energy_grid(self):
        source = self.system.source
        if isinstance(source, SynchrotronSource):
            if isinstance(source.spectrum, str):
                energies, _ = _physics.load_als_spectrum()
            else:
                energies = source.spectrum[0]
            max_e = float(energies.max())
            return _physics.default_energy_grid(max_e)
        max_v = max(s['voltage'] for s in self.scans)
        return _physics.default_energy_grid(max_v)

    def _source_term(self, scan, energies):
        """Return the _fit source term for one scan: ('fixed', spec) or
        ('table', grid, table)."""
        source = self.system.source
        if isinstance(source, SynchrotronSource):
            if isinstance(source.spectrum, str):
                e_tab, counts = _physics.load_als_spectrum()
            else:
                e_tab, counts = source.spectrum
            spec = np.interp(energies, e_tab, counts, left=0.0, right=0.0)
            return ('fixed', spec)

        if isinstance(source, ReflectionSource):
            angle = source.takeoff_angle
            if isinstance(angle, estimate):
                grid = np.linspace(angle.low, angle.high, 11)
                table = _physics.reflection_source_table(
                    scan['voltage'], grid, energies)
                return ('table', grid, table)
            table = _physics.reflection_source_table(
                scan['voltage'], [angle], energies)
            return ('fixed', table[0])

        # Transmission source: interpolate the Geant4 table to this
        # scan's voltage, leaving thickness as the table coordinate.
        voltages, th_mm, e_tab, spectra = _physics.transmission_source_table(
            source.spectra_table)
        v = scan['voltage']
        if not voltages.min() <= v <= voltages.max():
            raise ValueError(
                f"scan voltage {v} kV lies outside the transmission "
                f"source table range [{voltages.min()}, "
                f"{voltages.max()}] kV.")
        per_th = []
        for ti in range(len(th_mm)):
            ext = _physics.prepare_for_interpolation(spectra[:, ti])
            row = _physics.interpolate_rows(voltages, ext, v)
            per_th.append(np.clip(row, 0.0, None))
        table = np.stack([np.interp(energies, e_tab, r, left=0.0, right=0.0)
                          for r in per_th])
        thickness = source.target_thickness
        if isinstance(thickness, estimate):
            return ('table', th_mm, table)
        return ('fixed', _physics.interpolate_rows(th_mm, table, thickness))

    def _select_rays(self, scan, path_lengths, num_fit_views, num_fit_rows):
        """Choose the sinogram entries used in the fit: a subset of
        views and center rows, rays that hit at least one target, and
        finite positive transmission."""
        n_views, n_rows, n_chan = scan['sinogram'].shape
        view_idx = np.unique(np.linspace(0, n_views - 1,
                                         min(num_fit_views, n_views),
                                         dtype=int))
        row_lo = max(0, n_rows // 2 - num_fit_rows // 2)
        row_idx = np.arange(row_lo, min(n_rows, row_lo + num_fit_rows))
        sel = np.zeros(scan['sinogram'].shape, dtype=bool)
        sel[np.ix_(view_idx, row_idx, np.arange(n_chan))] = True

        trans = np.exp(-scan['sinogram'])
        hits = np.zeros(scan['sinogram'].shape, dtype=bool)
        grazing = np.zeros(scan['sinogram'].shape, dtype=bool)
        for L in path_lengths:
            hits |= (L > 0)
            # A ray that clips a target's edge has a path length
            # dominated by segmentation error; exclude rays below 30
            # percent of that target's maximum path.
            grazing |= (L > 0) & (L < 0.3 * L.max())
        sel &= hits & ~grazing
        sel &= np.isfinite(trans) & (trans > 1e-6) & (trans < 1.5)
        return sel

    # -- the pipeline -------------------------------------------------------

    def calibrate(self, learning_rate=0.02, max_iterations=5000,
                  stop_threshold=1e-6, num_fit_views=16, num_fit_rows=5,
                  verbose=1):
        """Run the calibration and return the result.

        The steps are: forward project each scan's target masks to
        get per-ray path lengths in mm, then jointly fit the system
        parameters to all scans.  Candidate materials are searched
        exhaustively; continuous parameters are fit with Adam within
        their bounds.

        Args:
            learning_rate (float, optional): Adam step size.
            max_iterations (int, optional): Iteration cap per material
                combination.
            stop_threshold (float, optional): Stop when no normalized
                parameter moves more than this in one iteration.
            num_fit_views (int, optional): Number of views per scan
                used in the spectral fit.
            num_fit_rows (int, optional): Number of center detector
                rows per scan used in the spectral fit.
            verbose (int, optional): 0 is silent, 1 prints progress.

        Returns:
            CalibrationResult: The estimated parameters and spectra.
        """
        if not self.scans:
            raise ValueError("no scans were added; call add_scan first.")
        for f in self.system.filters:
            if not any(any(f is sf for sf in s['filters'])
                       for s in self.scans):
                raise ValueError(
                    f"{self.system.filter_label(f)} appears in no scan, "
                    f"so its parameters cannot be estimated; remove it "
                    f"from the System or add a scan that used it.")

        energies = self._energy_grid()

        # Forward project each scan's masks for path lengths in mm.
        all_paths = []
        for si, scan in enumerate(self.scans):
            scale = _physics.mm_per_alu(scan['ct_model'])
            paths = [_as_numpy(scan['ct_model'].forward_project(m))
                     * scale for m in scan['target_masks']]
            all_paths.append(paths)

        # Build the discretized fit problem.
        filters = self.system.filters
        fit_scans = []
        selections = []
        for scan, paths in zip(self.scans, all_paths):
            sel = self._select_rays(scan, paths, num_fit_views,
                                    num_fit_rows)
            selections.append(sel)
            trans = np.exp(-scan['sinogram'])[sel]
            mu_targets = [_physics.attenuation_coefficients(
                tg.material, energies) for tg in scan['targets']]
            total = np.zeros((trans.size, len(energies)))
            for L, mu in zip(paths, mu_targets):
                total += np.outer(L[sel], mu)
            A = np.exp(-total)
            if scan['weights'] is not None:
                w = scan['weights'][sel]
            else:
                w = 1.0 / np.clip(trans, 1e-6, None)
            fit_scans.append({
                'A': A,
                'y': trans,
                'w': w,
                'filter_indices': [next(i for i, f in enumerate(filters)
                                        if f is sf)
                                   for sf in scan['filters']],
                'source': self._source_term(scan, energies),
            })

        source = self.system.source
        if isinstance(source, ReflectionSource):
            source_param = source.takeoff_angle
        elif isinstance(source, TransmissionSource):
            source_param = source.target_thickness
        else:
            source_param = None
        fit_filters = [{
            'mu_candidates': [_physics.attenuation_coefficients(m, energies)
                              for m in f.materials],
            'thickness': (f.thickness_per_candidate
                          if f.thickness_per_candidate is not None
                          else f.thickness),
        } for f in filters]
        detector = self.system.detector
        fit_detector = {
            'curve_candidates': [_physics.scintillator_curves(m, energies)
                                 for m in detector.materials],
            'thickness': (detector.thickness_per_candidate
                          if detector.thickness_per_candidate is not None
                          else detector.thickness),
        }

        from . import _fit
        problem = _fit.FitProblem(energies, fit_scans, source_param,
                                  fit_filters, fit_detector)
        solution = problem.solve(learning_rate=learning_rate,
                                 max_iterations=max_iterations,
                                 stop_threshold=stop_threshold,
                                 verbose=verbose)

        return CalibrationResult(self, energies, solution, all_paths,
                                 selections, fit_scans)


class CalibrationResult:
    """The output of :meth:`Calibrator.calibrate`.

    The result returns data and functions; it does not plot.  The
    spectral quantities are returned as functions of energy that the
    user evaluates and plots as they choose.  The one display
    convenience is :meth:`show`.  The target masks are reviewed before
    calibration, at the segmentation step.

    The returned functions satisfy R(E) proportional to
    S(E) * product of filter transmissions * D(E), with the effective
    spectrum R normalized to integrate to one.
    """

    def __init__(self, calibrator, energies, solution, paths,
                 selections, fit_scans):
        self._cal = calibrator
        self._system = calibrator.system
        self._energies = np.asarray(energies)
        self._solution = solution
        self._paths = paths
        self._selections = selections
        self._fit_scans = fit_scans

        combo = solution['combo']
        self.filters = list(self._system.filters)
        self._filter_materials = [
            f.materials[combo[i]] for i, f in enumerate(self.filters)]
        self._filter_thicknesses = list(solution['filter_thicknesses'])
        self._detector_material = \
            self._system.detector.materials[combo[-1]]
        self._detector_thickness = solution['detector_thickness']
        self._source_value = solution['source_value']
        self.cost = solution['cost']
        self.candidates = sorted(solution['all'], key=lambda c: c[1])

    # -- parameters ---------------------------------------------------------

    @property
    def est_system(self):
        """System: the estimated system, fully specified.

        A :class:`~xcal.System` with every estimated value filled
        in, usable exactly like a ground truth system.  The three
        stages of a calibration speak one language: a gt_system goes
        into the simulation, a feasible_system goes into the
        calibrator, and est_system comes out.
        """
        source = self._system.source
        if isinstance(source, ReflectionSource):
            src = ReflectionSource(takeoff_angle=self._source_value)
        elif isinstance(source, TransmissionSource):
            src = TransmissionSource(
                target_thickness=self._source_value,
                spectra_table=source.spectra_table)
        else:
            src = source
        filters = [Filter(material=m.formula, thickness=th,
                          name=f.name, density=m.density)
                   for f, m, th in zip(self.filters,
                                       self._filter_materials,
                                       self._filter_thicknesses)]
        det = Scintillator(material=self._detector_material.formula,
                           thickness=self._detector_thickness,
                           density=self._detector_material.density)
        return System(source=src, filters=filters, detector=det)

    def parameters(self):
        """Return the full parameter table with provenance.

        One row per system parameter.  Each row is a dict with keys
        'name', 'value', 'units', 'origin', 'low', 'high', and
        'note'.  The origin is one of three words: 'given' (stated
        by the user and held fixed), 'estimated' (fitted within the
        bounds in 'low' and 'high'), or 'setting' (an instrument
        knob such as the source voltage, which the result can
        evaluate at any value).

        Returns:
            list of dict: The rows, in system order.
        """
        if getattr(self, '_loaded_rows', None) is not None:
            return [dict(r) for r in self._loaded_rows]
        rows = []
        source = self._system.source

        def row(name, value, units='', origin='given', low=None,
                high=None, note=''):
            rows.append({'name': name, 'value': value, 'units': units,
                         'origin': origin, 'low': low, 'high': high,
                         'note': note})

        if isinstance(source, (ReflectionSource, TransmissionSource)):
            voltages = sorted({s['voltage'] for s in self._cal.scans})
            row('source voltage', ' '.join(f'{v:g}' for v in voltages),
                'kV', 'setting',
                note='the result evaluates any voltage in range')
        if isinstance(source, ReflectionSource):
            spec = source.takeoff_angle
            if isinstance(spec, estimate):
                row('source takeoff angle', self._source_value, 'deg',
                    'estimated', spec.low, spec.high)
            else:
                row('source takeoff angle', spec, 'deg', 'given')
        elif isinstance(source, TransmissionSource):
            spec = source.target_thickness
            if isinstance(spec, estimate):
                row('source target thickness', self._source_value,
                    'mm', 'estimated', spec.low, spec.high)
            else:
                row('source target thickness', spec, 'mm', 'given')
        else:
            name = (source.spectrum if isinstance(source.spectrum, str)
                    else 'user table')
            row('source spectrum', name, '', 'given')

        combo = self._solution['combo']
        for i, (f, mat, th) in enumerate(
                zip(self.filters, self._filter_materials,
                    self._filter_thicknesses)):
            label = self._system.filter_label(f)
            if len(f.materials) > 1:
                row(f'{label} material', mat.name, '', 'estimated',
                    note='chosen from ' + ', '.join(
                        m.name for m in f.materials))
            else:
                row(f'{label} material', mat.name, '', 'given')
            spec = f.thickness
            if f.thickness_per_candidate is not None:
                spec = f.thickness_per_candidate[combo[i]]
            if isinstance(spec, estimate):
                row(f'{label} thickness', th, 'mm', 'estimated',
                    spec.low, spec.high)
            else:
                row(f'{label} thickness', th, 'mm', 'given')

        det = self._system.detector
        if len(det.materials) > 1:
            row('detector material', self._detector_material.name, '',
                'estimated', note='chosen from ' + ', '.join(
                    m.name for m in det.materials))
        else:
            row('detector material', self._detector_material.name, '',
                'given')
        spec = det.thickness
        if det.thickness_per_candidate is not None:
            spec = det.thickness_per_candidate[combo[-1]]
        if isinstance(spec, estimate):
            row('detector thickness', self._detector_thickness, 'mm',
                'estimated', spec.low, spec.high)
        else:
            row('detector thickness', self._detector_thickness, 'mm',
                'given')
        return rows

    @property
    def params(self):
        """dict: Estimated parameters keyed by readable names.

        Components are named by their position, with any user-given
        name in parentheses, for example
        'filter 1 (Si) thickness (mm)'.
        """
        out = {}
        source = self._system.source
        if isinstance(source, ReflectionSource):
            out['source takeoff angle (deg)'] = self._source_value
        elif isinstance(source, TransmissionSource):
            out['source target thickness (mm)'] = self._source_value
        for f, mat, th in zip(self.filters, self._filter_materials,
                              self._filter_thicknesses):
            label = self._system.filter_label(f)
            out[f'{label} material'] = mat.name
            out[f'{label} thickness (mm)'] = th
        out['detector material'] = self._detector_material.name
        out['detector thickness (mm)'] = self._detector_thickness
        return out

    def summary(self):
        """Return the parameter table as a string, one row per
        parameter with its value, origin, and bounds."""
        lines = ['System parameters:']
        for r in self.parameters():
            value = (f'{r["value"]:.6g}' if isinstance(r['value'], float)
                     else str(r['value']))
            units = f' {r["units"]}' if r['units'] else ''
            tail = r['origin']
            if r['origin'] == 'estimated' and r['low'] is not None:
                tail += f', bounds {r["low"]:g} to {r["high"]:g}'
            if r['note']:
                tail += f'; {r["note"]}'
            lines.append(f'  {r["name"]}: {value}{units}  [{tail}]')
        lines.append(f'Final cost: {self.cost:.6e}')
        runners = [c for c in self.candidates[1:4]]
        if runners:
            lines.append('Next best material combinations:')
            for combo, cost in runners:
                names = [f.materials[ci].name
                         for f, ci in zip(self.filters, combo[:-1])]
                names.append(
                    self._system.detector.materials[combo[-1]].name)
                lines.append(f'  {names}: cost {cost:.6e}')
        return '\n'.join(lines)

    # -- spectral functions ---------------------------------------------

    def _source_spectrum_values(self, energies, voltage):
        source = self._system.source
        if isinstance(source, SynchrotronSource):
            if isinstance(source.spectrum, str):
                e_tab, counts = _physics.load_als_spectrum()
            else:
                e_tab, counts = source.spectrum
            return np.interp(energies, e_tab, counts, left=0.0, right=0.0)
        if voltage is None:
            raise ValueError("voltage is required for tube sources.")
        if isinstance(source, ReflectionSource):
            table = _physics.reflection_source_table(
                voltage, [self._source_value], energies)
            return table[0]
        voltages, th_mm, e_tab, spectra = \
            _physics.transmission_source_table(source.spectra_table)
        per_th = []
        for ti in range(len(th_mm)):
            ext = _physics.prepare_for_interpolation(spectra[:, ti])
            row = _physics.interpolate_rows(voltages, ext, voltage)
            per_th.append(np.clip(row, 0.0, None))
        row = _physics.interpolate_rows(th_mm, np.stack(per_th),
                                        self._source_value)
        return np.interp(energies, e_tab, row, left=0.0, right=0.0)

    def _filter_product(self, energies, filts):
        prod = np.ones_like(energies, dtype=float)
        for f in filts:
            i = next(j for j, ff in enumerate(self.filters) if ff is f)
            prod *= _physics.filter_transmission(
                self._filter_materials[i], self._filter_thicknesses[i],
                energies)
        return prod

    def effective_spectrum(self, voltage=None, filters=None):
        """Return the effective spectrum as a function of energy.

        The returned function maps energy in keV to spectral density
        in 1/keV.  It accepts a scalar or a numpy array and returns
        the same shape.  The density is zero above the source voltage
        and integrates to one, because the air scan normalization
        makes the absolute scale unidentifiable.

        The spectrum depends on the instrument setting, given by the
        two arguments.  Any setting in the calibrated range works,
        not only the scanned ones, because the parameters stay valid
        when the setting changes.

        Args:
            voltage (float, optional): Source voltage in kV.  Required
                for tube sources; ignored for synchrotron sources.
            filters (list of Filter, optional): The filters in the
                beam.  Defaults to all filters in the system.

        Returns:
            callable: A function R with R(energies) -> density.

        Example:
            >>> R = result.effective_spectrum(voltage=80)
            >>> E = np.linspace(1, 80, 320)
            >>> plt.plot(E, R(E))
        """
        filts = list(filters) if filters is not None else list(self.filters)
        grid = self._dense_grid(voltage)
        values = self._source_spectrum_values(grid, voltage)
        values = values * self._filter_product(grid, filts)
        values = values * _physics.scintillator_response(
            self._detector_material, self._detector_thickness, grid)
        area = np.trapezoid(values, grid)
        if area <= 0:
            raise ValueError(
                "the effective spectrum is zero everywhere; check the "
                "voltage and filters.")
        return _physics.SpectralFunction(grid, values / area)

    def source_spectrum(self, voltage=None):
        """Return the estimated source spectrum as a function of energy
        in keV, normalized to unit area like the effective spectrum.

        Args:
            voltage (float, optional): Source voltage in kV.  Required
                for tube sources.

        Returns:
            callable: A function S with S(energies) -> density.
        """
        grid = self._dense_grid(voltage)
        values = self._source_spectrum_values(grid, voltage)
        area = np.trapezoid(values, grid)
        return _physics.SpectralFunction(grid, values / area)

    def filter_response(self, filt):
        """Return one filter's estimated transmission as a function of
        energy in keV.  Values are between 0 and 1.

        Args:
            filt (Filter): The filter object whose response to return,
                the same object given to the System.

        Returns:
            callable: A function f with f(energies) -> transmission.
        """
        i = next(j for j, ff in enumerate(self.filters) if ff is filt)
        mat = self._filter_materials[i]
        th = self._filter_thicknesses[i]

        def response(energies):
            energies = np.asarray(energies, dtype=float)
            return _physics.filter_transmission(mat, th,
                                                np.atleast_1d(energies)
                                                ).reshape(energies.shape)
        return response

    def detector_response(self):
        """Return the estimated detector response as a function of
        energy in keV.  The scale is relative: only the shape is
        identifiable.

        Returns:
            callable: A function D with D(energies) -> response.
        """
        mat = self._detector_material
        th = self._detector_thickness

        def response(energies):
            energies = np.asarray(energies, dtype=float)
            return _physics.scintillator_response(
                mat, th, np.atleast_1d(energies)).reshape(energies.shape)
        return response

    def _dense_grid(self, voltage):
        source = self._system.source
        if isinstance(source, (ReflectionSource, TransmissionSource)):
            if voltage is None:
                raise ValueError("voltage is required for tube sources.")
            top = float(voltage)
        else:
            top = float(self._energies[-1]) + 0.5
        return np.linspace(1.0, top, max(int(top) * 4, 64))

    # -- per-scan data ----------------------------------------------------

    def transmission_fit(self, scan):
        """Return the measured and predicted transmission for the rays
        of one scan used in the fit.

        Args:
            scan (int): Index of the scan, in the order added
                (0 is the first).

        Returns:
            tuple: (measured, predicted) 1D numpy arrays, one entry per
            fit ray.
        """
        if getattr(self, '_loaded_scans', None) is not None:
            return self._loaded_scans[scan]
        fs = self._fit_scans[scan]
        spec = self._effective_values_for_scan(scan)
        predicted = np.trapezoid(fs['A'] * spec, self._energies, axis=-1)
        return fs['y'], predicted

    def _effective_values_for_scan(self, scan):
        s = self._cal.scans[scan]
        values = self._source_spectrum_values(self._energies, s['voltage'])
        values = values * self._filter_product(self._energies, s['filters'])
        values = values * _physics.scintillator_response(
            self._detector_material, self._detector_thickness,
            self._energies)
        return values / np.trapezoid(values, self._energies)

    # -- persistence and display -------------------------------------------

    def save(self, filename):
        """Save the estimated parameters and fit data to an HDF5 file.

        The file stores the estimated values with the resolved
        formulas and densities (never only catalog names), the energy
        grid, and the per-scan measured and predicted transmissions.
        The response functions are rebuilt from the parameters on
        load.

        Args:
            filename (str): Output path.
        """
        import h5py
        source = self._system.source
        with h5py.File(filename, 'w') as f:
            f.attrs['xcal_result_version'] = 1
            f.attrs['cost'] = self.cost
            grp = f.create_group('params')
            for key, value in self.params.items():
                grp.attrs[key] = value
            table = f.create_group('parameter_table')
            for idx, r in enumerate(self.parameters()):
                g = table.create_group(f'row_{idx:02d}')
                for key, value in r.items():
                    g.attrs[key] = '' if value is None else value
            src = f.create_group('source')
            if isinstance(source, ReflectionSource):
                src.attrs['type'] = 'reflection'
                src.attrs['value'] = self._source_value
            elif isinstance(source, TransmissionSource):
                src.attrs['type'] = 'transmission'
                src.attrs['value'] = self._source_value
            else:
                src.attrs['type'] = 'synchrotron'
                e_grid = self._energies
                src.create_dataset('energies', data=e_grid)
                src.create_dataset(
                    'counts',
                    data=self._source_spectrum_values(e_grid, None))
            for i, (filt, mat, th) in enumerate(
                    zip(self.filters, self._filter_materials,
                        self._filter_thicknesses)):
                g = f.create_group(f'filter_{i}')
                g.attrs['label'] = self._system.filter_label(filt)
                g.attrs['name'] = filt.name or ''
                g.attrs['formula'] = mat.formula
                g.attrs['density'] = mat.density
                g.attrs['thickness'] = th
            det = f.create_group('detector')
            det.attrs['formula'] = self._detector_material.formula
            det.attrs['density'] = self._detector_material.density
            det.attrs['thickness'] = self._detector_thickness
            f.create_dataset('energies', data=self._energies)
            for si in range(len(self._fit_scans)):
                y, pred = self.transmission_fit(si)
                g = f.create_group(f'scan_{si}')
                g.create_dataset('measured', data=y)
                g.create_dataset('predicted', data=pred)

    @classmethod
    def load(cls, filename):
        """Load a result saved by :meth:`save`.

        The loaded result rebuilds the response functions from the
        stored parameters, and exposes the filters as
        ``result.filters`` so ``filter_response(result.filters[0])``
        works in a new session.

        Args:
            filename (str): Path to a saved result.

        Returns:
            CalibrationResult: The loaded result.
        """
        import h5py
        from .system import System
        with h5py.File(filename, 'r') as f:
            src = f['source']
            src_type = src.attrs['type']
            if src_type == 'reflection':
                source = ReflectionSource(
                    takeoff_angle=float(src.attrs['value']))
                source_value = float(src.attrs['value'])
            elif src_type == 'transmission':
                source = TransmissionSource(
                    target_thickness=float(src.attrs['value']))
                source_value = float(src.attrs['value'])
            else:
                source = SynchrotronSource(
                    (np.array(src['energies']), np.array(src['counts'])))
                source_value = 0.0
            filters, filter_materials, filter_thicknesses = [], [], []
            i = 0
            while f'filter_{i}' in f:
                g = f[f'filter_{i}']
                filt = Filter(material=str(g.attrs['formula']),
                              thickness=float(g.attrs['thickness']),
                              name=str(g.attrs['name']) or None,
                              density=float(g.attrs['density']))
                filters.append(filt)
                filter_materials.append(filt.materials[0])
                filter_thicknesses.append(float(g.attrs['thickness']))
                i += 1
            det = f['detector']
            detector = Scintillator(material=str(det.attrs['formula']),
                                    thickness=float(det.attrs['thickness']),
                                    density=float(det.attrs['density']))
            energies = np.array(f['energies'])
            scans = []
            si = 0
            while f'scan_{si}' in f:
                scans.append((np.array(f[f'scan_{si}/measured']),
                              np.array(f[f'scan_{si}/predicted'])))
                si += 1
            cost = float(f.attrs['cost'])
            loaded_rows = []
            if 'parameter_table' in f:
                for key in sorted(f['parameter_table']):
                    a = f['parameter_table'][key].attrs
                    loaded_rows.append(
                        {k: (None if isinstance(a[k], str) and a[k] == ''
                             and k in ('low', 'high') else a[k])
                         for k in ('name', 'value', 'units', 'origin',
                                   'low', 'high', 'note')})

        system = System(source=source, filters=filters, detector=detector)
        solution = {
            'combo': tuple([0] * len(filters) + [0]),
            'cost': cost,
            'source_value': source_value,
            'filter_thicknesses': filter_thicknesses,
            'detector_thickness': float(det_thickness
                                        := detector.thickness),
            'iterations': 0,
            'all': [],
        }

        class _LoadedCal:
            pass
        cal = _LoadedCal()
        cal.system = system
        cal.scans = [{'voltage': None, 'filters': list(filters)}
                     for _ in scans]
        result = cls(cal, energies, solution, paths=None,
                     selections=None, fit_scans=None)
        result._loaded_scans = scans
        result._loaded_rows = loaded_rows
        return result

    def show(self, block=True):
        """Print the parameter table and plot the effective spectra
        and the transmission fit.  Everything shown is also available
        as data through the methods on this class.  Rod masks are
        reviewed before calibration, at the segmentation step.

        Args:
            block (bool, optional): Wait for the window to be closed.
        """
        print(self.summary())
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for si in range(len(self._fit_scans)):
            spec = self._effective_values_for_scan(si)
            axes[0].plot(self._energies, spec, label=f'scan {si}')
            y, pred = self.transmission_fit(si)
            axes[1].plot(y, pred, '.', markersize=2, label=f'scan {si}')
        axes[0].set_xlabel('Energy (keV)')
        axes[0].set_ylabel('Effective spectrum (1/keV)')
        axes[0].legend()
        axes[0].grid(True)
        lim = [0, 1.05]
        axes[1].plot(lim, lim, 'k-', linewidth=0.5)
        axes[1].set_xlabel('Measured transmission')
        axes[1].set_ylabel('Predicted transmission')
        axes[1].legend()
        axes[1].grid(True)
        fig.tight_layout()
        plt.show(block=block)
