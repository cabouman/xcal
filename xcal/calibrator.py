"""Defines the calibrator and its result.

The :class:`Calibrator` is built from a :class:`~xcal.System` and a
list of :class:`~xcal.Target` objects.  Each scan is added as three
things: the sinogram and CT model from mbirtorch preprocessing, and
the target masks from segmentation or from a mask builder.
:meth:`Calibrator.calibrate` returns the estimated system and a
:class:`CalibrationResult` holding the fit information.
"""

import os

import numpy as np

from . import _physics
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
        >>> masks = my_segmentation(recon)     # the application's job
        >>> cal = xcal.Calibrator(system, targets)
        >>> cal.add_scan(sino, ct_model, masks, voltage=80)
        >>> cal_result = cal.calibrate()
        >>> est_system = cal_result.est_system
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
                 weights=None, fit_views=None, valid_mask=None):
        """Adds one calibration scan to the calibrator.

        The sinogram and model are the pair returned by mbirtorch
        preprocessing, for example ``mtp.zeiss.get_sino_and_model(...)``.
        The target masks are segmented from a reconstruction by the
        application (see the demos, which use mbirtorch's
        segmentation utilities), or built by
        :func:`~xcal.cylinder_masks` when the target geometry is
        trusted.  Order the masks like the targets: masks[k]
        belongs to targets[k].
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
            voltage (float, optional): Peak tube voltage (kVp) of
                this scan, in kV.  Required for tube sources,
                ignored for synchrotron sources.
            targets (list of Target, optional): The targets present
                in this scan.  Defaults to all targets given to the
                constructor.
            filters (list of Filter, optional): The filters in the beam
                for this scan.  Defaults to all filters in the system.
            weights (numpy.ndarray, optional): Fit weights, one per
                sinogram entry, shaped like the sinogram.  The
                calibration minimizes the weighted sum of squared
                transmission errors, so a ray with twice the weight
                counts twice as much in the fit.  When omitted, xcal
                computes weights = exp(sinogram), that is, one over
                each ray's measured transmission: rays through more
                attenuating material count more, the standard
                approximation for photon counting noise (the paper's
                Eq. 17).  Pass numpy.ones_like(sinogram) for equal
                weighting.
            fit_views (array of int, optional): Indices of the views
                to use in the spectral fit for this scan.  Defaults
                to evenly spaced views over the whole scan (the
                num_fit_views argument of :meth:`calibrate`).  Which
                views are informative is the scan's business: for
                example, a 360 degree parallel-beam scan measures
                every ray direction twice, so its unique views lie
                in either half rotation.
            valid_mask (numpy.ndarray, optional): Boolean array
                shaped like the sinogram, True on entries the fit
                may use.  The fit uses an entry only where this is
                True, so an application can exclude outlier or dead
                detector pixels it has identified.  Defaults to all
                True.
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
        if fit_views is not None:
            fit_views = np.unique(np.asarray(fit_views, dtype=int))
            if (fit_views.size == 0 or fit_views[0] < 0
                    or fit_views[-1] >= sinogram.shape[0]):
                raise ValueError(
                    f"fit_views must be view indices in [0, "
                    f"{sinogram.shape[0] - 1}].")
        if valid_mask is not None:
            valid_mask = _as_numpy(valid_mask).astype(bool)
            if valid_mask.shape != sinogram.shape:
                raise ValueError(
                    f"valid_mask shape {valid_mask.shape} does not "
                    f"match the sinogram shape {sinogram.shape}.")
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
            'fit_views': fit_views,
            'valid_mask': valid_mask,
            'target_masks': target_masks,
        })

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _check_masks_inside_ror(target_masks, targets):
        """Refuses a mask outside the projector's circular region of
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
        if isinstance(self.system.source, SynchrotronSource):
            return self.system.energy_grid()
        return self.system.energy_grid(
            max(s['voltage'] for s in self.scans))

    def _source_term(self, scan, energies):
        """Returns the _fit source term for one scan: ('fixed', spec) or
        ('table', grid, table)."""
        source = self.system.source
        if isinstance(source, SynchrotronSource):
            e_tab, counts = source.table()
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
            apex_angle=source.apex_angle,
            physics_model=source.physics_model)
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

    def _select_rays(self, scan, path_lengths, num_fit_views,
                     num_fit_rows, edge_trim_percent):
        """Chooses the sinogram entries used in the fit: a subset of
        views and center rows, rays through the target shadow with
        its edges trimmed, and finite positive transmission.

        The shadow is where a forward-projected mask has positive
        path length.  In each view and row it is a contiguous span
        of channels; edge_trim_percent of that span's width is
        dropped from each end, because a ray near the shadow edge
        has a path length dominated by segmentation error."""
        n_views, n_rows, n_chan = scan['sinogram'].shape
        if scan.get('fit_views') is not None:
            view_idx = scan['fit_views']
        else:
            view_idx = np.unique(np.linspace(0, n_views - 1,
                                             min(num_fit_views,
                                                 n_views),
                                             dtype=int))
        row_lo = max(0, n_rows // 2 - num_fit_rows // 2)
        row_idx = np.arange(row_lo, min(n_rows, row_lo + num_fit_rows))
        sel = np.zeros(scan['sinogram'].shape, dtype=bool)
        sel[np.ix_(view_idx, row_idx, np.arange(n_chan))] = True

        hits = np.zeros(scan['sinogram'].shape, dtype=bool)
        for L in path_lengths:
            hits |= (L > 0)
        hits = self._trim_shadow_edges(hits, edge_trim_percent)

        trans = np.exp(-scan['sinogram'])
        sel &= hits
        sel &= np.isfinite(trans) & (trans > 1e-6) & (trans < 1.5)
        if scan.get('valid_mask') is not None:
            sel &= scan['valid_mask']
        return sel

    @staticmethod
    def _trim_shadow_edges(hits, edge_trim_percent):
        """Erodes each view-row's contiguous shadow span by
        edge_trim_percent of its width at each end."""
        if edge_trim_percent <= 0:
            return hits
        out = np.zeros_like(hits)
        for vi in range(hits.shape[0]):
            for ri in range(hits.shape[1]):
                cols = np.nonzero(hits[vi, ri])[0]
                if cols.size == 0:
                    continue
                width = cols[-1] - cols[0] + 1
                trim = int(round(edge_trim_percent / 100.0 * width))
                out[vi, ri, cols[0] + trim:cols[-1] - trim + 1] = \
                    hits[vi, ri, cols[0] + trim:cols[-1] - trim + 1]
        return out

    # -- the pipeline -------------------------------------------------------

    def calibrate(self, learning_rate=0.02, max_iterations=5000,
                  stop_threshold=1e-6, num_fit_views=16, num_fit_rows=5,
                  edge_trim_percent=5.0, verbose=1):
        """Runs the calibration and returns the result.

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
            edge_trim_percent (float, optional): Percent of each
                target shadow's width dropped from each edge before
                fitting, since edge rays carry the most segmentation
                error.  Default 5.
            verbose (int, optional): 0 is silent, 1 prints progress.

        Returns:
            CalibrationResult: The complete calibration result.  Its
            ``est_system`` property is the estimated system as a
            fully specified :class:`~xcal.System`; it also holds
            everything about how the fit went: the cost, the ranked
            material combinations, the measured and predicted
            transmissions, the parameter table with provenance, and
            save().
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
                                    num_fit_rows, edge_trim_percent)
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

        return CalibrationResult(self, energies, solution,
                                 all_paths, selections, fit_scans)


class CalibrationResult:
    """Holds the output of :meth:`Calibrator.calibrate`.

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
        """Returns the estimated system, fully specified.

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
                apex_angle=source.apex_angle,
                physics_model=source.physics_model)
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
        """Returns the full parameter table with provenance.

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
        """Returns the parameter table as a string, one row per
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
            e_tab, counts = source.table()
            return np.interp(energies, e_tab, counts, left=0.0, right=0.0)
        if voltage is None:
            raise ValueError("voltage is required for tube sources.")
        if isinstance(source, ReflectionSource):
            table = _physics.reflection_source_table(
                voltage, [self._source_value], energies)
            return table[0]
        voltages, th_mm, e_tab, spectra = \
            _physics.transmission_source_table(
                apex_angle=source.apex_angle,
                physics_model=source.physics_model)
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
        """Returns the effective spectrum as a function of energy.

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
            voltage (float, optional): Peak tube voltage (kVp) in kV.  Required
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
        """Returns the estimated source spectrum as a function of
        energy
        in keV, normalized to unit area like the effective spectrum.

        Args:
            voltage (float, optional): Peak tube voltage (kVp) in kV.  Required
                for tube sources.

        Returns:
            callable: A function S with S(energies) -> density.
        """
        grid = self._dense_grid(voltage)
        values = self._source_spectrum_values(grid, voltage)
        area = np.trapezoid(values, grid)
        return _physics.SpectralFunction(grid, values / area)

    def filter_response(self, filt):
        """Returns one filter's estimated transmission as a function
        of
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
        """Returns the estimated detector response as a function of
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
        """Returns the measured and predicted transmission for the
        rays
        of one scan used in the fit.

        Args:
            scan (int): Index of the scan, in the order added
                (0 is the first).

        Returns:
            tuple: (measured, predicted) 1D numpy arrays, one entry per
            fit ray.
        """
        if getattr(self, '_loaded', None) is not None:
            s = self._loaded[scan]
            return s['measured'], s['predicted']
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

    def _n_scans(self):
        if getattr(self, '_loaded', None) is not None:
            return len(self._loaded)
        return len(self._fit_scans)

    def _scan_label(self, scan):
        v = self._cal.scans[scan].get('voltage')
        return f'{v:g} kV' if v is not None else f'scan {scan}'

    def _ray_coordinates(self, scan):
        """Returns (view, row, channel, sinogram_shape) for the fit
        rays of one scan, or None if unavailable."""
        if getattr(self, '_loaded', None) is not None:
            s = self._loaded[scan]
            return s.get('coordinates')
        if self._selections is None or self._selections[scan] is None:
            return None
        sel = self._selections[scan]
        view, row, channel = np.nonzero(sel)
        return view, row, channel, sel.shape

    def save(self, directory):
        """Saves the calibration to a directory.

        The directory is the single saved object.  It holds
        summary.txt (the parameter report), feasible_system.yaml
        (the search space the calibrator was given),
        est_system.yaml (the estimated system), fit_data.h5 (the
        analog fit data: the final cost, the energy grid, and per
        scan the measured and predicted transmission of every fit
        ray with its view, row, and channel in the sinogram), and
        plots/ (the effective spectra and the transmission fit).
        Every file is readable on its own; :meth:`load` rebuilds
        the result from the directory.

        Args:
            directory (str): Output directory; created if needed.
        """
        import h5py
        os.makedirs(os.path.join(directory, 'plots'), exist_ok=True)
        with open(os.path.join(directory, 'summary.txt'), 'w') as f:
            f.write(self.summary() + '\n')
        self._system.save(os.path.join(directory,
                                       'feasible_system.yaml'))
        self.est_system.save(os.path.join(directory,
                                          'est_system.yaml'))

        path = os.path.join(directory, 'fit_data.h5')
        with h5py.File(path, 'w') as f:
            f.attrs['xcal_fit_data_version'] = 1
            f.attrs['cost'] = self.cost
            f.create_dataset('energies', data=self._energies)
            for si in range(self._n_scans()):
                y, pred = self.transmission_fit(si)
                g = f.create_group(f'scan_{si}')
                s = self._cal.scans[si]
                if s.get('voltage') is not None:
                    g.attrs['voltage'] = float(s['voltage'])
                g.attrs['filters'] = ', '.join(
                    self._system.filter_label(sf)
                    for sf in s['filters'])
                g.create_dataset('measured',
                                 data=np.asarray(y, np.float32))
                g.create_dataset('predicted',
                                 data=np.asarray(pred, np.float32))
                coords = self._ray_coordinates(si)
                if coords is not None:
                    view, row, channel, shape = coords
                    g.attrs['sinogram_shape'] = shape
                    g.create_dataset(
                        'view', data=np.asarray(view, np.uint32))
                    g.create_dataset(
                        'row', data=np.asarray(row, np.uint32))
                    g.create_dataset(
                        'channel', data=np.asarray(channel, np.uint32))

        self.save_plots(directory)

    def _scan_spectrum(self, scan):
        """Returns the estimated effective spectrum of one scan,
        evaluated on
        the fit energy grid, normalized to integrate to one."""
        s = self._cal.scans[scan]
        R = self.effective_spectrum(voltage=s.get('voltage'),
                                    filters=s['filters'])
        return R(self._energies)

    def save_plots(self, directory, compare_to=None):
        """Writes the calibration plots to <directory>/plots.

        Two files.  spectrum.png: the estimated effective spectrum
        of each scan; if ``compare_to`` is given, each scan's
        estimate is drawn beside that system's spectrum with their
        NRMSE.  transmission_fit.png: the measured versus predicted
        transmission of the fit rays.  :meth:`save` calls this with
        no comparison; call it directly to regenerate the plots.

        Args:
            directory (str): Output directory; its plots subfolder
                is created if needed.
            compare_to (System, optional): A fully specified
                reference system, e.g. the ground truth of a
                simulation.
        """
        import matplotlib.pyplot as plt
        plots_dir = os.path.join(directory, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        E = self._energies
        if compare_to is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            for si in range(self._n_scans()):
                ax.plot(E, self._scan_spectrum(si),
                        label=self._scan_label(si))
            ax.set_xlabel('Energy (keV)')
            ax.set_ylabel('Effective spectrum (1/keV)')
            ax.legend()
            ax.grid(True)
        else:
            n = self._n_scans()
            fig, axes = plt.subplots(1, n, figsize=(5 * n, 4),
                                     squeeze=False)
            for si, ax in enumerate(axes[0]):
                v = self._cal.scans[si].get('voltage')
                ref = compare_to.effective_spectrum(voltage=v)(E)
                est = self._scan_spectrum(si)
                nrmse = (np.linalg.norm(est - ref)
                         / np.linalg.norm(ref))
                ax.plot(E, ref, label='reference')
                ax.plot(E, est, '--', label='estimate')
                ax.set_title(f'{self._scan_label(si)},  '
                             f'NRMSE {nrmse:.4f}')
                ax.set_xlabel('Energy (keV)')
                ax.legend()
                ax.grid(True)
            axes[0][0].set_ylabel('Effective spectrum (1/keV)')
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, 'spectrum.png'), dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        for si in range(self._n_scans()):
            y, pred = self.transmission_fit(si)
            ax.plot(y, pred, '.', markersize=2,
                    label=self._scan_label(si))
        lim = [0, 1.05]
        ax.plot(lim, lim, 'k-', linewidth=0.5)
        ax.set_xlabel('Measured transmission')
        ax.set_ylabel('Predicted transmission')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, 'transmission_fit.png'),
                    dpi=130)
        plt.close(fig)

    @classmethod
    def load(cls, directory):
        """Loads a calibration saved by :meth:`save`.

        The result is rebuilt from feasible_system.yaml,
        est_system.yaml, and fit_data.h5, so the parameter table,
        the response functions, and the transmission fit all work
        in a new session without rerunning the calibration.

        Args:
            directory (str): Path to a saved calibration directory.

        Returns:
            CalibrationResult: The loaded result.
        """
        import h5py
        from .system import load_system
        feasible = load_system(os.path.join(directory,
                                            'feasible_system.yaml'))
        est = load_system(os.path.join(directory, 'est_system.yaml'))

        with h5py.File(os.path.join(directory, 'fit_data.h5'),
                       'r') as f:
            energies = np.array(f['energies'])
            cost = float(f.attrs['cost'])
            scans = []
            si = 0
            while f'scan_{si}' in f:
                g = f[f'scan_{si}']
                coordinates = None
                if 'view' in g:
                    coordinates = (np.array(g['view']),
                                   np.array(g['row']),
                                   np.array(g['channel']),
                                   tuple(g.attrs['sinogram_shape']))
                scans.append({
                    'voltage': (float(g.attrs['voltage'])
                                if 'voltage' in g.attrs else None),
                    'filter_labels': str(g.attrs.get('filters', '')),
                    'measured': np.array(g['measured']),
                    'predicted': np.array(g['predicted']),
                    'coordinates': coordinates,
                })
                si += 1

        source = feasible.source
        if isinstance(source, ReflectionSource):
            source_value = est.source.takeoff_angle
        elif isinstance(source, TransmissionSource):
            source_value = est.source.target_thickness
        else:
            source_value = 0.0

        def candidate_index(feasible_part, est_part, what):
            formula = est_part.materials[0].formula
            for j, m in enumerate(feasible_part.materials):
                if m.formula == formula:
                    return j
            raise ValueError(
                f"est_system.yaml names {what} material "
                f"{formula!r}, which is not among the candidates "
                f"in feasible_system.yaml.")

        combo = tuple(
            candidate_index(f, ef, 'a filter')
            for f, ef in zip(feasible.filters, est.filters))
        combo = combo + (candidate_index(feasible.detector,
                                         est.detector, 'the detector'),)
        solution = {
            'combo': combo,
            'cost': cost,
            'source_value': source_value,
            'filter_thicknesses': [f.thickness for f in est.filters],
            'detector_thickness': est.detector.thickness,
            'iterations': 0,
            'all': [],
        }

        label_to_filter = {feasible.filter_label(f): f
                           for f in feasible.filters}

        class _LoadedCal:
            pass
        cal = _LoadedCal()
        cal.system = feasible
        cal.scans = []
        for s in scans:
            labels = [x.strip() for x in s['filter_labels'].split(',')
                      if x.strip()]
            filts = [label_to_filter[x] for x in labels
                     if x in label_to_filter]
            cal.scans.append({'voltage': s['voltage'],
                              'filters': filts or
                              list(feasible.filters)})
        result = cls(cal, energies, solution, paths=None,
                     selections=None, fit_scans=None)
        result._loaded = scans
        return result

    def show(self, block=True):
        """Prints the parameter table and plots the effective spectra
        and the transmission fit.  Everything shown is also available
        as data through the methods on this class.  Rod masks are
        reviewed before calibration, at the segmentation step.

        Args:
            block (bool, optional): Wait for the window to be closed.
        """
        print(self.summary())
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for si in range(self._n_scans()):
            spec = self._effective_values_for_scan(si)
            axes[0].plot(self._energies, spec,
                         label=self._scan_label(si))
            y, pred = self.transmission_fit(si)
            axes[1].plot(y, pred, '.', markersize=2,
                         label=self._scan_label(si))
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
