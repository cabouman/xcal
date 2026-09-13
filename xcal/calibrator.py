"""The calibrator and its result.

The :class:`Calibrator` is built from a :class:`~xcal.System` and a list
of :class:`~xcal.Rod` objects.  Scans are added as (sinogram, model)
pairs produced by mbirtorch preprocessing, and :meth:`Calibrator.calibrate`
returns a :class:`CalibrationResult`.
"""

import numpy as np

from . import _physics
from . import _segment
from .system import (estimate, Filter, Scintillator, ReflectionSource,
                     TransmissionSource, SynchrotronSource, System)

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
        rods (list of Rod): The rods in the calibration object.

    Example:
        >>> cal = xcal.Calibrator(system, rods)
        >>> cal.add_scan(sino, ct_model, voltage=80)
        >>> result = cal.calibrate()
    """

    def __init__(self, system, rods):
        if not isinstance(system, System):
            raise TypeError(f"system must be an xcal.System, got "
                            f"{system!r}.")
        rods = list(rods)
        if not rods:
            raise ValueError("rods must contain at least one Rod.")
        self.system = system
        self.rods = rods
        self.scans = []

    def add_scan(self, sinogram, ct_model, voltage=None, rods=None,
                 filters=None, weights=None):
        """Add one calibration scan.

        The sinogram and model are the pair returned by mbirtorch
        preprocessing, for example ``mtp.zeiss.get_sino_and_model(...)``.
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
                this scan.  Any supported geometry works; xcal uses only
                its recon and forward projection methods.
            voltage (float, optional): Source voltage for this scan in
                kV.  Required for tube sources, ignored for synchrotron
                sources.
            rods (list of Rod, optional): The rods present in this scan.
                Defaults to all rods given to the constructor.
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
        scan_rods = list(rods) if rods is not None else list(self.rods)
        for r in scan_rods:
            if not any(r is rr for rr in self.rods):
                raise ValueError(
                    f"{r!r} was not in the rods list given to the "
                    f"Calibrator.")
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
        self.scans.append({
            'sinogram': sinogram,
            'ct_model': ct_model,
            'voltage': voltage,
            'rods': scan_rods,
            'filters': scan_filters,
            'weights': weights,
        })

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _mm_per_alu(ct_model):
        """Return how many mm one of the model's length units (ALU)
        represents, from the model's alu_unit and alu_value
        parameters.  Warns when the model declares no unit, because
        silent unit mistakes corrupt every path length."""
        import warnings
        unit, value = ct_model.get_params(['alu_unit', 'alu_value'])
        if unit is None:
            warnings.warn(
                "the tomography model declares no alu_unit; xcal is "
                "assuming 1 ALU = 1 mm.  Set alu_unit and alu_value "
                "on the model to make the units explicit.")
            return 1.0
        factors = {'um': 1e-3, 'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
        if unit not in factors:
            raise ValueError(
                f"the model's alu_unit is {unit!r}; supported units "
                f"are {sorted(factors)}.")
        return float(value) * factors[unit]

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
        views and center rows, rays that hit at least one rod, and
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
            # A ray that clips a rod's edge has a path length dominated
            # by segmentation error; exclude rays below 30 percent of
            # that rod's maximum path.
            grazing |= (L > 0) & (L < 0.3 * L.max())
        sel &= hits & ~grazing
        sel &= np.isfinite(trans) & (trans > 1e-6) & (trans < 1.5)
        return sel

    # -- the pipeline -------------------------------------------------------

    def calibrate(self, learning_rate=0.02, max_iterations=5000,
                  stop_threshold=1e-6, num_fit_views=16, num_fit_rows=5,
                  verbose=1):
        """Run the calibration and return the result.

        The steps are: reconstruct each scan, segment the rods, forward
        project the segmentation masks to get per-ray path lengths in
        mm, then jointly fit the system parameters to all scans.
        Candidate materials are searched exhaustively; continuous
        parameters are fit with Adam within their bounds.

        Args:
            learning_rate (float, optional): Adam step size.
            max_iterations (int, optional): Iteration cap per material
                combination.
            stop_threshold (float, optional): Stop when no normalized
                parameter moves more than this in one iteration.
            num_fit_views (int, optional): Number of views per scan
                used in the spectral fit.  The reconstruction and
                segmentation always use all views.
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

        # Reconstruct, segment, and compute path lengths per scan.
        recons, segmentations, all_paths = [], [], []
        for si, scan in enumerate(self.scans):
            if verbose:
                print(f"xcal: reconstructing scan {si} "
                      f"({scan['sinogram'].shape[0]} views)")
            mm_per_alu = self._mm_per_alu(scan['ct_model'])
            recon, _ = scan['ct_model'].recon(scan['sinogram'])
            # The reconstruction is in 1/ALU; convert to 1/mm so it is
            # comparable with the NIST attenuation coefficients.
            recon = _as_numpy(recon) / mm_per_alu
            mm_per_voxel = (float(scan['ct_model'].get_params(
                'delta_voxel')) * mm_per_alu)
            labels, masks = _segment.segment_rods(
                recon, scan['rods'], mm_per_voxel, energies,
                verbose=verbose)
            # Forward projection returns path lengths in ALU; convert
            # to mm.
            paths = [_as_numpy(scan['ct_model'].forward_project(m))
                     * mm_per_alu for m in masks]
            recons.append(recon)
            segmentations.append(labels)
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
            mu_rods = [_physics.attenuation_coefficients(r.material,
                                                         energies)
                       for r in scan['rods']]
            total = np.zeros((trans.size, len(energies)))
            for L, mu in zip(paths, mu_rods):
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

        return CalibrationResult(self, energies, solution, recons,
                                 segmentations, all_paths, selections,
                                 fit_scans)


class CalibrationResult:
    """The output of :meth:`Calibrator.calibrate`.

    The result returns data and functions; it does not plot.  The
    spectral quantities are returned as functions of energy that the
    user evaluates and plots as they choose.  The one display
    convenience is :meth:`show`.

    The returned functions satisfy R(E) proportional to
    S(E) * product of filter transmissions * D(E), with the effective
    spectrum R normalized to integrate to one.
    """

    def __init__(self, calibrator, energies, solution, recons,
                 segmentations, paths, selections, fit_scans):
        self._cal = calibrator
        self._system = calibrator.system
        self._energies = np.asarray(energies)
        self._solution = solution
        self._recons = recons
        self._segmentations = segmentations
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
        """Return a table of the estimated parameters as a string."""
        lines = ['Estimated parameters:']
        for key, value in self.params.items():
            if isinstance(value, float):
                lines.append(f'  {key}: {value:.6g}')
            else:
                lines.append(f'  {key}: {value}')
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

    def reconstruction(self, scan):
        """Return the reconstruction of one scan as a numpy volume.

        Args:
            scan (int): Index of the scan, in the order added
                (0 is the first).
        """
        if self._recons is None:
            raise ValueError("reconstructions are not stored in a saved "
                             "result; rerun the calibration to view them.")
        return self._recons[scan]

    def segmentation(self, scan):
        """Return the rod segmentation of one scan as a numpy label
        volume, 0 for background and k+1 for the k-th rod of that
        scan.

        Args:
            scan (int): Index of the scan, in the order added
                (0 is the first).
        """
        if self._segmentations is None:
            raise ValueError("segmentations are not stored in a saved "
                             "result; rerun the calibration to view them.")
        return self._segmentations[scan]

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
        works in a new session.  The reconstructions and segmentations
        are not stored, so :meth:`reconstruction`,
        :meth:`segmentation`, and :meth:`show` are unavailable on a
        loaded result.

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
        result = cls(cal, energies, solution, recons=None,
                     segmentations=None, paths=None, selections=None,
                     fit_scans=None)
        result._loaded_scans = scans
        return result

    def show(self, block=True):
        """Show the complete result for review.

        Opens the slice viewer on the segmented rods, prints the
        parameter table, and plots the effective spectra and the
        transmission fit.  Everything shown here is also available as
        data through the methods on this class.  On a machine without
        a display, use the data methods and save figures yourself.

        Args:
            block (bool, optional): Wait for the windows to be closed.
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
        plt.show(block=False)

        try:
            import mbirtorch as mt
            for si, (recon, labels) in enumerate(
                    zip(self._recons, self._segmentations)):
                mt.slice_viewer(recon, labels.astype(np.float32),
                                slice_axis=2, block=block and
                                si == len(self._recons) - 1,
                                title=f'Scan {si}: recon and segmentation')
        except Exception as e:
            print(f"(slice viewer unavailable: {e})")
        if block:
            plt.show(block=True)
