"""Internal differentiable fit.

The calibrator hands this module a discretized problem: per scan, a
forward matrix, measured transmissions, and weights; per component,
candidate materials with precomputed coefficient curves and a
continuous thickness (or angle) that is fixed or bounded.  Discrete
material choices are searched exhaustively; for each combination the
continuous parameters are fit with Adam on CPU torch.
"""

import itertools

import numpy as np
import torch


class _ClampWithGrad(torch.autograd.Function):
    """Clamp to [0, 1] in the forward pass, identity gradient in the
    backward pass, so Adam converges onto the constraint set instead of
    stalling at the boundary."""

    @staticmethod
    def forward(ctx, x):
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, g):
        return g


def _clamp01(x):
    return _ClampWithGrad.apply(x)


class _Bounded:
    """A continuous parameter with bounds, stored as a raw torch
    parameter whose clamped value maps linearly onto [low, high]."""

    def __init__(self, low, high, initial):
        self.low = float(low)
        self.high = float(high)
        if self.high > self.low:
            raw0 = (float(initial) - self.low) / (self.high - self.low)
            self.raw = torch.nn.Parameter(torch.tensor(raw0,
                                                       dtype=torch.float64))
        else:
            self.raw = None

    def value(self):
        if self.raw is None:
            return torch.tensor(self.low, dtype=torch.float64)
        return self.low + (self.high - self.low) * _clamp01(self.raw)

    def parameters(self):
        return [] if self.raw is None else [self.raw]


def _make_bounded(spec):
    """Build a _Bounded from a float (fixed) or an object with
    low/high/initial attributes (xcal.estimate)."""
    if hasattr(spec, 'low'):
        return _Bounded(spec.low, spec.high, spec.initial)
    return _Bounded(float(spec), float(spec), float(spec))


def _interp_row(grid, table, x):
    """Differentiable linear interpolation of table rows at scalar x.

    grid is a sorted 1D tensor, table is (len(grid), nE), x a scalar
    tensor.  x is clamped to the grid range.
    """
    if table.shape[0] == 1:
        return table[0]
    x = x.clamp(grid[0], grid[-1])
    idx = torch.searchsorted(grid, x.detach()).clamp(1, len(grid) - 1)
    x0, x1 = grid[idx - 1], grid[idx]
    a = (x - x0) / (x1 - x0)
    return (1 - a) * table[idx - 1] + a * table[idx]


class FitProblem:
    """One discretized calibration problem.

    Args:
        energies (numpy.ndarray): Energy grid in keV, shape (nE,).
        scans (list of dict): Per scan:
            'A' (nR, nE) attenuation factors, 'y' (nR,) measured
            transmission, 'w' (nR,) weights, 'filter_indices' list of
            indices into filters, 'source' one of
            ('fixed', spectrum (nE,)) or
            ('table', grid (nG,), table (nG, nE)) interpolated at the
            source's continuous parameter.
        source_param: None, a float, or an estimate for the source's
            continuous parameter (takeoff angle or target thickness).
        filters (list of dict): Per filter: 'mu_candidates' list of
            (nE,) attenuation curves, one per material candidate, and
            'thickness' as float or estimate.
        detector (dict): 'curve_candidates' list of (mu, ratio_e)
            pairs of (nE,) arrays, and 'thickness' as float or
            estimate.
    """

    def __init__(self, energies, scans, source_param, filters, detector):
        self.energies_np = np.asarray(energies, dtype=float)
        self.E = torch.tensor(self.energies_np, dtype=torch.float64)
        self.scans = []
        for s in scans:
            scan = dict(s)
            scan['A'] = torch.tensor(np.asarray(s['A'], dtype=float))
            scan['y'] = torch.tensor(np.asarray(s['y'], dtype=float))
            scan['w'] = torch.tensor(np.asarray(s['w'], dtype=float))
            kind = s['source'][0]
            if kind == 'fixed':
                scan['source'] = ('fixed',
                                  torch.tensor(np.asarray(s['source'][1],
                                                          dtype=float)))
            else:
                grid = torch.tensor(np.asarray(s['source'][1], dtype=float))
                table = torch.tensor(np.asarray(s['source'][2], dtype=float))
                scan['source'] = ('table', grid, table)
            self.scans.append(scan)
        self.source_param = source_param
        self.filters = filters
        self.detector = detector

    def _combinations(self):
        """All discrete material combinations: one candidate index per
        filter plus one for the detector."""
        pools = [range(len(f['mu_candidates'])) for f in self.filters]
        pools.append(range(len(self.detector['curve_candidates'])))
        return list(itertools.product(*pools))

    def _loss(self, combo, theta_s, filter_ts, detector_t):
        combo_filters = combo[:-1]
        det_choice = combo[-1]
        mu_d, ratio_e_d = self.detector['curve_candidates'][det_choice]
        mu_d = torch.tensor(mu_d)
        ratio_e_d = torch.tensor(ratio_e_d)
        det_resp = ratio_e_d * (1 - torch.exp(-mu_d * detector_t.value()))

        loss = 0.0
        for scan in self.scans:
            if scan['source'][0] == 'fixed':
                spec = scan['source'][1]
            else:
                _, grid, table = scan['source']
                spec = _interp_row(grid, table, theta_s.value())
            for fi in scan['filter_indices']:
                mu_f = torch.tensor(
                    self.filters[fi]['mu_candidates'][combo_filters[fi]])
                spec = spec * torch.exp(-mu_f * filter_ts[fi].value())
            spec = spec * det_resp
            spec = spec / torch.trapz(spec, self.E)
            y_pred = torch.trapz(scan['A'] * spec, self.E, dim=-1)
            loss = loss + 0.5 * torch.mean(
                scan['w'] * (y_pred - scan['y']) ** 2)
        return loss

    def _fit_one(self, combo, learning_rate, max_iterations,
                 stop_threshold):
        theta_s = _make_bounded(self.source_param
                                if self.source_param is not None else 0.0)
        filter_ts = [_make_bounded(f['thickness']) for f in self.filters]
        detector_t = _make_bounded(self.detector['thickness'])
        params = (theta_s.parameters()
                  + [p for t in filter_ts for p in t.parameters()]
                  + detector_t.parameters())

        if not params:
            loss = self._loss(combo, theta_s, filter_ts, detector_t)
            return float(loss), theta_s, filter_ts, detector_t, 0

        optimizer = torch.optim.Adam(params, lr=learning_rate)
        last = [p.detach().clone() for p in params]
        loss = None
        for it in range(1, max_iterations + 1):
            optimizer.zero_grad()
            loss = self._loss(combo, theta_s, filter_ts, detector_t)
            if not torch.isfinite(loss):
                return float('inf'), theta_s, filter_ts, detector_t, it
            loss.backward()
            optimizer.step()
            # Project the raw parameters back into [0, 1] so Adam does
            # not drift far outside the box and walk back for many
            # iterations.
            with torch.no_grad():
                for p in params:
                    p.clamp_(0.0, 1.0)
            moved = max(float((p.detach() - q.clamp(0, 1)).abs().max())
                        for p, q in zip(params, last))
            last = [p.detach().clone() for p in params]
            if moved < stop_threshold:
                break
        with torch.no_grad():
            final = float(self._loss(combo, theta_s, filter_ts, detector_t))
        return final, theta_s, filter_ts, detector_t, it

    def solve(self, learning_rate=0.02, max_iterations=2000,
              stop_threshold=1e-6, verbose=1):
        """Search all discrete combinations and return the best fit.

        Returns:
            dict: 'cost', 'combo' (candidate index per filter plus
            detector), 'source_value', 'filter_thicknesses',
            'detector_thickness', 'iterations', and 'all' with
            (combo, cost) for every combination.
        """
        combos = self._combinations()
        if verbose:
            print(f"xcal: fitting {len(combos)} material combination(s)")
        results = []
        for ci, combo in enumerate(combos):
            cost, theta_s, filter_ts, detector_t, iters = self._fit_one(
                combo, learning_rate, max_iterations, stop_threshold)
            results.append((combo, cost, theta_s, filter_ts, detector_t,
                            iters))
            if verbose:
                print(f"xcal:   combination {ci + 1}/{len(combos)} "
                      f"{combo}: cost {cost:.3e} after {iters} iterations")
        best = min(results, key=lambda r: r[1])
        combo, cost, theta_s, filter_ts, detector_t, iters = best
        return {
            'cost': cost,
            'combo': combo,
            'source_value': float(theta_s.value()),
            'filter_thicknesses': [float(t.value()) for t in filter_ts],
            'detector_thickness': float(detector_t.value()),
            'iterations': iters,
            'all': [(r[0], r[1]) for r in results],
        }
