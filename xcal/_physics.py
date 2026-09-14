"""Internal physics: material coefficient curves, component responses,
and source spectrum tables.

Everything here is numpy in, numpy out.  The differentiable fit layer
in _fit.py builds torch expressions from the coefficient curves
computed here.
"""

import os

import numpy as np

from .utils import get_lin_att_c_vs_E, get_lin_absp_c_vs_E

# Validity range of the NIST tables shipped in mu_en.h5, in keV.
ENERGY_MIN_KEV = 1.0
ENERGY_MAX_KEV = 20000.0


def check_energies(energies):
    """Validate an energy grid against the NIST table range.

    The tables cover 1 keV to 20 MeV.  Outside that range the
    interpolation would silently return wrong coefficients, so this
    raises instead.
    """
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    if energies.ndim != 1 or len(energies) == 0:
        raise ValueError("energies must be a 1D array in keV.")
    if np.any(np.diff(energies) <= 0):
        raise ValueError("energies must be strictly increasing.")
    if energies[0] < ENERGY_MIN_KEV or energies[-1] > ENERGY_MAX_KEV:
        raise ValueError(
            f"energies must lie within the NIST table range "
            f"[{ENERGY_MIN_KEV}, {ENERGY_MAX_KEV}] keV, got "
            f"[{energies[0]}, {energies[-1]}].")
    return energies


def default_energy_grid(max_voltage):
    """Return the default fit grid: 1 keV bins from 1.5 keV to just
    below the highest voltage, the convention used in the XCal paper."""
    max_voltage = int(round(max_voltage))
    return np.linspace(1.5, max_voltage - 0.5, max_voltage - 1)


def attenuation_coefficients(material, energies):
    """Return the linear attenuation coefficient curve of a resolved
    Material, in 1/mm."""
    energies = check_energies(energies)
    return get_lin_att_c_vs_E(material.density, material.formula, energies)


def filter_transmission(material, thickness, energies):
    """Return a filter's transmission exp(-mu * t) for thickness in
    mm."""
    mu = attenuation_coefficients(material, energies)
    return np.exp(-mu * thickness)


def scintillator_curves(material, energies):
    """Return the two coefficient curves the scintillator response is
    built from: (mu, mu_en_over_mu_times_E).

    The response for thickness t is
    mu_en/mu * E * (1 - exp(-mu * t)), the converted energy per
    incident photon of energy E.
    """
    energies = check_energies(energies)
    mu = get_lin_att_c_vs_E(material.density, material.formula, energies)
    mu_en = get_lin_absp_c_vs_E(material.density, material.formula, energies)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio_e = np.where(mu > 0, mu_en / mu, 0.0) * energies
    return mu, ratio_e


def scintillator_response(material, thickness, energies):
    """Return a scintillator's response for thickness in mm."""
    mu, ratio_e = scintillator_curves(material, energies)
    return ratio_e * (1.0 - np.exp(-mu * thickness))




def mm_per_alu(ct_model):
    """Return how many mm one of the model's length units (ALU)
    represents, from its alu_unit and alu_value parameters.  Warns
    when the model declares no unit, because silent unit mistakes
    corrupt every path length."""
    import warnings
    unit, value = ct_model.get_params(['alu_unit', 'alu_value'])
    if unit is None:
        warnings.warn(
            "the tomography model declares no alu_unit; xcal is "
            "assuming 1 ALU = 1 mm.  Set alu_unit and alu_value on "
            "the model to make the units explicit.")
        return 1.0
    factors = {'um': 1e-3, 'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
    if unit not in factors:
        raise ValueError(
            f"the model's alu_unit is {unit!r}; supported units are "
            f"{sorted(factors)}.")
    return float(value) * factors[unit]

# ---------------------------------------------------------------------------
# Source spectrum tables
# ---------------------------------------------------------------------------

def reflection_source_table(voltage, takeoff_angles, energies):
    """Generate reflection source spectra with Spekpy at one voltage
    over a grid of takeoff angles.

    Args:
        voltage (float): Peak tube voltage (kVp) in kV.
        takeoff_angles (numpy.ndarray): Anode takeoff angles in degrees.
        energies (numpy.ndarray): Energy grid in keV, 1 keV spacing.

    Returns:
        numpy.ndarray: Array of shape (len(takeoff_angles),
        len(energies)); relative photon flux, zero above the voltage.
    """
    import spekpy as sp
    energies = check_energies(energies)
    table = np.zeros((len(takeoff_angles), len(energies)))
    for i, angle in enumerate(takeoff_angles):
        s = sp.Spek(kvp=float(voltage), th=float(angle), dk=1, mas=1,
                    char=True)
        k, phi_k = s.get_spectrum(edges=False)
        table[i] = np.interp(energies, k, phi_k, left=0.0, right=0.0)
    return table


_SOURCE_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'source_models')


def available_transmission_models():
    """Return the physics model names of the shipped transmission
    source tables (files transmission_<model>.csv in
    xcal/source_models)."""
    import glob
    paths = glob.glob(os.path.join(_SOURCE_MODELS_DIR,
                                   'transmission_*.csv'))
    return sorted(os.path.basename(p)[len('transmission_'):-len('.csv')]
                  for p in paths)


def transmission_source_table(apex_angle=10.0,
                              physics_model='G4EmLivermorePhysics'):
    """Load one transmission source table.

    Each physics model is one CSV file
    xcal/source_models/transmission_<model>.csv with columns
    apex_angle_deg, voltage_kV, target_thickness_um, energy_keV,
    photons.

    Returns:
        tuple: (voltages_kV, thicknesses_mm, energies_keV, spectra)
        where spectra has shape (len(voltages), len(thicknesses),
        len(energies)) on a 1 keV grid.
    """
    import csv as _csv
    path = os.path.join(_SOURCE_MODELS_DIR,
                        f'transmission_{physics_model}.csv')
    if not os.path.exists(path):
        raise ValueError(
            f"physics model {physics_model!r} has no table; "
            f"available models: {available_transmission_models()}.")

    rows = []
    with open(path) as f:
        for line in _csv.reader(r for r in f if not r.startswith('#')):
            if line[0] == 'apex_angle_deg':
                continue
            rows.append([float(x) for x in line])
    rows = np.array(rows)
    angles = sorted(set(rows[:, 0]))
    if float(apex_angle) not in angles:
        raise ValueError(
            f"apex angle {apex_angle!r} is not in {path}; available "
            f"angles: {angles} degrees.")
    rows = rows[rows[:, 0] == float(apex_angle)]

    voltages = np.array(sorted(set(rows[:, 1])))
    thicknesses_um = np.array(sorted(set(rows[:, 2])))
    thicknesses_mm = thicknesses_um * 1e-3
    max_v = int(voltages.max())
    fine = np.round(np.arange(1, 10 * max_v + 1) * 0.1, 6)
    energies = np.linspace(1.0, max_v, max_v)

    spectra = np.zeros((len(voltages), len(thicknesses_mm),
                        len(energies)))
    for vi, v in enumerate(voltages):
        for ti, th in enumerate(thicknesses_um):
            sel = (rows[:, 1] == v) & (rows[:, 2] == th)
            dense = np.zeros(len(fine))
            idx = np.searchsorted(fine, np.round(rows[sel, 3], 6))
            dense[idx] = rows[sel, 4]
            # Sum each 10 fine bins into 1 keV bins, skipping the
            # first 0.9 keV so bins center on integer keV.
            n_bins = (len(dense) - 9) // 10
            binned = dense[9:9 + n_bins * 10].reshape(n_bins,
                                                      10).sum(axis=1)
            spectra[vi, ti, :min(n_bins, len(energies))] = \
                binned[:len(energies)]
    return voltages, thicknesses_mm, energies, spectra


class SpectralFunction:
    """A spectral quantity as a function of energy in keV.

    Calling it with a scalar or array of energies returns the density
    at those energies, zero outside the tabulated support.
    """

    def __init__(self, energies, values):
        self.energies = np.asarray(energies, dtype=float)
        self.values = np.asarray(values, dtype=float)

    def __call__(self, energies):
        energies = np.asarray(energies, dtype=float)
        out = np.interp(np.atleast_1d(energies), self.energies,
                        self.values, left=0.0, right=0.0)
        return out.reshape(energies.shape) if energies.shape else out[0]


def prepare_for_interpolation(spec_list):
    """Extend each spectrum below the next-higher voltage's endpoint so
    that linear interpolation between voltages keeps the interpolated
    spectrum zero above the interpolated voltage.

    This is the negative-extension construction from the XCal paper
    (Appendix A).  spec_list has shape (n_voltages, n_energies), sorted
    by increasing voltage.
    """
    spec_list = np.array(spec_list, dtype=float, copy=True)

    def last_nonzero(a):
        nz = np.nonzero(a)[0]
        return nz[-1] if len(nz) else -1

    for s in range(len(spec_list) - 1):
        v0 = last_nonzero(spec_list[s])
        v1 = last_nonzero(spec_list[s + 1])
        f1 = spec_list[s + 1]
        # Extend strictly above this spectrum's own cutoff bin.  (The
        # v1 code started at v0 and zeroed the cutoff bin itself.)
        for v in range(v0 + 1, v1):
            r = (v - float(v0)) / (v1 - float(v0))
            spec_list[s][v] = -r / (1 - r) * f1[v]
    return spec_list


def interpolate_rows(x_grid, table, x):
    """Linearly interpolate the rows of a table at one coordinate.

    Args:
        x_grid (numpy.ndarray): Sorted coordinates, one per table row.
        table (numpy.ndarray): Shape (len(x_grid), n).
        x (float): Coordinate to evaluate at; clamped to the grid range.

    Returns:
        numpy.ndarray: The interpolated row of length n.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    if len(x_grid) == 1:
        return np.array(table[0], dtype=float)
    x = float(np.clip(x, x_grid[0], x_grid[-1]))
    i = int(np.clip(np.searchsorted(x_grid, x) - 1, 0, len(x_grid) - 2))
    a = (x - x_grid[i]) / (x_grid[i + 1] - x_grid[i])
    return (1 - a) * table[i] + a * table[i + 1]


def available_synchrotron_spectra():
    """Return the names of the shipped synchrotron spectra (files
    synchrotron_<name>.csv in xcal/source_models)."""
    import glob
    paths = glob.glob(os.path.join(_SOURCE_MODELS_DIR,
                                   'synchrotron_*.csv'))
    return sorted(os.path.basename(p)[len('synchrotron_'):-len('.csv')]
                  for p in paths)


def synchrotron_source_table(spectrum='als_bm832'):
    """Load one synchrotron spectrum, rebinned to uniform 1 keV bins.

    A spectrum is one CSV file
    xcal/source_models/synchrotron_<name>.csv with columns
    energy_keV (bin edge in keV) and photon_counts (counts in the
    bin starting at that edge).

    Returns:
        tuple: (energies, counts).  Bin center energies 0.5, 1.5,
        ..., and photon counts per bin, total counts preserved.
    """
    import csv as _csv
    path = os.path.join(_SOURCE_MODELS_DIR,
                        f'synchrotron_{spectrum}.csv')
    if not os.path.exists(path):
        raise ValueError(
            f"spectrum {spectrum!r} has no file; available "
            f"spectra: {available_synchrotron_spectra()}.")

    energies, counts = [], []
    with open(path) as f:
        for line in _csv.reader(r for r in f if not r.startswith('#')):
            if line[0] == 'energy_keV':
                continue
            energies.append(float(line[0]))
            counts.append(float(line[1]))
    energies = np.array(energies)
    counts = np.array(counts)

    top = int(np.ceil(energies.max()))
    edges = np.linspace(0, top, num=top + 1)
    rebinned = np.zeros(top)
    for i in range(len(counts) - 1):
        start, end = energies[i], energies[i + 1]
        j0 = np.searchsorted(edges, start, side='right') - 1
        j1 = np.searchsorted(edges, end, side='left')
        for j in range(j0, j1):
            overlap = ((min(end, edges[j + 1]) - max(start, edges[j]))
                       / (end - start))
            rebinned[j] += overlap * counts[i]
    return edges[1:] - 0.5, rebinned
