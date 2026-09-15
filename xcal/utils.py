"""Provides physical-parameter access and data helpers.

The physical parameters live as data files in xcal/physical_params
(the periodic table, the NIST attenuation tables, the materials
catalog).  The functions here read them.  Source spectrum tables
(xcal/source_models) are read by xcal._physics.
"""

import os

import numpy as np

_PHYSICAL_PARAMS_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'physical_params')

_periodic_table_cache = None


def periodic_table():
    """Returns the periodic table as {symbol: {'atomic_weight',
    'density'}}, atomic weight in g/mol and density in g/cm^3, read
    once from physical_params/periodic_table.yaml."""
    global _periodic_table_cache
    if _periodic_table_cache is None:
        import yaml
        path = os.path.join(_PHYSICAL_PARAMS_DIR, 'periodic_table.yaml')
        with open(path) as f:
            _periodic_table_cache = yaml.safe_load(f)['elements']
    return _periodic_table_cache


def atomic_weights():
    """Returns {element symbol: atomic weight in g/mol}."""
    return {s: e['atomic_weight'] for s, e in periodic_table().items()}


def element_densities():
    """Returns {element symbol: density in g/cm^3} for the elements
    that have one."""
    return {s: e['density'] for s, e in periodic_table().items()
            if 'density' in e}


_nist_tables_cache = None


def nist_tables():
    """Returns the NIST tables as {element: array of shape (n, 3)}
    with columns energy in keV, mass attenuation, and mass
    energy-absorption in cm^2/g, read once from
    physical_params/nist_attenuation.csv."""
    global _nist_tables_cache
    if _nist_tables_cache is None:
        import csv
        path = os.path.join(_PHYSICAL_PARAMS_DIR,
                            'nist_attenuation.csv')
        rows = {}
        with open(path) as f:
            for line in csv.reader(
                    r for r in f if not r.startswith('#')):
                if line[0] == 'element':
                    continue
                rows.setdefault(line[0], []).append(
                    [float(x) for x in line[1:]])
        _nist_tables_cache = {el: np.array(v) for el, v in rows.items()}
    return _nist_tables_cache


def nist_element_symbols():
    """Returns the set of element symbols the NIST tables cover
    (hydrogen through uranium, plus 'Air')."""
    return set(nist_tables().keys())


def interpret_formula(formula):
    """Returns a chemical formula as an {element: count} dict.  A
    dict passes through unchanged."""
    if isinstance(formula, dict):
        return formula
    import chemparse
    return chemparse.parse_formula(formula)


def molecular_mass(formula):
    """Returns the molecular mass of a formula in g/mol."""
    weights = atomic_weights()
    return sum(count * weights[element]
               for element, count in interpret_formula(formula).items())


def _mass_coefficient(formula, energies, column):
    """Returns the mass-weighted NIST coefficient curve of a
    compound, in cm^2/g, log-log interpolated at the given
    energies in keV."""
    parsed = interpret_formula(formula)
    weights = atomic_weights()
    total_mass = molecular_mass(parsed)
    tables = nist_tables()
    out = np.zeros(len(energies), dtype=float)
    for element, count in parsed.items():
        fraction = count * weights[element] / total_mass
        table = tables[element]
        log_interp = np.interp(np.log(energies),
                               np.log(table[:, 0]),
                               np.log(table[:, column]))
        out += fraction * np.exp(log_interp)
    return out


def get_lin_att_c_vs_E(density, formula, energies):
    """Returns the linear attenuation coefficient curve in 1/mm.

    Args:
        density (float): Material density in g/cm^3.
        formula (str or dict): Chemical formula, e.g. 'Gd2O2S'.
        energies (numpy.ndarray): Energies in keV, within the NIST
            table range of 1 keV to 20 MeV.
    """
    energies = np.asarray(energies, dtype=float)
    return density * _mass_coefficient(formula, energies, 1) / 10.0


def get_lin_absp_c_vs_E(density, formula, energies):
    """Returns the linear energy-absorption coefficient curve in
    1/mm.

    Args:
        density (float): Material density in g/cm^3.
        formula (str or dict): Chemical formula.
        energies (numpy.ndarray): Energies in keV, within the NIST
            table range.
    """
    energies = np.asarray(energies, dtype=float)
    return density * _mass_coefficient(formula, energies, 2) / 10.0


# ---------------------------------------------------------------------------
# Measurement masking helpers (used with the measured ALS data)
# ---------------------------------------------------------------------------

def detect_inliers(sinogram, window_size, threshold_std=3):
    """Masks the inliers of an attenuation sinogram (Wenrui Li's
    method from xcal 1).  The sinogram is converted to
    transmission, exp(-sinogram), and each value is compared with
    the mean of its channel neighborhood.  A value is an inlier
    when its deviation is within threshold_std standard deviations
    of that row's deviations.

    Args:
        sinogram (numpy.ndarray): Attenuation sinogram with shape
            (views, rows, channels), that is, the negative log of
            the transmission.
        window_size (int): Width of the neighborhood.
        threshold_std (float): Deviation threshold in standard
            deviations.

    Returns:
        numpy.ndarray: Boolean array, True where a value is an
        inlier.
    """
    from scipy.ndimage import convolve1d
    data = np.exp(-np.asarray(sinogram))    # to transmission
    kernel = np.full(window_size, -1 / (window_size - 1))
    kernel[window_size // 2] = 1
    mask = np.zeros_like(data, dtype=bool)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            convolved = convolve1d(data[i, j, :], kernel,
                                   mode='constant', cval=1.0)
            threshold = np.std(convolved) * threshold_std
            mask[i, j, :] = np.abs(convolved) < threshold
    return mask


def only_center_mask(data, window_size=None):
    """Masks, per view and row, a window of channels centered
    on the darkest region, which is where the calibration target is
    (Wenrui Li's method from xcal 1).

    Args:
        data (numpy.ndarray): Transmission data, shape (views, rows,
            channels).
        window_size (int, optional): Number of channels to keep.
            Defaults to all channels.

    Returns:
        numpy.ndarray: Boolean array, True on the kept channels.
    """
    from scipy.ndimage import convolve1d
    new_mask = np.ones_like(data, dtype=bool)
    if window_size is None:
        window_size = data.shape[2]
    half_window = window_size // 2
    kernel = np.ones(window_size)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            window_sums = convolve1d(data[i, j, :], kernel,
                                     mode='constant', cval=np.inf)
            argmin = np.argmin(window_sums)
            start = max(argmin - half_window, 0)
            end = min(argmin + half_window + 1, data.shape[2])
            new_mask[i, j, :start] = False
            new_mask[i, j, end + 1:] = False
    return new_mask
