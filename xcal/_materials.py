"""Internal material resolution.

A user names a material either by a catalog name or by a chemical
formula.  This module turns that name into a Material with a formula
and a density, and raises plain-language errors when it cannot.
"""

from dataclasses import dataclass

import chemparse

from . import catalog
from .chem_consts._periodictabledata import density as _element_density
from .chem_consts._periodictabledata import atom_weights as _atom_weights

# Elements covered by the NIST tables shipped in mu_en.h5: hydrogen
# through uranium.  The atom weight table also carries transuranic
# elements that have no NIST entry, so membership is checked against
# this set, not against atom_weights.
_MAX_Z_SYMBOLS = None


def _nist_symbols():
    global _MAX_Z_SYMBOLS
    if _MAX_Z_SYMBOLS is None:
        import h5py
        from .chem_consts._consts_from_table import __file__ as _cc_file
        import os
        path = os.path.join(os.path.dirname(_cc_file), 'mu_en.h5')
        with h5py.File(path, 'r') as f:
            _MAX_Z_SYMBOLS = set(f.keys())
    return _MAX_Z_SYMBOLS


@dataclass(frozen=True)
class Material:
    """A resolved material: display name, chemical formula, density in
    g/cm^3."""
    name: str
    formula: str
    density: float


def parse_formula(formula):
    """Parse a chemical formula into an element -> count dict, raising
    a plain error for unknown or unsupported elements."""
    parsed = chemparse.parse_formula(formula)
    if not parsed:
        raise ValueError(f"'{formula}' is not a chemical formula.")
    for element in parsed:
        if element not in _atom_weights:
            raise ValueError(
                f"'{formula}' contains '{element}', which is not a known "
                f"element symbol.")
        if element not in _nist_symbols():
            raise ValueError(
                f"'{formula}' contains '{element}', which has no NIST "
                f"attenuation table; elements hydrogen (H) through "
                f"uranium (U) are supported.")
    return parsed


def resolve(spec, kind, density=None, context=''):
    """Resolve a material specification into a Material.

    Args:
        spec (str): A catalog name (for example 'GOS') or a chemical
            formula (for example 'Gd2O2S').
        kind (str): 'filter', 'rod', or 'scintillator'; selects the
            catalog section searched for a name match.
        density (float, optional): Density in g/cm^3, overriding the
            catalog or element value.
        context (str): Prefix for error messages, for example 'rod 2'.

    Returns:
        Material: The resolved material.
    """
    where = f"{context}: " if context else ""
    if not isinstance(spec, str):
        raise TypeError(
            f"{where}material must be a catalog name or a chemical "
            f"formula string, got {spec!r}.")

    entry = catalog._find(kind, spec)
    if entry is not None:
        parse_formula(entry['formula'])
        return Material(name=entry['name'], formula=entry['formula'],
                        density=float(density if density is not None
                                      else entry['density']))

    parsed = parse_formula(spec)
    if density is not None:
        return Material(name=spec, formula=spec, density=float(density))
    if len(parsed) == 1 and next(iter(parsed.values())) == 1:
        element = next(iter(parsed))
        return Material(name=spec, formula=spec,
                        density=float(_element_density[element]))
    raise ValueError(
        f"{where}'{spec}' is a compound that is not in the materials "
        f"catalog, so its density is unknown.  Pass density= in g/cm^3, "
        f"or add the material to a catalog file and load it with "
        f"xcal.add_materials().")
