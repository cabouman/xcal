"""The materials catalog.

The catalog lists the named materials a user can select from, with the
parameters the software needs: chemical formula, density, and a default
thickness range.  It ships with the materials used in the XCal paper and
defines the default candidate list for each component type.  Users
extend it with their own catalog files; the shipped file is never
modified.
"""

import copy
import os

import yaml

__all__ = ['list_materials', 'add_materials']

_KIND_KEYS = {
    'filter': 'filter_materials',
    'target': 'calibration_target_materials',
    'scintillator': 'scintillator_materials',
}

# The shipped catalog plus any user additions for this session.
_catalog = None


def _shipped_catalog_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'data', 'materials.yaml')


def _load():
    """Return the session catalog, loading the shipped file once."""
    global _catalog
    if _catalog is None:
        with open(_shipped_catalog_path()) as f:
            _catalog = yaml.safe_load(f)
        for key in _KIND_KEYS.values():
            _catalog.setdefault(key, [])
        _catalog.setdefault('defaults', {})
    return _catalog


def _reset():
    """Discard user additions and reload the shipped catalog.  Used by
    tests."""
    global _catalog
    _catalog = None


def list_materials(kind=None):
    """List the materials in the catalog.

    Args:
        kind (str, optional): 'filter', 'target', or 'scintillator' to
            list one component type; None lists all.

    Returns:
        list of dict: One entry per material with keys 'name',
        'formula', 'density', and 'kind', plus the default range keys
        from the catalog file.
    """
    cat = _load()
    if kind is not None and kind not in _KIND_KEYS:
        raise ValueError(f"kind must be one of {sorted(_KIND_KEYS)} or None, "
                         f"got {kind!r}.")
    kinds = [kind] if kind is not None else sorted(_KIND_KEYS)
    entries = []
    for k in kinds:
        for entry in cat[_KIND_KEYS[k]]:
            entry = copy.deepcopy(entry)
            entry['kind'] = k
            entries.append(entry)
    return entries


def add_materials(filename):
    """Add materials from a user catalog file.

    The file uses the same YAML layout as the shipped catalog.  Entries
    with new names are added; entries that reuse a shipped name override
    it for this session.

    Args:
        filename (str): Path to a YAML catalog file.
    """
    with open(filename) as f:
        added = yaml.safe_load(f)
    if not isinstance(added, dict):
        raise ValueError(f"{filename} does not contain a catalog mapping.")
    unknown = set(added) - set(_KIND_KEYS.values()) - {'defaults'}
    if unknown:
        raise ValueError(
            f"{filename} has unknown sections {sorted(unknown)}; valid "
            f"sections are {sorted(_KIND_KEYS.values()) + ['defaults']}.")
    cat = _load()
    for key in _KIND_KEYS.values():
        for entry in added.get(key, []):
            _validate_entry(entry, key, filename)
            existing = [e for e in cat[key] if e['name'] == entry['name']]
            for e in existing:
                cat[key].remove(e)
            cat[key].append(copy.deepcopy(entry))
    cat['defaults'].update(added.get('defaults', {}))


def _validate_entry(entry, section, filename):
    for field in ('name', 'formula', 'density'):
        if field not in entry:
            raise ValueError(
                f"{filename}: entry {entry!r} in {section} is missing the "
                f"required field '{field}'.")
    if not isinstance(entry['density'], (int, float)) or entry['density'] <= 0:
        raise ValueError(
            f"{filename}: material '{entry['name']}' has invalid density "
            f"{entry['density']!r}; density must be a positive number in "
            f"g/cm^3.")


def _find(kind, name):
    """Return the catalog entry of one kind with the given name, or
    None."""
    for entry in _load()[_KIND_KEYS[kind]]:
        if entry['name'] == name:
            return copy.deepcopy(entry)
    return None


def _default_candidates(kind):
    """Return the default candidate entries for one component type."""
    return list_materials(kind)


def _defaults():
    """Return the catalog defaults section."""
    return copy.deepcopy(_load()['defaults'])
