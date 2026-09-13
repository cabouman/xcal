"""The materials catalog.

The catalog lists the named materials a user can select from, with the
parameters the software needs: chemical formula, density, and a default
thickness range.  It ships with the materials used in the XCal paper and
defines the default candidate list for each component type.  Users
extend it with their own catalog files; the shipped file is never
modified.
"""

__all__ = ['list_materials', 'add_materials']


def list_materials(kind=None):
    """List the materials in the catalog.

    Args:
        kind (str, optional): 'filter', 'target', or 'scintillator' to
            list one component type; None lists all.

    Returns:
        list of dict: One entry per material with keys 'name',
        'formula', 'density', 'kind', and 'thickness_range'.
    """
    raise NotImplementedError("xcal 2 skeleton")


def add_materials(filename):
    """Add materials from a user catalog file.

    The file uses the same YAML layout as the shipped catalog.  Entries
    with new names are added; entries that reuse a shipped name override
    it for this session.

    Args:
        filename (str): Path to a YAML catalog file.
    """
    raise NotImplementedError("xcal 2 skeleton")
