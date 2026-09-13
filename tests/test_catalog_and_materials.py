"""Catalog and material resolution tests."""
import os
import tempfile

import pytest

import xcal
from xcal import catalog, _materials


@pytest.fixture(autouse=True)
def fresh_catalog():
    catalog._reset()
    yield
    catalog._reset()


def test_shipped_catalog_contents():
    assert len(xcal.list_materials('filter')) == 3
    assert len(xcal.list_materials('target')) == 4
    assert len(xcal.list_materials('scintillator')) == 7
    names = {e['name'] for e in xcal.list_materials('scintillator')}
    assert 'CsI' in names and 'GOS' in names


def test_list_materials_rejects_unknown_kind():
    with pytest.raises(ValueError, match='kind'):
        xcal.list_materials('detector')


def test_add_materials_overrides_and_extends():
    with tempfile.NamedTemporaryFile('w', suffix='.yaml',
                                     delete=False) as f:
        f.write("filter_materials:\n"
                "  - {name: Sn, formula: Sn, density: 7.31,"
                " thickness_range: [0, 2]}\n"
                "  - {name: Al, formula: Al, density: 2.70,"
                " thickness_range: [0, 20]}\n")
        path = f.name
    try:
        xcal.add_materials(path)
        entries = {e['name']: e for e in xcal.list_materials('filter')}
        assert 'Sn' in entries
        assert entries['Al']['density'] == 2.70
    finally:
        os.unlink(path)


def test_add_materials_rejects_bad_entries():
    with tempfile.NamedTemporaryFile('w', suffix='.yaml',
                                     delete=False) as f:
        f.write("filter_materials:\n  - {name: Bad, formula: X}\n")
        path = f.name
    try:
        with pytest.raises(ValueError, match='density'):
            xcal.add_materials(path)
    finally:
        os.unlink(path)


def test_resolve_element_uses_builtin_density():
    m = _materials.resolve('Ti', 'target')
    assert m.formula == 'Ti'
    assert m.density == pytest.approx(4.507, rel=0.01)


def test_resolve_catalog_compound():
    m = _materials.resolve('GOS', 'scintillator')
    assert m.formula == 'Gd2O2S'
    assert m.density == pytest.approx(7.32)


def test_resolve_unknown_compound_requires_density():
    with pytest.raises(ValueError, match='density'):
        _materials.resolve('Gd2O2S', 'target')
    m = _materials.resolve('Gd2O2S', 'target', density=7.32)
    assert m.density == 7.32


def test_resolve_rejects_unknown_element():
    with pytest.raises(ValueError, match='element'):
        _materials.resolve('Xz2O', 'filter', density=1.0)


def test_resolve_rejects_transuranic():
    with pytest.raises(ValueError, match='NIST'):
        _materials.resolve('Pu', 'target', density=19.8)
