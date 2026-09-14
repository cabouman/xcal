.. _CatalogDocs:

=================
Materials Catalog
=================

The catalog lists the named materials you can select, with their
formulas and densities.  When you omit a material, xcal searches the
catalog's candidates for that component type:

* Filters: Al, Cu, Si.
* Targets (rods): V, Ti, Al, Mg.
* Scintillators: CsI, GAGG, LuAG, CdWO4, YAG, BGO, GOS.

You are not limited to these names.  Any chemical formula of
elements 1 through 92 works (NIST tables, 1 keV to 20 MeV); supply
the density if it is not in the catalog.  To add your own named
materials, put them in a YAML file with the same layout as the
shipped catalog and load it with :func:`~xcal.add_materials`.

.. autofunction:: xcal.list_materials

.. autofunction:: xcal.add_materials
