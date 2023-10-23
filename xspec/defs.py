import numpy as np
from xspec._utils import is_sorted
from copy import deepcopy
from itertools import product

class Bound:
    def __init__(self, lower:float, upper:float):
        """

        Parameters
        ----------
        lower: float
            Lower bound
        upper: float
            Uppder bound
        """
        # Validate bounds before assignment
        self._validate_bounds(lower, upper)

        self.lower = lower
        self.upper = upper

    def _validate_bounds(self, lower, upper):
        """
        Validates the bounds to ensure lower is less than upper and both are positive.
        Throws ValueError if conditions are not met.
        """
        if lower < 0 or upper < 0:
            raise ValueError("Both lower and upper bounds must be non-negative.")

        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound.")

    def is_within_bound(self, value):
        """
        Check if the provided value is within the bounds.

        Parameters
        ----------
        value : float or int
            The value to check.

        Returns
        -------
        is_bounded: bool
            True if within bounds, False otherwise.
        """
        # Assuming the value is a float or an integer, you might want to add checks for the type of value.
        return self.lower <= value <= self.upper

    def update(self, lower=None, upper=None):
        """
        Update the values of the bounds.
        Validates new values and changes only if they are valid.

        Parameters
        ----------
        lower: float
        upper: float

        Returns
        -------

        """
        new_lower = lower if lower is not None else self.lower
        new_upper = upper if upper is not None else self.upper

        # Validate before making any changes
        self._validate_bounds(new_lower, new_upper)

        # Set the new values after validation
        self.lower = new_lower
        self.upper = new_upper

    def __repr__(self):
        """
        String representation of the Bound instance.
        """
        return f"Bound(lower={self.lower}, upper={self.upper})"

class Material:
    def __init__(self, formula, density):
        """

        Parameters
        ----------
        formula: str
            Chemical formula
        density: float
            Material density g/cm³
        """
        # Validate the inputs
        if not isinstance(formula, str):
            raise ValueError("Formula must be a string representing the chemical composition.")

        if not isinstance(density, (float, int)) or density <= 0:
            raise ValueError(
                "Density must be a positive number representing the density in some units (e.g., g/cm^3).")

        # If inputs are valid, assign them to the instance attributes
        self.formula = formula
        self.density = float(density)  # Ensure density is stored as a float, even if an int is passed

    def __repr__(self):
        return f"Material(formula='{self.formula}', density={self.density})"


class src_spec_params:
    def __init__(self, energies, src_vol_list, src_spec_list, src_vol_bound, voltage=None, require_gradient=True):
        """A data structure to store and check source spectrum parameters.

        Parameters
        ----------
        energies : numpy.ndarray
            1D numpy array of X-ray energies of a poly-energetic source in units of keV.
        src_vol_list : list
            A list of source voltage corresponding to src_spect_list.
        src_spec_list: list
            A list of source spectrum corresponding to src_vol_list.
        src_vol_bound: Bound
            Source voltage lower and uppder bound.
        voltage: float or int
            Source voltage. Default is None. Can be set for initial value.

        Returns
        -------

        """
        # Check if energy in energies is sorted from small to large
        if not is_sorted(energies):
            raise ValueError("Warning: energy bin in energies are not sorted!")
        else:
            self.energies = energies

        # Check if voltages in src_vol_list is sorted from small to large
        if not is_sorted(src_vol_list):
            raise ValueError("Warning: source voltage in src_response_dict are not sorted!")
        else:
            self.src_vol_list = src_vol_list

        self.src_spec_list = src_spec_list

        # Check if src_vol_bound is an instance of Bound
        if not isinstance(src_vol_bound, Bound):
            raise ValueError(
                "Expected an instance of Bound for src_vol_bound, but got {}.".format(type(src_vol_bound).__name__))
        else:
            self.src_vol_bound = src_vol_bound

        # Check voltage
        if voltage is None:
            voltage = 0.5 * (src_vol_bound.lower + src_vol_bound.upper)
        elif isinstance(voltage, float):
            # It's already a float, no action needed
            voltage = voltage
        elif isinstance(voltage, int):
            # It's an integer, convert to float
            voltage = float(voltage)
        else:
            # It's not a float or int, so raise an error
            raise ValueError(f"Expected 'voltage' to be a float or an integer, but got {type(voltage).__name__}.")

        if voltage <= 0:
            raise ValueError(f"Expected 'voltage' to be positive, but got {voltage}.")

        if not self.src_vol_bound.is_within_bound(voltage):
            raise ValueError(f"Expected 'voltage' to be inside src_vol_bound, but got {voltage}.")
        self.voltage = voltage
        self.require_gradient= require_gradient

class fltr_resp_params:
    def __init__(self, psb_fltr_mat, fltr_th_bound, fltr_th=None, require_gradient=True):
        """A data structure to store and check filter response parameters.

        Parameters
        ----------
        psb_fltr_mat: list
            List of possible filter material.
            fltr_mat is an instances of class Material, containing chemical formula and density.
        fltr_th_bound: Bound
            fltr_th_bound is an instance of class Bound.
        fltr_th: float
            filter thickness.

        Returns
        -------

        """

        # Check if psb_fltr_mat's elements are all instance of class Material.
        for mat in psb_fltr_mat:
            # Check mat is an instance of Material
            if not isinstance(mat, Material):  # The tolerance can be adjusted
                raise ValueError(
                    "Expected an instance of class Material for mat, but got {}.".format(type(mat).__name__))

            self.psb_fltr_mat = psb_fltr_mat

        self.fltr_mat = None

        # Check if fltr_th_bound is an instance of Bound
        if not isinstance(fltr_th_bound, Bound):
            raise ValueError(
                "Expected an instance of Bound for ftb, but got {}.".format(type(fltr_th_bound).__name__))
        self.fltr_th_bound = fltr_th_bound

        # Check fltr_th
        if fltr_th is None:
            fltr_th = 0.5 * (fltr_th_bound.lower + fltr_th_bound.upper)
        else:
            try:
                fltr_th = float(fltr_th)
            except ValueError:
                raise ValueError("All elements in 'fltr_th' must be convertible to float")

        # Check if fltr_th within fltr_th_bound
        if not fltr_th_bound.is_within_bound(fltr_th):
            raise ValueError(f"Expected 'ft' to be inside ftb, but got {fltr_th}.")
        self.fltr_th = fltr_th
        self.require_gradient = require_gradient

    def next_psb_fltr_mat(self):
        for fltr_mat in self.psb_fltr_mat:
            self.fltr_mat = fltr_mat
            yield deepcopy(self)

    def set_mat(self, fltr_mat: Material):
        # Check fltr_mat is an instance of Material
        if not isinstance(fltr_mat, Material):
            raise ValueError(
                "Expected an instance of class Material for fm, but got {}.".format(type(fltr_mat).__name__))
        self.fltr_mat = fltr_mat

class scint_cvt_func_params:
    def __init__(self, psb_scint_mat:[Material], scint_th_bound: Bound, scint_th=None, require_gradient=True):
        """A data structure to store and check scintillator response parameters.

        Parameters
        ----------
        psb_scint_mat: Material
            Scintillator material
        scint_th_bound: Bound
            Scintillator thickness bound
        scint_th: float
            Scintillator thickness. Default is None. Can be set for initial value.

        Returns
        -------

        """
        self.possible_scint_mat = psb_scint_mat
        self.scint_mat = None

        if not isinstance(scint_th_bound, Bound):
            raise ValueError(
                "Expected an instance of Bound for scint_th_bound, but got {}.".format(
                    type(scint_th_bound).__name__))
        if scint_th_bound.lower <= 0.001:
            raise ValueError(
                f"Expected lower bound of scint_th is greater than 0.001, but got {scint_th_bound.lower}.")
        self.scint_th_bound = scint_th_bound

        if scint_th is None:
            scint_th = 0.5 * (scint_th_bound.lower + scint_th_bound.upper)
        elif isinstance(scint_th, float):
            # It's already a float, no action needed
            scint_th = scint_th
        elif isinstance(scint_th, int):
            # It's an integer, convert to float
            scint_th = float(scint_th)
        else:
            # It's not a float or int, so raise an error
            raise ValueError(f"Expected 'voltage' to be a float or an integer, but got {type(scint_th).__name__}.")

        if not self.scint_th_bound.is_within_bound(scint_th):
            raise ValueError(f"Expected 'voltage' to be inside scint_th_bound, but got {scint_th}.")
        self.scint_th = scint_th
        self.require_gradient = require_gradient

    def next_psb_scint_mat(self):
        for mat in self.possible_scint_mat:
            self.scint_mat = mat
            yield deepcopy(self)

    def set_mat(self, mat:Material):
        # Check scint_mat is an instance of Material
        if not isinstance(mat, Material):  # The tolerance can be adjusted
            raise ValueError(
                "Expected an instance of class Material for mat, but got {}.".format(
                    type(mat).__name__))
        self.scint_mat = mat

class Model_combination:
    def __init__(self, src_ind=0, fltr_ind_list=0, scint_ind=0):
        """

        Parameters
        ----------
        src_ind: int
            Index of source model
        fltr_ind_list: int
            Index of filter model
        scint_ind: int
            Index of scintillator model

        Returns
        -------

        """
        self.src_ind=src_ind
        self.fltr_ind_list=fltr_ind_list
        self.scint_ind=scint_ind
