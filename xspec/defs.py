import numpy as np
from xspec._utils import is_sorted
from copy import deepcopy
from itertools import product

class Bound:
    def __init__(self, lower:float, upper:float):
        """ A class to store continuous parameters range.

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
        """ A class to store chemical formula and density.

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

