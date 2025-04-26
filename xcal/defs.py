
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

