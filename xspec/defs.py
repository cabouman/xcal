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


class Source:
    def __init__(self, energies, src_voltage_list, takeoff_angle_cur, src_spec_list,
                 src_voltage_bound=Bound(10,300), takeoff_angle_bound=Bound(0,90), voltage=None,
                 takeoff_angle=None, optimize_voltage=True, optimize_takeoff_angle=True):
        """A data structure to store and check source spectrum parameters.

        Parameters
        ----------
        energies : list
            List of X-ray energies of a poly-energetic source in units of keV.
        src_voltage_list : list
            A list of source voltage corresponding to src_spect_list.
        src_spec_list: list
            A list of source spectrum corresponding to src_vol_list.
        src_voltage_bound: Bound
            Source voltage lower and uppder bound.
        voltage: float or int
            Source voltage. Default is None. Can be set for initial value.
        optimize_voltage : bool
            Specify if requiring optimization over source voltage.

        Returns
        -------

        """
        # Check if energy in energies is sorted from small to large
        if not is_sorted(energies):
            raise ValueError("Warning: energy bin in energies are not sorted!")
        else:
            self.energies = energies

        # Check if voltages in src_vol_list is sorted from small to large
        if not is_sorted(src_voltage_list):
            raise ValueError("Warning: source voltage in src_response_dict are not sorted!")
        else:
            self.src_voltage_list = src_voltage_list

        self.src_spec_list = src_spec_list
        self.takeoff_angle_cur = takeoff_angle_cur

        # Check if src_vol_bound is an instance of Bound
        if not isinstance(src_voltage_bound, Bound):
            raise ValueError(
                "Expected an instance of Bound for src_vol_bound, but got {}.".format(type(src_voltage_bound).__name__))
        else:
            self.src_voltage_bound = src_voltage_bound

        if not isinstance(takeoff_angle_bound, Bound):
            raise ValueError(
                "Expected an instance of Bound for takeOffAngle_bound, but got {}.".format(type(takeoff_angle_bound).__name__))
        else:
            self.takeoff_angle_bound = takeoff_angle_bound


        # Check voltage
        if voltage is None:
            voltage = 0.5 * (src_voltage_bound.lower + src_voltage_bound.upper)
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

        if not self.src_voltage_bound.is_within_bound(voltage):
            raise ValueError(f"Expected 'voltage' to be inside src_vol_bound, but got {voltage}.")
        self.voltage = voltage
        self.optimize_voltage= optimize_voltage

        # Check takeoff_angle
        if takeoff_angle is None:
            takeoff_angle = 0.5 * (takeoff_angle_bound.lower + takeoff_angle_bound.upper)
        elif isinstance(takeoff_angle, float):
            # It's already a float, no action needed
            takeoff_angle = takeoff_angle
        elif isinstance(takeoff_angle, int):
            # It's an integer, convert to float
            takeoff_angle = float(takeoff_angle)
        else:
            # It's not a float or int, so raise an error
            raise ValueError(f"Expected 'takeoff_angle' to be a float or an integer, but got {type(takeoff_angle).__name__}.")

        if not self.takeoff_angle_bound.is_within_bound(takeoff_angle):
            raise ValueError(f"Expected 'takeoff_angle' to be inside takeoff_angle_bound, but got {takeoff_angle}.")
        self.takeoff_angle = takeoff_angle
        self.optimize_takeoff_angle = optimize_takeoff_angle

class Filter:
    def __init__(self, possible_mat=[], fltr_th_bound=Bound(0,1000), fltr_mat=None, fltr_th=None, optimize=True):
        """A data structure to store and check filter response parameters.

        Parameters
        ----------
        possible_mat: list
            List of possible filter material.
        fltr_th_bound: Bound
            fltr_th_bound is an instance of class Bound.
        fltr_mat : Material
            fltr_mat is an instances of class Material, containing chemical formula and density.
        fltr_th: float
            filter thickness.
        optimize : bool
            Specify if requiring optimization over filter thickness.

        Returns
        -------

        """

        # Check if psb_fltr_mat's elements are all instance of class Material.
        for mat in possible_mat:
            # Check mat is an instance of Material
            if not isinstance(mat, Material):  # The tolerance can be adjusted
                raise ValueError(
                    "Expected an instance of class Material for mat, but got {}.".format(type(mat).__name__))

            self.possible_mat = possible_mat

        self.fltr_mat = fltr_mat

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
        self.optimize = optimize

    def next_psb_fltr_mat(self):
        for fltr_mat in self.possible_mat:
            self.fltr_mat = fltr_mat
            yield deepcopy(self)

    def set_mat(self, fltr_mat: Material):
        # Check fltr_mat is an instance of Material
        if not isinstance(fltr_mat, Material):
            raise ValueError(
                "Expected an instance of class Material for fm, but got {}.".format(type(fltr_mat).__name__))
        self.fltr_mat = fltr_mat

class Scintillator:
    def __init__(self, possible_mat:[Material]=[], scint_th_bound=Bound(0.001, 2), scint_mat=None, scint_th=None, optimize=True):
        """A data structure to store and check scintillator response parameters.

        Parameters
        ----------
        possible_mat: list of Material
            Possible list of scintillator materials
        scint_th_bound: Bound
            Scintillator thickness bound
        scint_mat : Material
            Scintillator Material, an instance of class Material. Default None.
        scint_th: float
            Scintillator thickness. Default is None. Can be set for initial value. Default None.
        optimize : bool
            Specify if requiring optimization over scintillator thickness.

        Returns
        -------

        """
        self.possible_mat = possible_mat
        self.scint_mat = scint_mat

        if not isinstance(scint_th_bound, Bound):
            raise ValueError(
                "Expected an instance of Bound for scint_th_bound, but got {}.".format(
                    type(scint_th_bound).__name__))
        if scint_th_bound.lower < 0.001:
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
        self.optimize = optimize

    def next_psb_scint_mat(self):
        for mat in self.possible_mat:
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
        """Specify combination of source, filter, and scintillator models with indexes.

        Parameters
        ----------
        src_ind: int
            Index of source model
        fltr_ind_list: int
            List of indexes of filter models, specifing the combination of filters in one dataset.
        scint_ind: int
            Index of scintillator model

        Returns
        -------

        """
        self.src_ind=src_ind
        self.fltr_ind_list=fltr_ind_list
        self.scint_ind=scint_ind

def dict_to_sources(source_params, energies):
    num_voltage = source_params.get('num_voltage')
    reference_voltages = source_params.get('reference_voltages')
    reference_anode_angle = source_params.get('reference_anode_angle')
    reference_spectra = source_params.get('reference_spectra')
    anode_angle = source_params.get('anode_angle', None)
    anode_angle_range = source_params.get('anode_angle_range', None)
    optimize_voltage = source_params.get('optimize_voltage', True)
    optimize_anode_angle = source_params.get('optimize_anode_angle', True)

    # Create Bound object for anode angle range
    takeoff_angle_bound = Bound(anode_angle_range[0],anode_angle_range[1]) if anode_angle_range else Bound(0, 90)

    # List to hold created Source objects
    sources = []

    for i in range(1, num_voltage + 1):
        voltage_key = f'voltage_{i}'
        voltage_range_key = f'voltage_{i}_range'

        voltage = source_params.get(voltage_key, None)
        voltage_range = source_params.get(voltage_range_key, None)

        # Create Bound object for voltage range
        src_voltage_bound = Bound(voltage_range[0], voltage_range[1]) if voltage_range else Bound(30.0, 200.0)

        # Create a Source object and append to the list
        src = Source(energies=energies,
                     src_voltage_list=reference_voltages,
                     takeoff_angle_cur=reference_anode_angle,
                     src_spec_list=reference_spectra,
                     src_voltage_bound=src_voltage_bound,
                     takeoff_angle_bound=takeoff_angle_bound,
                     voltage=voltage,
                     takeoff_angle=anode_angle,
                     optimize_voltage=optimize_voltage,
                     optimize_takeoff_angle=optimize_anode_angle)

        sources.append(src)

    return sources




def dict_to_filters(filter_params):
    num_filters = filter_params.get('num_filter')
    possible_materials = filter_params.get('possible_material')
    optimize = filter_params.get('optimize',True)

    # List to hold created Filter objects
    filters = []

    for i in range(1, num_filters + 1):
        material_key = f'material_{i}'
        thickness_key = f'thickness_{i}'
        thickness_range_key = f'thickness_{i}_range'

        # Extracting the material, thickness, and thickness range for each filter
        material = filter_params.get(material_key, None)
        thickness = filter_params.get(thickness_key, None)
        thickness_range = filter_params.get(thickness_range_key, None)

        # Create a Bound object for thickness range, if available
        fltr_th_bound = Bound(thickness_range[0],thickness_range[1]) if thickness_range else Bound(0.0,10.0)

        # Create a Filter object and append to the list
        fltr = Filter(possible_mat=possible_materials,
                      fltr_th_bound=fltr_th_bound,
                      fltr_mat=material,
                      fltr_th=thickness,
                      optimize=optimize)  # Assuming optimization is required
        filters.append(fltr)

    return filters


def dict_to_scintillator(scintillator_params):
    possible_materials = scintillator_params.get('possible_material')
    material = scintillator_params.get('material', None)
    thickness = scintillator_params.get('thickness', None)
    thickness_range = scintillator_params.get('thickness_range', None)
    optimize = scintillator_params.get('optimize', True)

    # Create a Bound object for thickness range
    scint_th_bound = Bound(thickness_range[0], thickness_range[1]) if thickness_range else Bound(0.01,0.5)

    # Create a Scintillator object
    scintillator = Scintillator(possible_mat=possible_materials,
                                scint_th_bound=scint_th_bound,
                                scint_mat=material,
                                scint_th=thickness,
                                optimize=optimize)

    return [scintillator]
