import numpy as np
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from xspec.chem_consts._consts_from_table import get_mass_absp_c_vs_E
from xspec.chem_consts._periodictabledata import atom_weights, ptableinverse
from xspec.dict_gen import gen_fltr_res, gen_scint_cvt_func


def linear_interp(x, xp, fp):
    """
    Performs linear interpolation.

    Args:
        x (torch.Tensor): The x-coordinates at which to evaluate the interpolated values.
        xp (torch.Tensor): The x-coordinates of the data points.
        fp (torch.Tensor): The y-coordinates of the data points (same shape as xp).

    Returns:
        torch.Tensor: The interpolated values.
    """
    # Find the indices of the rightmost value less than or equal to x
    idx = torch.searchsorted(xp, x) - 1
    idx = idx.clamp(0, len(xp) - 2)  # Clamp values to range to avoid out of bounds

    # Compute the slope of the segments
    slope = (fp[idx + 1] - fp[idx]) / (xp[idx + 1] - xp[idx])

    # Evaluate the line segment at x
    return fp[idx] + slope * (x - xp[idx])

class Interp2D:
    def __init__(self, x, y, z):
        """
        Initialize the Interp2D class for performing bilinear interpolation on a 2D grid.

        Args:
            x (torch.Tensor): A 2-D tensor representing the x-coordinates of the grid points,
                              with shape (M, N), where M is the number of rows and N is the number
                              of columns. Assumes uniform x-coordinates across each row.
            y (torch.Tensor): A 2-D tensor representing the y-coordinates of the grid points,
                              with shape (M, N), where M is the number of rows and N is the number
                              of columns. Assumes uniform y-coordinates across each column.
            z (torch.Tensor): An N-D tensor of z-values corresponding to the grid points
                              defined by x and y coordinates, with the first two dimensions
                              matching the shape of x and y (M, N), and any additional dimensions
                              representing different variables or measurements at each grid point.
        """
        self.x = x
        self.y = y
        self.z = z

    def __call__(self, new_x, new_y):
        """
        Perform bilinear interpolation to find z-values at a single new (x, y) coordinate.

        Args:
            new_x (torch.Tensor): A scalar tensor representing the new x-coordinate where
                                  the z-value is to be interpolated.
            new_y (torch.Tensor): A scalar tensor representing the new y-coordinate where
                                  the z-value is to be interpolated.

        Returns:
            torch.Tensor: A tensor of the interpolated z-value at the specified new_x and new_y
                          coordinate. If z is an N-D tensor, the returned tensor will maintain
                          the additional dimensions of z beyond the first two.

        Raises:
            ValueError: If the new_x or new_y values are outside the range of the original
                        x or y grid coordinates.
        """
        if not (self.x.min() <= new_x <= self.x.max()) or not (self.y.min() <= new_y <= self.y.max()):
            raise ValueError("The new_x or new_y values are outside the range of x or y.")

        # Find indices for the closest points in x and y
        x_indices = torch.searchsorted(self.x[:, 0], new_x) - 1
        y_indices = torch.searchsorted(self.y[0, :], new_y) - 1

        # Ensure indices are within the bounds of the x and y arrays
        x_indices = torch.clamp(x_indices, 0, self.x.size(1) - 2)
        y_indices = torch.clamp(y_indices, 0, self.y.size(0) - 2)

        # Calculate the four corner points for bilinear interpolation
        x0 = self.x[x_indices, 0]
        x1 = self.x[x_indices + 1, 0]
        y0 = self.y[0, y_indices]
        y1 = self.y[0, y_indices + 1]

        # Extract the z-values at the corner points
        z00 = self.z[x_indices, y_indices]
        z01 = self.z[x_indices, y_indices + 1]
        z10 = self.z[x_indices + 1, y_indices]
        z11 = self.z[x_indices + 1, y_indices + 1]

        # Compute the weights for bilinear interpolation
        w00 = (x1 - new_x) * (y1 - new_y)
        w01 = (x1 - new_x) * (new_y - y0)
        w10 = (new_x - x0) * (y1 - new_y)
        w11 = (new_x - x0) * (new_y - y0)

        # Perform bilinear interpolation
        interpolated_z = (w00 * z00 + w01 * z01 + w10 * z10 + w11 * z11) / ((x1 - x0) * (y1 - y0))
        interpolated_z = torch.clamp(interpolated_z, min=0)  # Ensure non-negativity
        return interpolated_z


class Interp1D:
    def __init__(self, x, y):
        """
        Initialize the Interp1d class for performing linear interpolation.

        Args:
            x (torch.Tensor): A 1-D tensor of x-coordinates, representing the positions
                              at which the y-values are known.
            y (torch.Tensor): An N-D tensor of y-coordinates corresponding to x. The first
                              dimension of y must match the length of x, and any additional
                              dimensions represent different sets of values to interpolate.
        """
        self.x = x
        self.y = y

    def __call__(self, new_x):
        """
        Perform linear interpolation to find y-values at new_x, a scalar or 1-D tensor
        of new x-coordinates.

        Args:
            new_x (torch.Tensor): A scalar or 1-D tensor of new x-coordinates where y-values
                                  are to be interpolated. This allows for interpolation at
                                  multiple points in a single call.

        Returns:
            torch.Tensor: An N-D tensor of interpolated y-values at new_x. The shape of the
                          output tensor maintains the additional dimensions of the input y,
                          with the first dimension size equal to the number of elements in new_x.

        Raises:
            ValueError: If any of the new_x values are outside the range of the original x
                        coordinates, indicating that interpolation cannot be performed at
                        those points.
        """
        # Handle the special case of a single data point
        if len(self.x) == 1:
            # Return y directly since there's no interpolation needed
            return self.y.repeat(len(new_x), *([1] * (self.y.dim() - 1)))

        if not torch.all(torch.logical_and(new_x >= self.x.min(), new_x <= self.x.max())):
            raise ValueError("Some values in new_x are outside the range of x.")

        # Find the indices of the nearest x-values in the original data
        indices = torch.searchsorted(self.x, new_x)
        indices = torch.clamp(indices, 1, len(self.x) - 1)  # Ensure indices are within range

        # Calculate the weights for interpolation
        x0 = self.x[indices - 1]
        x1 = self.x[indices]
        y0 = self.y[indices - 1]
        y1 = self.y[indices]
        alpha = (new_x - x0) / (x1 - x0)

        # Perform linear interpolation
        interpolated_y = y0 + alpha * (y1 - y0)
        interpolated_y = torch.clamp(interpolated_y, min=0)  # Ensure non-negativity
        return interpolated_y

class ClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        """
        """
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None


def clamp_with_grad(input, min, max):
    return ClampFunction.apply(input, min, max)

def normalize_tuple_as_parameter(tuple_value):
    """
    Normalize a tuple and represent it as a (normalized Pytorch Parameter, lower bound, and upper boun).

    Args:
        tuple_value (tuple): A tuple containing the initial value, lower bound, and upper bound.

    Returns:
        Parameter: A Parameter scalar representing the normalized value of the tuple.
    """
    # Unpack the tuple
    initial_value, lower_bound, upper_bound = tuple_value
    if initial_value is None:
        raise ValueError("initial_value tuple_value[0] cannot be None.")
    if lower_bound is None:
        lower_bound = initial_value
    if upper_bound is None:
        upper_bound = initial_value
    if lower_bound == upper_bound:
        return (torch.tensor(1, dtype=torch.float32), lower_bound, upper_bound)
    # Normalize the initial value
    normalized_value = (initial_value - lower_bound) / (upper_bound - lower_bound)

    # Create a Parameter scalar
    parameter_scalar = Parameter(torch.tensor(normalized_value, dtype=torch.float32))

    return (parameter_scalar, lower_bound, upper_bound)


def denormalize_parameter_as_tuple(tuple_value):
    """
    Denormalize a normalized PyTorch Parameter scalar and represent it as a tuple.

    Args:
        tuple_value (tuple): A tuple containing the normalized Pytorch Parameter, lower bound, and upper bound.

    Returns:
        tuple: A tuple containing the denormalized value, lower bound, and upper bound.
    """
    # Unpack the tuple
    parameter_scalar, lower_bound, upper_bound = tuple_value
    if lower_bound is None or upper_bound is None:
        raise ValueError("lower_bound tuple_value[1] and upper_bound tuple_value[2] cannot be None.")
    if lower_bound == upper_bound:
        return (parameter_scalar*lower_bound, lower_bound, upper_bound)
    # Get the normalized value from the Parameter scalar
    normalized_value = parameter_scalar

    # Denormalize the value
    denormalized_value = normalized_value * (upper_bound - lower_bound) + lower_bound

    return (denormalized_value, lower_bound, upper_bound)

def merge_dicts(list1, list2):
    merged_list = []
    for dict1 in list1:
        for dict2 in list2:
            common_keys = set(dict1.keys()) & set(dict2.keys())
            if all(dict1[key] == dict2[key] for key in common_keys):
                merged_dict = {**dict1, **dict2}
                merged_list.append(merged_dict)
    return merged_list
def get_merged_params_list(lists):
    merged_params_list =[]
    for params_list in lists:
        if len(merged_params_list) == 0:
            merged_params_list = params_list
        else:
            merged_params_list = merge_dicts(merged_params_list, params_list)
    return merged_params_list

def pair_params_list(list1, list2):
    """
    Pair parameters lists with other instance of Base_Spec_Model or its child model.

    Args:
        other (Base_Spec_Model): Another instance of Base_Spec_Model.

    Returns:
        list: List of paired parameters lists.
    """
    paired_params_list = []
    for params1 in list1:
        for params2 in list2:
            # Merge dictionaries and add instance names as prefixes to keys
            merged_params_dict = {}
            for key, value in list(params1.items())+list(params2.items()):
                if isinstance(value, tuple):
                    merged_params_dict[key] = value
                else:
                    merged_params_dict[key] = value
            paired_params_list.append(merged_params_dict)
    return paired_params_list

def get_concatenated_params_list(lists):
    merged_params_list =[]
    for params_list in lists:
        if len(merged_params_list) == 0:
            merged_params_list = params_list
        else:
            merged_params_list = pair_params_list(merged_params_list, params_list)
    return merged_params_list

class Base_Spec_Model(Module):

    def __init__(self, params_list=[]):
        """Base class for all spectral components in xspec.

        Args:
            params_list (list): List of dictionaries containing possible discrete and continuous parameters combinations.
                All dictionaries share the same keywords. Each dictionary contains both discrete and continuous parameters.
                Continuous parameters should be specified as tuples with the format (initial value, lower bound, upper bound),
                while discrete parameters can be directly specified.
        """
        super().__init__()

        if not hasattr(self.__class__, '_count'):
            self.__class__._count = 0
        self.__class__._count += 1
        self.prefix = f"{self.__class__.__name__}_{self.__class__._count}"

        # params_list contains all possible discrete parameters combinations and related continuous parameters.
        self._params_list = []
        for params in params_list:
            new_params = {}
            for key, value in params.items():
                if self.__class__.__name__ != 'Base_Spec_Model':
                    modified_key = f"{self.prefix}_{key}"
                else:
                    modified_key =f"{key}"
                if isinstance(value, tuple):
                    new_params[f"{modified_key}"] = normalize_tuple_as_parameter(value)
                else:
                    new_params[f"{modified_key}"] = value
            self._params_list.append(new_params)
        self._init_estimates()

    def set_spectrum(self, energies, sp):
        """

        Args:
            energies (numpy.array): A numpy array containing the X-ray energies of a poly-energetic source in units of keV.
            sp (numpy.array): Spectrum.

        Returns:

        """
        self.ref_sp_energies = torch.tensor(energies)
        self.ref_sp = torch.tensor(sp)

    def forward(self, energies):
        """
        Placeholder forward method.

        Args:
            energies (numpy.array): A numpy array containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: Output response.
        """
        energies = torch.tensor(energies)
        # Check if ref_sp_energies and ref_sp attributes are set
        if hasattr(self, 'ref_sp_energies') and hasattr(self, 'ref_sp'):
            return linear_interp(energies, self.ref_sp_energies, self.ref_sp)
        else:
            # Handle the case where ref_sp is not set, e.g., return a placeholder or raise an error
            print("ref_sp_energies or ref_sp is not set.")
            return torch.ones(len(energies))  # or any other appropriate default action


    def _init_estimates(self):
        """
        Initialize estimates from the first dictionary in params_list.
        """
        self.estimates = {}
        if len(self._params_list) == 0:
            return
        for key, value in self._params_list[0].items():
            self.estimates[key] = value
            if isinstance(value, tuple):
                setattr(self, key, self.estimates[key][0])
            else:
                setattr(self, key, value)

    def set_params(self, params):
        """
        Set estimates from a dictionary of parameters.

        Args:
            params (dict): Dictionary containing parameters.
        """
        for key, value in params.items():
            if key in self.estimates.keys():
                if isinstance(value, tuple):
                    if not isinstance(value[0], torch.Tensor):
                        normalized_value = normalize_tuple_as_parameter(value)
                        setattr(self, key, normalized_value[0])
                        self.estimates[key] = normalized_value
                    else:
                        setattr(self, key, value[0])
                        self.estimates[key] = value
                else:
                    setattr(self, key, value)
                    self.estimates[key] = value

    def get_params(self):
        """
        Read estimated parameters as a dictionary.

        Returns:
            dict: Dictionary containing estimated parameters.
        """
        display_estimates = {}
        for key, value in self.estimates.items():
            if isinstance(value, tuple):
                dv = denormalize_parameter_as_tuple(value)
                display_estimates[key] =  clamp_with_grad(dv[0], dv[1], dv[2])
            else:
                display_estimates[key] = value
        return display_estimates

def first_nonzero_from_right(arr):
    """
    Finds the index of the first non-zero element from right to left in a 1D NumPy array.

    Args:
        arr (numpy.ndarray): A 1D NumPy array.

    Returns:
        int: The index of the first non-zero element from the right. 
             If no non-zero element is found, returns -1.
    """
    # Reverse the array and use np.argmax to find the first non-zero element from the right
    rev_index = np.argmax(arr[::-1] != 0)
    
    # Check if all elements are zero
    if arr[::-1][rev_index] == 0:
        return -1
    else:
        # Convert the reverse index to the original index
        return len(arr) - 1 - rev_index
    
    
def prepare_for_interpolation(src_spec_list, kV_index=None):
    """
    Prepare the source spectral list for interpolation over voltage.

    Args:
        src_spec_list (list): List of source spectral responses for each voltage.
        kV_index (list): List of simulated voltages.

    Returns:
        list: Modified source spectral list ready for interpolation.
    """
    modified_src_spec_list = src_spec_list.copy()
    kV_index = [first_nonzero_from_right(modified_src_spec_list[i]) for i in range(len(modified_src_spec_list))]
    for sid, m_src_spec in enumerate(modified_src_spec_list[:-1]):
        v0 = kV_index[sid]
        v1 = kV_index[sid+1]
        f1 = modified_src_spec_list[sid+1]
        for v in range(v0, v1):
            if v == kV_index[sid+1]:
                m_src_spec[v] = 0
            else:
                r = (v - float(v0)) / (v1 - float(v0))
                m_src_spec[v] = -r / (1 - r) * f1[v]
    return modified_src_spec_list

def philibert_absorption_correction_factor(voltage, sin_psi, energies):
    Z = 74  # Tungsten
    target_material = ptableinverse[Z]
    PhilibertConstant = 4.0e5
    PhilibertExponent = 1.65
    # sin_psi = torch.sin(takeOffAngle * torch.pi / 180.0)
    h_local = 1.2 * atom_weights[target_material] / (Z ** 2)
    h_factor = h_local / (1.0 + h_local)

    kVp_e165 = voltage ** PhilibertExponent
    kappa = torch.zeros((energies.shape))
    if not isinstance(energies, torch.Tensor):
        energies = torch.tensor(energies)
    kappa[:-1] = (PhilibertConstant / (kVp_e165 - energies ** PhilibertExponent)[:-1])
    kappa[-1] = np.inf
    mu = torch.tensor(get_mass_absp_c_vs_E(ptableinverse[Z], energies))  # cm^-1

    return (1 + mu / kappa / sin_psi) ** -1 * (1 + h_factor * mu / kappa / sin_psi) ** -1


def takeoff_angle_conversion_factor(voltage, sin_psi_cur, sin_psi_new, energies):
    # Assuming takeOffAngle_cur is already defined
    if not isinstance(sin_psi_cur, torch.Tensor):
        sin_psi_cur = torch.tensor(sin_psi_cur)
    if not isinstance(sin_psi_new, torch.Tensor):
        sin_psi_new = torch.tensor(sin_psi_new)
    return philibert_absorption_correction_factor(voltage, sin_psi_new,
                                                  energies) / philibert_absorption_correction_factor(voltage,
                                                                                                     sin_psi_cur,
                                                                                                     energies)


def angle_sin(psi, torch_mode=False):
    if torch_mode:
        return torch.sin(psi * torch.pi / 180.0)
    else:
        return np.sin(psi * np.pi / 180.0)


class Reflection_Source(Base_Spec_Model):
    def __init__(self, voltage, takeoff_angle, single_takeoff_angle=True):
        """
        A template source model designed specifically for reflection sources, including all necessary methods.

        Args:
            voltage (tuple): (initial value, lower bound, upper bound) for the source voltage.
                These three values cannot be all None. It will not be optimized when lower == upper.
            takeoff_angle (tuple): (initial value, lower bound, upper bound) for the takeoff angle, in degrees.
                These three values cannot be all None. It will not be optimized when lower == upper.
            single_takeoff_angle (bool, optional): Determines whether the takeoff angle is same for all instances.
                If set to True (default), the same takeoff angle is applied to all instances of Reflection_Source.
                If set to False, each instance may have a distinct takeoff angle, with different prefix.
        """
        params_list = [{'voltage': voltage, 'takeoff_angle': takeoff_angle}]
        super().__init__(params_list)
        self.single_takeoff_angle = single_takeoff_angle
        if self.single_takeoff_angle:
            for params in self._params_list:
                params[f"{self.__class__.__name__}_takeoff_angle"] = params.pop(f"{self.prefix}_takeoff_angle")
            self._init_estimates()

    def set_src_spec_list(self, src_spec_list, src_voltage_list, ref_takeoff_angle):
        """Set source spectra for interpolation, which will be used only by forward function.

        Args:
            src_spec_list (numpy.ndarray): This array contains the reference X-ray source spectra. Each spectrum in this array corresponds to a specific combination of the ref_takeoff_angle and one of the source voltages from src_voltage_list.
            src_voltage_list (numpy.ndarray): This is a sorted array containing the source voltages, each corresponding to a specific reference X-ray source spectrum.
            ref_takeoff_angle (float): This value represents the anode take-off angle, expressed in degrees, which is used in generating the reference X-ray spectra.
        """
        self.src_spec_list = np.array(src_spec_list)
        self.src_voltage_list = np.array(src_voltage_list)
        modified_src_spec_list = prepare_for_interpolation(self.src_spec_list)
        self.src_spec_interp_func_over_v = Interp1D(torch.tensor(self.src_voltage_list, dtype=torch.float32),
                                                    torch.tensor(modified_src_spec_list, dtype=torch.float32))

        self.ref_takeoff_angle = ref_takeoff_angle

    def forward(self, energies):
        """
        Takes X-ray energies and returns the source spectrum.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The source response.
        """

        voltage = self.get_params()[f"{self.prefix}_voltage"]
        src_spec = self.src_spec_interp_func_over_v(voltage)

        if self.single_takeoff_angle:
            takeoff_angle = self.get_params()[f"{self.__class__.__name__}_takeoff_angle"]
        else:
            takeoff_angle = self.get_params()[f"{self.prefix}_takeoff_angle"]
        # print('ID takeoff_angle:', id(takeoff_angle))
        sin_psi_cur = angle_sin(self.ref_takeoff_angle, torch_mode=False)
        sin_psi_new = angle_sin(takeoff_angle, torch_mode=True)
        src_spec = src_spec * takeoff_angle_conversion_factor(voltage, sin_psi_cur, sin_psi_new, energies)

        return src_spec

class Transmission_Source(Base_Spec_Model):
    def __init__(self, voltage, target_thickness, single_target_thickness):
        """
        A template source model designed specifically for reflection sources, including all necessary methods.

        Args:
            voltage (tuple): (initial value, lower bound, upper bound) for the source voltage.
                These three values cannot be all None. It will not be optimized when lower == upper.

        """
        params_list = [{'voltage': voltage, 'target_thickness': target_thickness}]
        super().__init__(params_list)
        self.single_target_thickness = single_target_thickness
        if self.single_target_thickness:
            for params in self._params_list:
                params[f"{self.__class__.__name__}_target_thickness"] = params.pop(f"{self.prefix}_target_thickness")
            self._init_estimates()

    def set_src_spec_list(self, src_spec_list, voltages, target_thicknesses):
        """Set source spectra for interpolation, which will be used only by forward function.

        Args:
            src_spec_list (numpy.ndarray): This array contains the reference X-ray source spectra. Each spectrum in this array corresponds to a specific combination of the ref_takeoff_angle and one of the source voltages from src_voltage_list.
            src_voltage_list (numpy.ndarray): This is a sorted array containing the source voltages, each corresponding to a specific reference X-ray source spectrum.
            ref_takeoff_angle (float): This value represents the anode take-off angle, expressed in degrees, which is used in generating the reference X-ray spectra.
        """
        self.src_spec_list = np.array(src_spec_list)
        self.voltages = np.array(voltages)
        self.target_thicknesses = np.array(target_thicknesses)
        modified_src_spec_list = src_spec_list.copy()
        for tti, tt in enumerate(target_thicknesses):
            modified_src_spec_list[:, tti] = prepare_for_interpolation(modified_src_spec_list[:, tti])

        # Generate 2D grids for x and y coordinates
        V, T = torch.meshgrid(torch.tensor(self.voltages, dtype=torch.float32), torch.tensor(self.target_thicknesses, dtype=torch.float32), indexing='ij')
        self.src_spec_interp_func = Interp2D(V, T, torch.tensor(modified_src_spec_list, dtype=torch.float32))

    def forward(self, energies):
        """
        Takes X-ray energies and returns the source spectrum.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The source response.
        """

        voltage = self.get_params()[f"{self.prefix}_voltage"]
        if self.single_target_thickness:
            target_thickness = self.get_params()[f"{self.__class__.__name__}_target_thickness"]
        else:
            target_thickness = self.get_params()[f"{self.prefix}_target_thickness"]
        src_spec = self.src_spec_interp_func(voltage, target_thickness)

        return src_spec

class Filter(Base_Spec_Model):
    def __init__(self, materials, thickness):
        """
        A template filter model based on Beer's Law and NIST mass attenuation coefficients, including all necessary methods.

        Args:
            materials (list): A list of possible materials for the filter,
                where each material should be an instance containing formula and density.
            thickness (tuple or list): If a tuple, it should be (initial value, lower bound, upper bound) for the filter thickness.
                If a list, it should have the same length as the materials list, specifying thickness for each material.
                These values cannot be all None. It will not be optimized when lower == upper.
        """
        if isinstance(thickness, tuple):
            if all(t is None for t in thickness):
                raise ValueError("Thickness tuple cannot have all None values.")
            params_list = [{'material': mat, 'thickness': thickness} for mat in materials]
        elif isinstance(thickness, list):
            if len(thickness) != len(materials):
                raise ValueError("Length of thickness list must match length of materials list.")
            params_list = [{'material': mat, 'thickness': th} for mat, th in zip(materials, thickness)]
        else:
            raise TypeError("Thickness must be either a tuple or a list.")

        super().__init__(params_list)

    def forward(self, energies):
        """
        Takes X-ray energies and returns the filter response.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The filter response as a function of input energies, selected material, and its thickness.
        """
        mat = self.get_params()[f"{self.prefix}_material"]
        th = self.get_params()[f"{self.prefix}_thickness"]
        # print('ID filter th:', id(th))
        return gen_fltr_res(energies, mat, th)


class Scintillator(Base_Spec_Model):
    def __init__(self, thickness, materials, device=None, dtype=None):
        """
        A template scintillator model based on Beer's Law, NIST mass attenuation coefficients, and mass energy-absorption coefficients, including all necessary methods.

        Args:
            materials (list): A list of possible materials for the scintillator,
                where each material should be an instance containing formula and density.
            thickness (tuple): (initial value, lower bound, upper bound) for the scintillator thickness.
                These three values cannot be all None. It will not be optimized when lower == upper.
        """
        params_list = [{'material': mat, 'thickness': thickness} for mat in materials]
        super().__init__(params_list)

    def forward(self, energies):
        """
        Takes X-ray energies and returns the scintillator response.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The scintillator conversion function as a function of input energies, selected material, and its thickness.
        """
        mat = self.get_params()[f"{self.prefix}_material"]
        th = self.get_params()[f"{self.prefix}_thickness"]
        # print('ID scintillator th:', id(th))
        return gen_scint_cvt_func(energies, mat, th)

class Scintillator_MCNP(Base_Spec_Model):
    def __init__(self, thickness):
        """
        Initializes a scintillator model for interpolation over scintillator thickness.

        Args:
            thickness (tuple): A tuple containing three elements:
                - initial value (float or None): The initial value of the scintillator thickness.
                - lower bound (float or None): The lower bound for the scintillator thickness.
                - upper bound (float or None): The upper bound for the scintillator thickness.

                At least one of these values cannot be None. If the lower bound equals the upper bound,
                the thickness will not be optimized.
        """
        params_list = [{'thickness': thickness}]
        super().__init__(params_list)

    def set_scint_spec_list(self, scint_spec_list, thicknesses):
        """
        Sets the lookup table for interpolation, which will be used in the forward function.

        Args:
            scint_spec_list (numpy.ndarray): A 2D array containing the reference scintillator spectra.
                Each row in this array corresponds to a specific scintillator thickness from the `thicknesses` array.

            thicknesses (numpy.ndarray): A sorted 1D array containing the scintillator thicknesses corresponding
                to each spectrum in `scint_spec_list`.

        The method also computes the logarithmic attenuation for each spectrum, which is used for interpolation
        over the thickness range.
        """
        self.scint_spec_list = np.array(scint_spec_list)
        self.thicknesses = np.array(thicknesses)
        self.log_scint_spec_list = np.array([-np.log(1 - ss) for ss in scint_spec_list])
        self.scint_spec_interp_func_over_th = Interp1D(torch.tensor(self.thicknesses, dtype=torch.float32),
                                                       torch.tensor(self.log_scint_spec_list, dtype=torch.float32))

    def forward(self, energies):
        """
        Computes the scintillator response for given X-ray energies.

        Args:
            energies (torch.Tensor): A tensor containing X-ray energies of a poly-energetic response in keV.

        Returns:
            torch.Tensor: A tensor representing the interpolated scintillator response for energy integrating detector.
        """
        energies = torch.tensor(energies, dtype=torch.float32)
        thickness = self.get_params()[f"{self.prefix}_thickness"]
        src_spec = self.scint_spec_interp_func_over_th(thickness)
        src_spec = 1 - torch.exp(-src_spec)
        return src_spec * energies
