import types
import numpy as np
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

import spekpy as sp  # Import SpekPy

from xspec.chem_consts._consts_from_table import get_mass_absp_c_vs_E
from xspec.chem_consts._periodictabledata import atom_weights, density, ptableinverse
from xspec.dict_gen import gen_fltr_res, gen_scint_cvt_func

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

    def __init__(self, params_list):
        """
        Initialize the Base_Spec_Model.

        Args:
            params_list (list): List of dictionaries containing parameters.
        """
        super().__init__()
        if not hasattr(self.__class__, '_count'):
            self.__class__._count = 0
        self.__class__._count += 1
        self.name = f"{self.__class__.__name__}_{self.__class__._count}"


        # params_list contains all possible discrete parameters combinations and related continous parameters.
        self._params_list = []
        for params in params_list:
            new_params = {}
            for key, value in params.items():
                if self.__class__.__name__ != 'Base_Spec_Model':
                    modified_key = f"{self.name}_{key}"
                else:
                    modified_key =f"{key}"
                if isinstance(value, tuple):
                    new_params[f"{modified_key}"] = normalize_tuple_as_parameter(value)
                else:
                    new_params[f"{modified_key}"] = value
            self._params_list.append(new_params)
        self._init_estimates()

    def forward(self, energies):
        """
        Placeholder forward method.

        Args:
            energies (torch.Tensor): Input energies.

        Returns:
            torch.Tensor: Output energies.
        """
        # Placeholder forward method of self.estimates, replace with actual implementation.
        return torch.ones(len(energies))

    def _init_estimates(self):
        """
        Initialize estimates from the first dictionary in params_list.
        """
        self.estimates = {}
        for key, value in self._params_list[0].items():
            self.estimates[key] = value
            if isinstance(value, tuple):
                setattr(self, key, self.estimates[key][0])
            else:
                setattr(self, key, value)

    def set_estimates(self, params):
        """
        Set estimates from a dictionary of parameters.

        Args:
            params (dict): Dictionary containing parameters.
        """
        for key, value in params.items():
            if key in self.estimates.keys():
                if isinstance(value, tuple):
                    setattr(self, key, value[0])
                else:
                    setattr(self, key, value)
                self.estimates[key] = value


    def get_estimates(self):
        """
        Get estimates as a dictionary.

        Returns:
            dict: Dictionary containing estimates.
        """
        display_estimates = {}
        for key, value in self.estimates.items():
            if isinstance(value, tuple):
                dv = denormalize_parameter_as_tuple(value)
                display_estimates[key] =  clamp_with_grad(dv[0], dv[1], dv[2])
            else:
                display_estimates[key] = value
        return display_estimates

    # def _pair_params_list(self, other):
    #     """
    #     Pair parameters lists with other instance of Base_Spec_Model or its child model.
    #
    #     Args:
    #         other (Base_Spec_Model): Another instance of Base_Spec_Model.
    #
    #     Returns:
    #         list: List of paired parameters lists.
    #     """
    #     paired_params_list = []
    #     for params1 in self._params_list:
    #         for params2 in other._params_list:
    #             # Merge dictionaries and add instance names as prefixes to keys
    #             merged_params_dict = {}
    #             for key, value in list(params1.items())+list(params2.items()):
    #                 if isinstance(value, tuple):
    #                     vv = denormalize_parameter_as_tuple(value)
    #                     merged_params_dict[key] = (vv[0].item(), vv[1], vv[2])
    #                 else:
    #                     merged_params_dict[key] = value
    #             paired_params_list.append(merged_params_dict)
    #     return paired_params_list
    #
    # def new_forward(self, energies):
    #     # Calculate forward pass by multiplying outputs of forward passes of input models
    #     output1 = self.forward(energies)
    #     output2 = self.other.forward(energies)
    #     return output1 * output2
    #
    # def __mul__(self, other):
    #     """
    #     Concatenate with other instance of Base_Spec_Model or its child model.
    #     Update _params_list for discrete parameters combination.
    #     Update forward function for elementwise multiplication over energies.
    #
    #     Args:
    #         other (Base_Spec_Model): Another instance of Base_Spec_Model.
    #
    #     Returns:
    #         Base_Spec_Model: New instance resulting from the multiplication.
    #     """
    #     if isinstance(other, Base_Spec_Model):
    #         self.other = other
    #         paired_params_list = self._pair_params_list(other)
    #         # Return a new instance of the class
    #         new_model = Base_Spec_Model(paired_params_list)
    #
    #         # Define forward method for the new model
    #
    #         new_model.forward = self.new_forward
    #         return new_model
    #     else:
    #         raise TypeError("Unsupported operand type for *: Base_Spec_Model and {}".format(type(other)))


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


def interp_src_spectra(voltage, voltage_list, src_spec_list, torch_mode=True):
    """
    Interpolate the source spectral response based on a given source voltage.

    Parameters
    ----------
    voltage : float or int
        The source voltage at which the interpolation is desired.

    voltage_list : list
        List of source voltages representing the maximum X-ray energy for each source spectrum.

    src_spec_list : list
        List of corresponding source spectral responses for each voltage in `voltage_list`.

    torch_mode : bool, optional
        Determines the computation method. If set to True, PyTorch is used for optimization.
        If set to False, the function calculates the cost function without optimization.
        Default is True.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Interpolated source spectral response at the specified `interp_voltage`.
    """

    # Find corresponding voltage bin.
    if torch_mode:
        voltage_list = torch.tensor(voltage_list, dtype=torch.int32)
        src_spec_list = [torch.tensor(src_spec, dtype=torch.float32) for src_spec in src_spec_list]
        index = np.searchsorted(voltage_list.detach().clone().numpy(), voltage.detach().clone().numpy())
    else:
        index = np.searchsorted(voltage_list, voltage)
    v0 = voltage_list[index - 1]
    f0 = src_spec_list[index - 1]
    if index >= len(voltage_list):
        if torch_mode:
            return torch.clamp(f0, min=0)
        else:
            return np.clip(f0, 0, None)

    v1 = voltage_list[index]
    f1 = src_spec_list[index]

    # Extend 𝑓0 (v) to be negative for v>v0.
    f0_modified = f0.clone()  # Clone to avoid in-place modifications
    for v in range(v0, v1):
        if v == v1:
            f0_modified[v] = 0
        else:
            r = (v - float(v0)) / (v1 - float(v0))
            f0_modified[v] = -r / (1 - r) * f1[v]

    # Interpolation
    rr = (voltage - float(v0)) / (v1 - float(v0))
    interpolated_values = rr * f1 + (1 - rr) * f0_modified

    if torch_mode:
        return torch.clamp(interpolated_values, min=0)
    else:
        return np.clip(interpolated_values, 0, None)


def angle_sin(psi, torch_mode=False):
    if torch_mode:
        return torch.sin(psi * torch.pi / 180.0)
    else:
        return np.sin(psi * np.pi / 180.0)

class Reflection_Source(Base_Spec_Model):
    def __init__(self, voltage, takeoff_angle, single_takeoff_angle=False):
        """
        A template source model designed specifically for reflection sources, including all necessary methods.

        Args:
            voltage (tuple): (initial value, lower bound, upper bound) for the source voltage.
                These three values cannot be all None. It will not be optimized when lower == upper.
            takeoff_angle (tuple): (initial value, lower bound, upper bound) for the takeoff angle, in degrees.
                These three values cannot be all None. It will not be optimized when lower == upper.
        """
        params_list = [{'voltage': voltage, 'takeoff_angle': takeoff_angle}]
        super().__init__(params_list)
        self.single_takeoff_angle = single_takeoff_angle
        if self.single_takeoff_angle:
            for params in self._params_list:
                params[f"{self.__class__.__name__}_takeoff_angle"] = params.pop(f"{self.name}_takeoff_angle")
            self._init_estimates()

    def set_src_spec_list(self, src_voltage_list, src_spec_list, cur_takeoff_angle):
        self.src_voltage_list = src_voltage_list
        self.src_spec_list = src_spec_list
        self.cur_takeoff_angle = cur_takeoff_angle

    def forward(self, energies):
        """
        Takes X-ray energies and returns the source spectrum.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The source response.
        """

        voltage = self.get_estimates()[f"{self.name}_voltage"]
        src_spec = interp_src_spectra(voltage, self.src_voltage_list, self.src_spec_list)

        if self.single_takeoff_angle:
            takeoff_angle = self.get_estimates()[f"{self.__class__.__name__}_takeoff_angle"]
        else:
            takeoff_angle = self.get_estimates()[f"{self.name}_takeoff_angle"]
        # print('ID takeoff_angle:', id(takeoff_angle))
        sin_psi_cur = angle_sin(self.cur_takeoff_angle, torch_mode=False)
        sin_psi_new = angle_sin(takeoff_angle, torch_mode=True)
        src_spec = src_spec * takeoff_angle_conversion_factor(voltage, sin_psi_cur, sin_psi_new, energies)

        return src_spec


class Filter(Base_Spec_Model):

    def __init__(self, materials, thickness):
        """
        A template filter model based on Beer's Law and NIST mass attenuation coefficients, including all necessary methods.

        Args:
            materials (list): A list of possible materials for the filter,
                where each material should be an instance containing formula and density.
            thickness (tuple): (initial value, lower bound, upper bound) for the filter thickness.
                These three values cannot be all None. It will not be optimized when lower == upper.
        """
        params_list = [{'material': mat, 'thickness': thickness} for mat in materials]
        super().__init__(params_list)

    def forward(self, energies):
        """
        Takes X-ray energies and returns the filter response.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The filter response as a function of input energies, selected material, and its thickness.
        """
        mat = self.get_estimates()[f"{self.name}_material"]
        th = self.get_estimates()[f"{self.name}_thickness"]
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
        mat = self.get_estimates()[f"{self.name}_material"]
        th = self.get_estimates()[f"{self.name}_thickness"]
        # print('ID scintillator th:', id(th))
        return gen_scint_cvt_func(energies, mat, th)