import sys
import atexit
import warnings
import numpy as np

import torch
import torch.optim as optim
from torch.nn.parameter import Parameter

from torch.multiprocessing import Pool
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

import logging

from xspec._utils import *
from xspec.defs import *
from xspec.dict_gen import gen_fltr_res, gen_scint_cvt_func
from xspec.chem_consts._consts_from_table import get_mass_absp_c_vs_E
from xspec.chem_consts._periodictabledata import atom_weights, density, ptableinverse
from xspec.opt._pytorch_lbfgs.functions.LBFGS import FullBatchLBFGS as NNAT_LBFGS
from itertools import product


def estimate(energies, normalized_rads, forward_matrices, source_params, filter_params, scintillator_params,
             weight=None, weight_type='unweighted', blank_rads=None,
             learning_rate=0.001, max_iterations=5000, stop_threshold=1e-4, optimizer_type='Adam', loss_type='wmse', logpath=None,
             num_processes=1, return_all_result=False):
    """
    Estimate the X-ray CT parameters that determine the X-ray energy spectrum including the source voltage,
    anode take-off angle, filter material and thickness, and scintillator and thickness.

    Args:
        energies (numpy.ndarray): Array of interested X-ray photon energies in keV with size N_energies_bins.
        normalized_rads (list of numpy.ndarray): Normalized radiographs at different source voltages and filters.
            Each radiograph has dimensions [N_views, N_rows, N_cols].
        forward_matrices (list of numpy.ndarray): Corresponding forward matrices for normalized_rads. We provide ``xspec.calc_forward_matrix`` to calculate a forward matrix from a 3D mask for a homogenous object. Each forward matrix, corresponding to radiograph, has dimensions [N_views, N_rows, N_cols, N_energiy_bins].

        source_params (dict): Parameters defining the source model. Keys include:

            - 'num_voltage' (int): Number of used voltage to collect all normalized radiographs.
            - 'reference_voltages' (numpy.ndarray):  This is a sorted array containing the source voltages, each corresponding to a specific reference X-ray source spectrum.
            - 'reference_anode_angle' (float): This value represents the reference anode take-off angle, expressed in degrees, which is used in generating the reference X-ray spectra.
            - 'reference_spectra' (numpy.ndarray):  This array contains the reference X-ray source spectra. Each spectrum in this array corresponds to a specific combination of the reference_anode_angle and one of the source voltages from reference_voltages.
            - 'voltage_1' (float): X-ray source tube voltage in kVp, defines the maximum energy that these electrons can gain. A voltage of 50 kVp will produce a spectrum of X-ray energies with the theoretical maximum being 50 keV.
            - 'voltage_1_range' (tuple): Range of first voltage in kVp.
            - ...
            - 'anode_target_type (str)': 'transmission' or 'reflection'. The angle between the X-ray direction to the center of the detector and the anode target surface.
            - 'anode_angle' (float): Anode take-off angle in degrees. The anode take-off angle is the angle between the surface of the anode in an X-ray tube and the direction in which the majority of the X-rays are emitted.
            - 'anode_angle_range' (tuple): Range of anode angle in degrees.
            - 'optimize_voltage' (bool): Specify if requiring optimization over source voltage.
            - 'optimize_anode_angle' (bool): Specify if requiring optimization over anode_angle.
            - 'source_voltage_indices' (list): Specify which source voltage index corresponds to each radiograph.

        filter_params (dict): Parameters defining the filter system. Keys include:

            - 'num_filter' (int): Number of used filters to collect all normalized radiographs.
            - 'possible_material' (list): List of possible filter materials. Each item is an instance of Material.
            - 'material_1' (object): An instance of ``xspec.Material`` for the first filter.
            - 'thickness_1' (float): Thickness of the first filter in mm.
            - 'thickness_1_range' (tuple): Range of filter thickness in mm.
            - ...
            - 'optimize' (bool): Specify if requiring optimization over filter thickness.
            - 'filter_indices' (list): Each item is a list specify which filters used for each radiograph.

        scintillator_params (dict): Parameters defining the scintillator properties. Keys include:

            - 'possible_material' (list): List of possible scintillator materials. Each item is an instance of Material.
            - 'material' (object): An instance of ``xspec.Material`` for the scintillator.
            - 'thickness' (float): Thickness of the scintillator in mm.
            - 'thickness_range' (tuple): Range of scintillator thickness in mm.
            - 'optimize' (bool): Specify if requiring optimization over scintillator thickness.


        weight (optional): [Default=None] Weights to apply during the estimation process.
        weight_type (str, optional): [Default='unweighted'] Type of noise model used for data.
            Option “unweighted” corresponds to unweighted reconstruction;
            Option “transmission” is the correct weighting for transmission CT with constant dose or given blank
            radiograph.
        blank_rads (list of numpy.ndarray, optional): A list of blank (object-free) radiograph arrays, where each array corresponds to a specific radiograph and has dimensions [N_views, N_rows, N_cols]. These arrays are used in scenarios where the weight calculation is necessary. Specifically, when 'weight_type' is “transmission”, the weight is determined by the formula: blank radiograph / normalized radiograph. This approach assumes that the variance of the object radiograph is proportional to the object radiograph divided by the square of the blank radiograph.

        learning_rate (float, optional): [Default=0.001] Learning rate for the optimization process.
        max_iterations (int, optional): [Default=5000] Maximum number of iterations for the optimization.
        stop_threshold (float, optional): [Default=1e-4] Scalar valued stopping threshold in percent.
            If stop_threshold=0.0, then run max iterations.
        optimizer_type (str, optional): [Default='Adam'] Type of optimizer to use. If we do not have
            accurate initial guess use 'Adam', otherwise, 'NNAT_LBFGS' can provide a faster convergence.
        logpath (optional): [Default=None] Path for logging, if required.
        num_processes (int, optional): [Default=1] Number of processes to use for parallel computation.
        return_all_result (bool, optional): [Default=False] Flag to return all discrete cases results.

    Returns:
        dict: The estimated X-ray CT parameters.
    """
    # Create a tuple of arguments to be used later, excluding 'self', 'num_processes', and 'logpath'
    args = tuple(v for k, v in locals().items() if k != 'self' and k != 'num_processes' and k != 'logpath')

    # Create lists of Source, Filter, and Scintillator objects from the provided parameter dictionaries
    sources = dict_to_sources(source_params, energies)
    filters = dict_to_filters(filter_params)
    scintillators = dict_to_scintillator(scintillator_params)

    # Adjust indices from source and filter params to be zero-based (Python indexing starts from 0)
    source_voltage_indices = [i - 1 for i in source_params['source_voltage_indices']]
    filter_indices = [[i - 1 for i in fp] for fp in filter_params['filter_indices']]

    # Create Model_combination objects for all combinations of sources and filters
    model_combination = [Model_combination(src_ind, fltr_ind_set, 0) for src_ind, fltr_ind_set in
                         zip(source_voltage_indices, filter_indices)]

    # Generate all possible combinations of filters and scintillators
    possible_filters_combinations = [[fc for fc in fcm.next_psb_fltr_mat()] for fcm in filters]
    possible_scintillators_combinations = [[sc for sc in scm.next_psb_scint_mat()] for scm in scintillators]

    # Combine possible filters and scintillators into a single list of parameter combinations
    model_params_list = list(product(*possible_filters_combinations, *possible_scintillators_combinations))

    # Regroup filters and scintillators combinations into a structured format
    model_params_list = [nested_list(model_params, [len(d) for d in [possible_filters_combinations,
                                                                     possible_scintillators_combinations]]) for
                         model_params in model_params_list]

    # Initialize weights based on the weight type
    if weight is None:
        if weight_type == 'unweighted':
            weight = [1.0 + 0 * yy for yy in normalized_rads]
        elif weight_type == 'transmission':
            if blank_rads is None:
                weight = [1.0 / yy for yy in normalized_rads]
            else:
                weight = [br / yy for br, yy in zip(blank_rads, normalized_rads)]

    # Convert weights to tensor format
    weight = [torch.tensor(np.concatenate([w.reshape((-1, 1)) for w in ww]), dtype=torch.float32) for ww in weight]

    # Use multiprocessing pool to parallelize the optimization process
    with Pool(processes=num_processes, initializer=init_logging, initargs=(logpath, num_processes)) as pool:
        # Apply optimization function to each combination of model parameters
        result_objects = [
            pool.apply_async(
                param_based_spec_estimate_cell,
                args=args[:3] + (
                sources, possible_filters, possible_scintillators, model_combination,
                weight, learning_rate,
                max_iterations, stop_threshold,
                optimizer_type, loss_type, False)
            )
            for possible_filters, possible_scintillators in model_params_list
        ]

        # Gather results from all parallel optimizations
        print('Number of cases for different discrete parameters:', len(result_objects))
        results = [r.get() for r in result_objects]  # Retrieve results from async calls

    # Decide what to return based on 'return_all_result' flag
    if return_all_result:
        return results  # Return all results if requested
    else:
        # Find and return the result with the optimal cost
        cost_list = [res[1] for res in results]
        optimal_cost_ind = np.argmin(cost_list)
        best_res = results[optimal_cost_ind][2]
        res_params = dict()
        if source_params['optimize_voltage']:
            for i in range(source_params['num_voltage']):
                res_params['voltage_%d'%(i+1)] = best_res.src_spec_list[i].get_voltage().item()
        if source_params['anode_target_type']=='reflection':
            if source_params['optimize_anode_angle']:
                res_params['anode_angle'] = best_res.src_spec_list[0].get_takeoff_angle()
        if filter_params['optimize']:
            for i in range(filter_params['num_filter']):
                res_params['filter_%d_mat'%(i+1)] = best_res.fltr_resp_list[i].get_fltr_mat()
                res_params['filter_%d_thickness'%(i+1)] = best_res.fltr_resp_list[i].get_fltr_th().item()
        if scintillator_params['optimize']:
            res_params['scintillator_mat']=best_res.scint_cvt_list[0].get_scint_mat()
            res_params['scintillator_thickness']=best_res.scint_cvt_list[0].get_scint_th().item()
        return res_params


def calc_source_spectrum(energies, reference_voltages, reference_anode_angle, reference_spectra, voltage, anode_angle):
    """Calculate source spectrum with given parameters.

    Args:
        energies (numpy.ndarray): Array of interested X-ray photon energies in keV.
        reference_voltages (numpy.ndarray): This is a sorted array containing the source voltages, each corresponding to a specific reference X-ray source spectrum.
        reference_anode_angle (float): This value represents the reference anode take-off angle, expressed in degrees, which is used in generating the reference X-ray spectra.
        reference_spectra (numpy.ndarray): This array contains the reference X-ray source spectra. Each spectrum in this array corresponds to a specific combination of the reference_anode_angle and one of the source voltages from reference_voltages.
        voltage (float): X-ray source tube voltage in kVp, defines the maximum energy that these electrons can gain. A voltage of 50 kVp will produce a spectrum of X-ray energies with the theoretical maximum being 50 keV.
        anode_angle (float): Anode take-off angle in degrees. The anode take-off angle is the angle between the surface of the anode in an X-ray tube and the direction in which the majority of the X-rays are emitted.

    Returns:
        numpy.ndarray: The calculated source spectrum with given parameters.
    """
    source = Source(energies=energies,
                     src_voltage_list=reference_voltages,
                     takeoff_angle_cur=reference_anode_angle,
                     src_spec_list=reference_spectra,
                     voltage=voltage,
                     takeoff_angle=anode_angle,
                     optimize_voltage=False,
                     optimize_takeoff_angle=False)
    src_model = Source_Model(source)

    return src_model(energies).data


def calc_filter_response(energies, material, thickness):
    """Calculate filter response with given parameters.

    Args:
        energies (numpy.ndarray): Array of interested X-ray photon energies in keV.
        material (object): An instance of ``xspec.Material`` for the filter, containing chemical formula and density.
        thickness (float): Thickness of the filter in mm.
    Returns:
        numpy.ndarray: The calculated filter response with given parameters.
    """
    filter = Filter(fltr_mat=material, fltr_th=thickness, optimize=False)
    fltr_model = Filter_Model(filter)
    return fltr_model(energies).data

def calc_scintillator_response(energies, material, thickness):
    """Calculate scintillator response with given parameters.

    Args:
        energies (numpy.ndarray): Array of interested X-ray photon energies in keV.
        material (object): An instance of ``xspec.Material`` for the scintillator, containing chemical formula and density.
        thickness (float): Thickness of the scintillator in mm.
    Returns:
        numpy.ndarray: The calculated scintillator response with given parameters.
    """
    scintillator = Scintillator(scint_mat=material, scint_th=thickness, optimize=False)
    scint_model = Scintillator_Model(scintillator)
    return scint_model(energies).data


def calc_forward_matrix(homogenous_vol_masks, lac_vs_energies, forward_projector):
    """
    Calculate the forward matrix for a combination of multiple solid objects using a given forward projector.

    Args:
        homogenous_vol_masks (list of numpy.ndarray): Each 3D array in the list represents a mask for a homogenous,
            pure object.
        lac_vs_energies (list of numpy.ndarray): Each 1D array contains the linear attenuation coefficient (LAC)
            curve and the corresponding energies for the materials represented in `homogenous_vol_masks`.
        forward_projector (object): An instance of a class that implements a forward projection method. This
            instance should have a method, forward(mask), takes a 3D volume mask as input and computes the photon's line
            path length.

    Returns:
        numpy.ndarray: The calculated forward matrix for spectral estimation.
    """

    linear_att_intg_list = []
    for mask, lac_vs_energies in zip(homogenous_vol_masks, lac_vs_energies):
        linear_intg = forward_projector.forward(mask)
        linear_att_intg_list.append(
            linear_intg[np.newaxis, :, :, :] * lac_vs_energies[:, np.newaxis, np.newaxis, np.newaxis])

    tot_lai = np.sum(np.array(linear_att_intg_list), axis=0)
    forward_matrix = np.exp(- tot_lai.transpose((1, 2, 3, 0)))

    return forward_matrix


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


def interp_src_spectra(voltage_list, src_spec_list, interp_voltage, torch_mode=True):
    """
    Interpolate the source spectral response based on a given source voltage.

    Parameters
    ----------
    voltage_list : list
        List of source voltages representing the maximum X-ray energy for each source spectrum.

    src_spec_list : list
        List of corresponding source spectral responses for each voltage in `voltage_list`.

    interp_voltage : float or int
        The source voltage at which the interpolation is desired.

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
        index = np.searchsorted(voltage_list.detach().clone().numpy(), interp_voltage.detach().clone().numpy())
    else:
        index = np.searchsorted(voltage_list, interp_voltage)
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
    rr = (interp_voltage - float(v0)) / (v1 - float(v0))
    interpolated_values = rr * f1 + (1 - rr) * f0_modified

    if torch_mode:
        return torch.clamp(interpolated_values, min=0)
    else:
        return np.clip(interpolated_values, 0, None)


def angle_sin(psi):
    return np.sin(psi * np.pi / 180.0)


class Source_Model(torch.nn.Module):
    def __init__(self, source: Source, device=None, dtype=None) -> None:
        """Source Model

        Parameters
        ----------
        source: Source
            Source model configuration.
        device
        dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.source = source

        # Min-Max Normalization
        self.volt_lower = source.src_voltage_bound.lower
        self.volt_scale = source.src_voltage_bound.upper - source.src_voltage_bound.lower
        normalized_voltage = (source.voltage - self.volt_lower) / self.volt_scale

        # Instantiate parameters
        if source.optimize_voltage:
            self.normalized_voltage = Parameter(torch.tensor(normalized_voltage, **factory_kwargs))
        else:
            self.normalized_voltage = torch.tensor(normalized_voltage, **factory_kwargs)

        if self.source.anode_target_type == 'reflection':
            self.toa_lower = angle_sin(source.takeoff_angle_bound.lower)
            self.toa_scale = angle_sin(source.takeoff_angle_bound.upper) - angle_sin(source.takeoff_angle_bound.lower)
            normalized_sin_psi = (angle_sin(source.takeoff_angle) - self.toa_lower) / self.toa_scale
            if self.source.optimize_takeoff_angle:
                self.normalized_sin_psi = Parameter(torch.tensor(normalized_sin_psi, **factory_kwargs))
            else:
                self.normalized_sin_psi = torch.tensor(normalized_sin_psi, **factory_kwargs)

    def get_voltage(self):
        """Read voltage.

        Returns
        -------
        voltage: float
            Read voltage.
        """

        return clamp_with_grad(self.normalized_voltage, 0, 1) * self.volt_scale + self.volt_lower

    def get_takeoff_angle(self):
        """Read takeoff_angle.

        Returns
        -------
        voltage: float
            Read takeoff_angle.
        """
        if self.source.anode_target_type == 'reflection':
            return np.arcsin(np.clip(self.normalized_sin_psi.detach().numpy(), 0,
                                     1) * self.toa_scale + self.toa_lower) * 180.0 / np.pi
        else:
            # If anode_target_type is not 'reflection', raise an error
            raise ValueError("anode_target_type must be 'reflection' to run get_takeoff_angle.")


    def get_sin_psi(self):
        """Read takeoff_angle.

        Returns
        -------
        voltage: float
            Read takeoff_angle.
        """
        if self.source.anode_target_type == 'reflection':
            return clamp_with_grad(self.normalized_sin_psi, 0, 1) * self.toa_scale + self.toa_lower
        else:
            # If anode_target_type is not 'reflection', raise an error
            raise ValueError("anode_target_type must be 'reflection' to run get_sin_psi.")

    def forward(self, energies):
        """Calculate source spectrum.

        Returns
        -------
        src_spec: torch.Tensor
            Source spectrum.

        """
        src_spec = interp_src_spectra(self.source.src_voltage_list, self.source.src_spec_list, self.get_voltage())
        if self.source.anode_target_type == 'reflection':
            sin_psi_cur = angle_sin(self.source.takeoff_angle_cur)
            src_spec = src_spec * takeoff_angle_conversion_factor(self.get_voltage(), sin_psi_cur, self.get_sin_psi(),
                                                                  energies)
        return src_spec


class Filter_Model(torch.nn.Module):
    def __init__(self, filter: Filter, device=None, dtype=None) -> None:
        """Filter module

        Parameters
        ----------
        filter: Filter
            Filter model configuration.
        device
        dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.filter = filter

        # Min-Max Normalization
        self.lower = filter.fltr_th_bound.lower
        self.scale = filter.fltr_th_bound.upper - filter.fltr_th_bound.lower
        normalized_fltr_th = (filter.fltr_th - self.lower) / self.scale
        # Instantiate parameters
        if filter.optimize:
            self.normalized_fltr_th = Parameter(torch.tensor(normalized_fltr_th, **factory_kwargs))
        else:
            self.normalized_fltr_th = torch.tensor(normalized_fltr_th, **factory_kwargs)

    def get_fltr_th(self):
        """Get filter thickness.

        Returns
        -------
        fltr_th_list: list
            List of filter thickness. Length is equal to num_fltr.

        """
        return clamp_with_grad(self.normalized_fltr_th, 0, 1) * self.scale + self.lower

    def get_fltr_mat(self):
        """Get filter thickness.

        Returns
        -------
        fltr_mat: xspec.Material
            Filter material.

        """
        return self.filter.fltr_mat

    def forward(self, energies):
        """Calculate filter response.

        Parameters
        ----------
        energies : list
            List of X-ray energies of a poly-energetic source in units of keV.

        fltr_ind_list: list of int

        Returns
        -------
        fltr_resp: torch.Tensor
            Filter response.

        """

        return gen_fltr_res(energies, self.filter.fltr_mat, self.get_fltr_th())


class Scintillator_Model(torch.nn.Module):
    def __init__(self, scintillator: Scintillator, device=None, dtype=None) -> None:
        """Scintillator convertion model

        Parameters
        ----------
        scintillator: Scintillator
            Sinctillator model configuration.
        device
        dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.scintillator = scintillator

        # Min-Max Normalization
        self.lower = scintillator.scint_th_bound.lower
        self.scale = scintillator.scint_th_bound.upper - scintillator.scint_th_bound.lower
        normalized_scint_th = (scintillator.scint_th - self.lower) / self.scale
        # Instantiate parameter
        if scintillator.optimize:
            self.normalized_scint_th = Parameter(torch.tensor(normalized_scint_th, **factory_kwargs))
        else:
            self.normalized_scint_th = torch.tensor(normalized_scint_th, **factory_kwargs)

    def get_scint_th(self):
        """

        Returns
        -------
        scint_th: float
            Sintillator thickness.

        """
        return clamp_with_grad(self.normalized_scint_th, 0, 1) * self.scale + self.lower

    def get_scint_mat(self):
        """

        Returns
        -------
        scint_th: float
            Sintillator thickness.

        """
        return self.scintillator.scint_mat

    def forward(self, energies):
        """Calculate scintillator convertion function.

        Parameters
        ----------
        energies: list
            List of X-ray energies of a poly-energetic source in units of keV.

        Returns
        -------
        scint_cvt_func: torch.Tensor
            Scintillator convertion function.

        """

        return gen_scint_cvt_func(energies, self.scintillator.scint_mat, self.get_scint_th())


class spec_distrb_energy_resp(torch.nn.Module):
    def __init__(self,
                 energies,
                 sources: [Source],
                 filters: [Filter],
                 scintillators: [Scintillator], optimizer_type='Adam', device=None, dtype=None):
        """Total spectrally distributed energy response model based on torch.nn.Module.

        Parameters
        ----------
        energies: list
            List of X-ray energies of a poly-energetic source in units of keV.
        sources: list of Source
            List of source model configurations.
        filters: list of Filter
            List of filter model configurations.
        scintillators: list of Scintillator
            List of scintillator model configurations.
        device
        dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.ot = optimizer_type
        self.energies = torch.Tensor(energies) if energies is not torch.Tensor else energies
        self.src_spec_list = torch.nn.ModuleList(
            [Source_Model(source, **factory_kwargs) for source in sources])

        if sources[0].anode_target_type == 'reflection':
            if sources[0].optimize_takeoff_angle:
                for smm in self.src_spec_list[1:]:
                    smm._parameters['normalized_sin_psi'] = self.src_spec_list[0]._parameters['normalized_sin_psi']
        self.fltr_resp_list = torch.nn.ModuleList(
            [Filter_Model(filter, **factory_kwargs) for filter in filters])
        self.scint_cvt_list = torch.nn.ModuleList(
            [Scintillator_Model(scintillator, **factory_kwargs) for scintillator in scintillators])
        self.logger = logging.getLogger(str(mp.current_process().pid))

    def print_method(self, *args, **kwargs):
        message = ' '.join(map(str, args))
        self.logger.info(message)

    def update_optimizer_type(self, optimizer_type):
        self.ot = optimizer_type

    def forward(self, F: torch.Tensor, mc: Model_combination):
        """

        Parameters
        ----------
        F:torch.Tensor
            Forward matrix. Row is number of measurements. Column is number of energy bins.
        mc: Model_combination
            Guide which source, filter and scintillator models are used.

        Returns
        -------
        trans_value: torch.Tensor
            Transmission value calculated by total spectrally distributed energy response model.

        """
        if self.ot == 'Adam':
            with torch.no_grad():
                if self.src_spec_list[mc.src_ind].source.anode_target_type == 'reflection':
                    if self.src_spec_list[mc.src_ind].source.optimize_takeoff_angle:
                        self.src_spec_list[mc.src_ind]._parameters['normalized_sin_psi'].data.clamp_(min=1e-6, max=1 - 1e-6)
                for fii in mc.fltr_ind_list:
                    if self.fltr_resp_list[fii].filter.optimize:
                        self.fltr_resp_list[fii]._parameters['normalized_fltr_th'].data.clamp_(min=1e-6, max=1 - 1e-6)
                self.scint_cvt_list[mc.scint_ind]._parameters['normalized_scint_th'].data.clamp_(min=1e-6, max=1 - 1e-6)

        src_func = self.src_spec_list[mc.src_ind](self.energies)
        fltr_func = self.fltr_resp_list[mc.fltr_ind_list[0]](self.energies)
        for fii in mc.fltr_ind_list[1:]:
            fltr_func = fltr_func * self.fltr_resp_list[fii](self.energies)
        scint_func = self.scint_cvt_list[mc.scint_ind](self.energies)

        # Calculate total system response as a product of source, filter, and scintillator responses.
        total_sder = src_func * fltr_func * scint_func
        total_sder /= torch.trapz(total_sder, self.energies)
        trans_value = torch.trapz(F * total_sder, self.energies, axis=-1).reshape((-1, 1))
        return trans_value

    def print_parameters(self):
        """
        Print all parameters of the model.
        """
        if self.print_method is not None:
            print = self.print_method
        for name, param in self.named_parameters():
            print(f"Name: {name} | Size: {param.size()} | Values : {param.data} | Requires Grad: {param.requires_grad}")

    def print_ori_parameters(self):
        """
        Print all scaled-back parameters of the model.
        """
        if self.print_method is not None:
            print = self.print_method

        for src_i, src_spec in enumerate(self.src_spec_list):
            print('Source %d: Voltage: %.2f;' % (src_i, src_spec.get_voltage().item()))

        if self.src_spec_list[0].source.anode_target_type == 'reflection':
            print('Take-off Angle: %.2f' % (src_spec.get_takeoff_angle().item()))

        for fltr_i, fltr_resp in enumerate(self.fltr_resp_list):
            print(
                f'Filter {fltr_i}: Material: {fltr_resp.get_fltr_mat()}, Thickness: {fltr_resp.get_fltr_th()}')

        for scint_i, scint_cvt in enumerate(self.scint_cvt_list):
            print(f'Scintillator {scint_i}: Material:{scint_cvt.get_scint_mat()} Thickness:{scint_cvt.get_scint_th()}')


def weighted_mse_loss(input, target, weight):
    return 0.5 * torch.mean(weight * (input - target) ** 2)


def param_based_spec_estimate_cell(energies,
                                   y,
                                   F,
                                   sources: [Source],
                                   filters: [Filter],
                                   scintillators: [Scintillator],
                                   model_combination: [Model_combination],
                                   weight=None,
                                   learning_rate=0.02,
                                   max_iterations=5000,
                                   stop_threshold=1e-3,
                                   optimizer_type='NNAT_LBFGS',
                                   loss_type='wmse',
                                   return_history=False):
    """Other arguments are same as param_based_spec_estimate.

    Parameters
    ----------
    filters : list of Filter
        Each Filter.fltr_mat should be specified to a Material instead of None.
    scintillators
        Each Scintillator.scint_mat should be specified to a Material instead of None.

    Returns
    -------
    stop_iter : int
        Stop iteration.
    final_cost_value : float
        Final cost value
    final_model : spec_distrb_energy_resp
        An instance of spec_distrb_energy_resp after optimization, containing
    """
    logger = logging.getLogger(str(mp.current_process().pid))

    def print(*args, **kwargs):
        message = ' '.join(map(str, args))
        logger.info(message)

    # Check Variables
    if return_history:
        src_voltage_list = []
        fltr_th_list = []
        scint_th_list = []
        cost_list = []

    # Construct our model by instantiating the class defined above
    model = spec_distrb_energy_resp(energies, sources, filters, scintillators, device='cpu', dtype=torch.float32)
    model.print_parameters()

    loss = torch.nn.MSELoss()

    if optimizer_type == 'Adam':
        ot = 'Adam'
        iter_prt = 50
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'NNAT_LBFGS':
        ot = 'NNAT_LBFGS'
        iter_prt = 5
        optimizer = NNAT_LBFGS(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'Adam+NNAT_LBFGS':
        ot = 'Adam'
        iter_prt = 50
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizerN = NNAT_LBFGS(model.parameters(), lr=learning_rate)
    else:
        warnings.warn(f"The optimizer type {optimizer_type} is not supported.")
        sys.exit("Exiting the script due to unsupported optimizer type.")

    model.update_optimizer_type(ot)

    # print('Initial optimizer:', optimizer)
    # print('Initial optimizer_type:', model.ot)

    y = [torch.tensor(np.concatenate([sig.reshape((-1, 1)) for sig in yy]), dtype=torch.float32) for yy in y]
    num_sp_datasets = len(y)
    if weight is None:
        weight = [1.0 / yy for yy in y]
    else:
        weight = [torch.tensor(np.concatenate([w.reshape((-1, 1)) for w in ww]), dtype=torch.float32) for ww in weight]

    F = [torch.tensor(FF, dtype=torch.float32) for FF in F]

    cost = np.inf
    LBFGS_iter = 0
    for iter in range(1, max_iterations + 1):
        if iter % iter_prt == 0:
            print('Iteration:', iter)

        if optimizer_type == 'Adam+NNAT_LBFGS':
            if cost <= 0.6 * num_sp_datasets or iter > 5000:

                if optimizer != optimizerN:
                    print('Start use NNAT_LBFGS')
                    print('Current cost value:', cost.item())
                    iter_prt = 5
                    ot = 'NNAT_LBFGS'
                    optimizer = optimizerN
                    model.update_optimizer_type(ot)
                else:
                    LBFGS_iter += 1

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            cost = 0
            for yy, FF, ww, mc in zip(y, F, weight, model_combination):
                trans_val = model(FF, mc)
                if loss_type == 'mse':
                    sub_cost = 0.5 * loss(trans_val, yy)
                elif loss_type == 'wmse':
                    sub_cost = weighted_mse_loss(trans_val, yy, ww)
                elif loss_type == 'attmse':
                    sub_cost = 0.5 * loss(-torch.log(trans_val), -torch.log(yy))
                else:
                    raise ValueError('loss_type should be \'mse\' or \'wmse\' or \'attmse\'. ', 'Given', loss_type)
                cost += sub_cost
            if cost.requires_grad and ot != 'NNAT_LBFGS':
                cost.backward()
            return cost

        cost = closure()
        if torch.isnan(cost):
            model.print_parameters()
            return iter, closure().item(), model

        if ot == 'NNAT_LBFGS':
            cost.backward()

        has_nan = check_gradients_for_nan(model)
        if has_nan:
            return iter, closure().item(), model

        with (torch.no_grad()):
            if iter == 1:
                print('Initial cost: %e' % (closure().item()))

        # Before the update, clone the current parameters
        old_params = {k: v.clone() for k, v in model.state_dict().items()}

        if ot == 'Adam':
            optimizer.step()
        elif ot == 'NNAT_LBFGS':
            options = {'closure': closure, 'current_loss': cost,
                       'max_ls': 100, 'damping': False}
            cost, grad_new, _, _, closures_new, grads_new, desc_dir, fail = optimizer.step(options=options)

        with (torch.no_grad()):
            if iter % iter_prt == 0:
                print('Cost:', cost.item())
                model.print_ori_parameters()

            # After the update, check if the update is too small
            small_update = True
            for k, v in model.state_dict().items():
                if torch.norm(v.clamp(0, 1) - old_params[k].clamp(0, 1)) > stop_threshold:
                    small_update = False
                    break

            if small_update: #or LBFGS_iter > max_iterations - 5000
                print(f"Stopping at epoch {iter} because updates are too small.")
                print('Cost:', cost.item())
                # for k, v in model.state_dict().items():
                #     print(v.item(), old_params[k].item())
                model.print_ori_parameters()
                break
    return iter, cost.item(), model


def init_logging(filename, num_processes):
    worker_id = mp.current_process().pid
    logger = logging.getLogger(str(worker_id))
    logger.setLevel(logging.INFO)

    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(f"{filename}_{worker_id % num_processes}.log")

    formatter = logging.Formatter('%(asctime)s  - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Register a cleanup function to close the logger when the process exits
    atexit.register(close_logging, logger)


def close_logging(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def param_based_spec_estimate(energies,
                              y,
                              F,
                              sources: [Source],
                              filters: [Filter],
                              scintillators: [Scintillator],
                              model_combination: [Model_combination],
                              weight=None,
                              learning_rate=0.02,
                              max_iterations=5000,
                              stop_threshold=1e-3,
                              optimizer_type='NNAT_LBFGS',
                              loss_type='wmse',
                              logpath=None,
                              num_processes=1,
                              return_history=False):
    """

    Parameters
    ----------
    energies : list
        List of X-ray energies of a poly-energetic source in units of keV.
    y : list
        Transmission Data :math:`y`.  (#datasets, #samples, #views, #rows, #columns).
        Normalized transmission data, background should be close to 1.
    F : list
        Forward matrix. (#datasets, #samples, #views, #rows, #columns, #energy_bins)
    sources : list of Source
        Specify all sources used across datasets.
    filters : list of Filter
        Specify all filters used across datasets. For each filter, Filter.possible_mat is required.
        The function will find out the best filter material among Filter.possible_mat.
    scintillators : list of Scintillator
        Specify all scintillators used across datasets. For each scintillator, Scintillator.possible_mat is required.
        The function will find out the best scintillator material among Scintillator.possible_mat.
    model_combination : list of Model_combination
        Each instance of Model_combination specify one experimental scenario. Length is equal to #datasets of y.
    learning_rate : int
        Learning rate for optimization.
    max_iterations : int
        Integer valued specifying the maximum number of iterations.
    stop_threshold : float
        Stop when all parameters update is less than tolerance.
    optimizer_type : str
        'Adam' or 'NNAT_LBFGS'
    loss_type : str
        'mse' or 'wmse'
    num_processes : int
        Number of parallel processes to run over possible filters and scintillators.
    logpath : str or None
        If None, print in terminal.
        If str, print to logpath and for each processor, print to a specific logfile with name logpath+'_'+pid.
    return_history : bool
        Save history of parameters.

    Returns
    -------

    """
    args = tuple(v for k, v in locals().items() if k != 'self' and k != 'num_processes' and k != 'logpath')

    possible_filters_combinations = [[fc for fc in fcm.next_psb_fltr_mat()] for fcm in filters]
    possible_scintillators_combinations = [[sc for sc in scm.next_psb_scint_mat()] for scm in scintillators]

    # Combine possible filters and scintillators
    model_params_list = list(product(*possible_filters_combinations, *possible_scintillators_combinations))
    # Regroup filters and scintillators
    model_params_list = [nested_list(model_params, [len(d) for d in [possible_filters_combinations,
                                                                     possible_scintillators_combinations]]) for
                         model_params in
                         model_params_list]

    with Pool(processes=num_processes, initializer=init_logging, initargs=(logpath, num_processes)) as pool:
        result_objects = [
            pool.apply_async(
                param_based_spec_estimate_cell,
                args=args[:4] + (possible_filters, possible_scintillators,) + args[6:]
            )
            for possible_filters, possible_scintillators in model_params_list
        ]

        # Gather results
        print('Number of parallel optimizations:', len(result_objects))
        results = [r.get() for r in result_objects]

    cost_list = [res[1] for res in results]
    optimal_cost_ind = np.argmin(cost_list)
    best_res = results[optimal_cost_ind][2]
    print('Optimal Result:')
    print('Cost:', cost_list[optimal_cost_ind])
    for src_i, src_spec in enumerate(best_res.src_spec_list):
        print('Source %d: Voltage: %.2f; Take-off Angle: %.2f' % (
        src_i, src_spec.get_voltage().item(), src_spec.get_takeoff_angle().item()))

    for fltr_i, fltr_resp in enumerate(best_res.fltr_resp_list):
        print(
            f'Filter {fltr_i}: Material: {fltr_resp.get_fltr_mat()}, Thickness: {fltr_resp.get_fltr_th()}')

    for scint_i, scint_cvt in enumerate(best_res.scint_cvt_list):
        print(f'Scintillator {scint_i}: Material:{scint_cvt.get_scint_mat()} Thickness:{scint_cvt.get_scint_th()}')
    return results
