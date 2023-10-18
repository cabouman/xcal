import sys
import warnings
import numpy as np

import torch
import torch.optim as optim
from torch.nn.parameter import Parameter

from multiprocessing import Pool

from xspec._utils import is_sorted, min_max_normalize_scalar, min_max_denormalize_scalar, concatenate_items, split_list
from xspec.defs import *
from xspec.dict_gen import gen_fltr_res, gen_scint_cvt_func
from xspec.opt._pytorch_lbfgs.functions.LBFGS import FullBatchLBFGS as NNAT_LBFGS
from itertools import product


# def anal_cost(energies, y, F, src_response, fltr_mat, scint_mat, th_fl, th_sc, signal_weight=None, torch_mode=True):
#     signal = np.concatenate([sig.reshape((-1, 1)) for sig in y])
#     forward_mat = np.concatenate([fwm.reshape((-1, fwm.shape[-1])) for fwm in F])
#     if signal_weight is None:
#         signal_weight = np.ones(signal.shape)
#     signal_weight = np.concatenate([sig.reshape((-1, 1)) for sig in signal_weight])
#
#     if torch_mode:
#         if not isinstance(src_response, torch.Tensor):
#             src_response = torch.tensor(src_response)
#         signal = torch.tensor(signal)
#         forward_mat = torch.tensor(forward_mat)
#         signal_weight = torch.tensor(signal_weight)
#
#     # Calculate filter response
#     fltr_params = [
#         {'formula': fltr_mat['formula'], 'density': fltr_mat['density'], 'thickness_list': [th_fl]},
#     ]
#     fltr_response, fltr_info = gen_filts_specD(energies, composition=fltr_params, torch_mode=torch_mode)
#
#     # Calculate scintillator response
#     scint_params = [
#         {'formula': scint_mat['formula'], 'density': scint_mat['density'], 'thickness_list': [th_sc]},
#     ]
#     scint_response, scint_info = gen_scints_specD(energies, composition=scint_params, torch_mode=torch_mode)
#
#     # Calculate total system response as a product of source, filter, and scintillator responses.
#     sys_response = (src_response * fltr_response * scint_response).T
#
#     if torch_mode:
#         Fx = torch.trapz(forward_mat * sys_response.T, torch.tensor(energies), axis=1).reshape((-1, 1))
#         e = signal - Fx / torch.trapz(sys_response, torch.tensor(energies), axis=0)
#     else:
#         Fx = np.trapz(forward_mat * sys_response.T, energies, axis=1).reshape((-1, 1))
#         e = signal - Fx / np.trapz(sys_response, energies, axis=0)
#
#     return cal_cost(e, signal_weight, torch_mode)
#
#
# def project_onto_constraints(x, lower_bound, upper_bound):
#     # return torch.clamp(x, min=lower_bound, max=upper_bound)
#     return torch.clamp(x, min=torch.tensor(lower_bound), max=torch.tensor(upper_bound))
#

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
        # index = torch.searchsorted(voltage_list, interp_voltage)
        voltage_list = torch.tensor(voltage_list, dtype=torch.int32)
        src_spec_list = torch.tensor(src_spec_list, dtype=torch.float32)
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


# def anal_sep_model(energies, signal_train_list, spec_F_train_list, src_response_dict=None, fltr_mat=None,
#                    scint_mat=None,
#                    init_src_vol=[50.0], init_fltr_th=1.0, init_scint_th=0.1, src_vol_bound=None, fltr_th_bound=(0, 10),
#                    scint_th_bound=(0.01, 1),
#                    learning_rate=0.1, iterations=5000, tolerance=1e-6, optimizer_type='Adam', return_history=False):
#     """
#
#     Parameters
#     ----------
#     energies
#     signal_train_list
#     spec_F_train_list
#     src_response_dict
#     fltr_mat
#     scint_mat
#     init_src_vol
#     init_fltr_th
#     init_scint_th
#     src_vol_bound
#     fltr_th_bound
#     scint_th_bound
#     learning_rate
#     iterations
#     tolerance
#     optimizer_type
#     return_history
#
#     Returns
#     -------
#
#     """
#
#     if return_history:
#         src_voltage_list = []
#         fltr_th_list = []
#         scint_th_list = []
#         cost_list = []
#
#     torch.autograd.set_detect_anomaly(True)
#
#     src_spec_list = [ssl['spectrum'] for ssl in src_response_dict]
#     src_kV_list = [ssl['source_voltage'] for ssl in src_response_dict]
#
#     if not is_sorted(src_kV_list):
#         raise ValueError("Warning: source voltage in src_response_dict are not sorted!")
#
#     # Sorted list of values
#     src_spec_list = torch.tensor(src_spec_list, dtype=torch.float32)
#     src_kV_list = torch.tensor(src_kV_list, dtype=torch.int32)
#
#     if 'thickness_bound' in fltr_mat:
#         if fltr_mat['thickness_bound'] is not None:
#             fltr_th_bound = fltr_mat['thickness_bound']
#     if 'thickness_bound' in scint_mat:
#         if scint_mat['thickness_bound'] is not None:
#             scint_th_bound = scint_mat['thickness_bound']
#
#     src_vol_range = [(src_kV_list[0], src_kV_list[-1]) for sv in init_src_vol]
#     src_vol_lower_range_list = [lb for lb, _ in src_vol_range]
#     src_vol_upper_range_list = [ub for _, ub in src_vol_range]
#
#     if src_vol_bound is None:
#         src_vol_bound = src_vol_range
#     src_vol_lower_bound_list = min_max_normalize_scalar([lb for lb, _ in src_vol_bound],
#                                                         src_vol_lower_range_list,
#                                                         src_vol_upper_range_list).tolist()
#     src_vol_upper_bound_list = min_max_normalize_scalar([ub for _, ub in src_vol_bound],
#                                                         src_vol_lower_range_list,
#                                                         src_vol_upper_range_list).tolist()
#
#     init_params, length = concatenate_items(init_src_vol, init_fltr_th, init_scint_th)
#     init_lower_ranges, _ = concatenate_items(src_vol_lower_range_list, fltr_th_bound[0], scint_th_bound[0])
#     init_upper_ranges, _ = concatenate_items(src_vol_upper_range_list, fltr_th_bound[1], scint_th_bound[1])
#     init_lower_bounds, _ = concatenate_items(src_vol_lower_bound_list, 0, 0)
#     init_upper_bounds, _ = concatenate_items(src_vol_upper_bound_list, 1, 1)
#     norm_params = torch.tensor(min_max_normalize_scalar(init_params,
#                                                         init_lower_ranges,
#                                                         init_upper_ranges), requires_grad=True)
#
#     if optimizer_type == 'Adam':
#         optimizer = optim.Adam([norm_params], lr=learning_rate)
#     elif optimizer_type == 'LBFGS':
#         optimizer = optim.LBFGS([norm_params], lr=learning_rate, max_iter=100, tolerance_grad=1e-8,
#                                 tolerance_change=1e-11)
#     elif optimizer_type == 'NNAT_LBFGS':
#         optimizer = NNAT_LBFGS([norm_params], lr=learning_rate, device='cpu')
#     else:
#         warnings.warn(f"The optimizer type {optimizer_type} is not supported.")
#         sys.exit("Exiting the script due to unsupported optimizer type.")
#
#     prev_cost = None
#     for i in range(1, iterations + 1):
#         def closure():
#             if torch.is_grad_enabled():
#                 optimizer.zero_grad()
#             cost = 0
#             norm_src_voltage, norm_fltr_th, norm_scint_th = split_list(norm_params, length)
#             src_voltage = min_max_denormalize_scalar(norm_src_voltage,
#                                                      src_vol_lower_range_list,
#                                                      src_vol_upper_range_list)
#             fltr_th = min_max_denormalize_scalar(norm_fltr_th,
#                                                  fltr_th_bound[0],
#                                                  fltr_th_bound[1])
#             scint_th = min_max_denormalize_scalar(norm_scint_th,
#                                                   scint_th_bound[0],
#                                                   scint_th_bound[1])
#             for signal_train, sv, spec_F_train in zip(signal_train_list, src_voltage, spec_F_train_list):
#                 src_response = interp_src_spectra(src_kV_list, src_spec_list, sv)
#
#                 cost += anal_cost(energies, signal_train, spec_F_train, src_response,
#                                   fltr_mat,
#                                   scint_mat,
#                                   fltr_th,
#                                   scint_th, signal_weight=[1.0 / sig for sig in signal_train])
#             if cost.requires_grad and optimizer_type != 'NNAT_LBFGS':
#                 cost.backward()
#             return cost
#
#         cost = closure()
#         if optimizer_type == 'NNAT_LBFGS':
#             cost.backward()
#
#         with (torch.no_grad()):
#             if i == 1:
#                 print('Initial cost: %e' % (closure().item()))
#
#         if optimizer_type == 'Adam':
#             # cost = closure()
#             optimizer.step()
#         elif optimizer_type == 'LBFGS':
#             optimizer.step(closure)
#         elif optimizer_type == 'NNAT_LBFGS':
#             options = {'closure': closure, 'current_loss': cost,
#                        'max_ls': 100, 'damping': True}
#             cost, grad_new, _, _, closures_new, grads_new, desc_dir, fail = optimizer.step(
#                 options=options)
#
#         with (torch.no_grad()):
#             if i % 2 == 0:
#                 print(
#                     'Iteration:{0}: before update cost: {1:e}, source voltage: {2} filter {3} thickness: {4:e}, scintillator {5} thickness: {6:e}'
#                     .format(i, closure().item(), [sv.item() for sv in src_voltage], fltr_mat['formula'], fltr_th.item(),
#                             scint_mat['formula'],
#                             scint_th.item()))
#
#             # Project the updated x back onto the feasible set
#
#             norm_params.data = project_onto_constraints(norm_params.data,
#                                                         init_lower_bounds,
#                                                         init_upper_bounds)
#
#             norm_src_voltage, norm_fltr_th, norm_scint_th = split_list(norm_params, length)
#
#             src_voltage = min_max_denormalize_scalar(norm_src_voltage,
#                                                      src_vol_lower_range_list,
#                                                      src_vol_upper_range_list)
#             fltr_th = min_max_denormalize_scalar(norm_fltr_th,
#                                                  fltr_th_bound[0],
#                                                  fltr_th_bound[1])
#             scint_th = min_max_denormalize_scalar(norm_scint_th,
#                                                   scint_th_bound[0],
#                                                   scint_th_bound[1])
#
#             # Check the stopping criterion based on changes in x and y
#             if prev_cost is not None and \
#                     torch.abs(closure() - prev_cost) / prev_cost < tolerance and \
#                     torch.mean(torch.abs(src_voltage - prev_src_voltage) / prev_src_voltage) < tolerance and \
#                     torch.abs(fltr_th - prev_fltr_th) < tolerance and \
#                     torch.abs(scint_th - prev_scint_th) / prev_scint_th < tolerance:
#                 print(f"Stopping after {i} iterations")
#                 break
#
#             prev_cost = closure().item()
#             prev_src_voltage = torch.tensor([sv.item() for sv in src_voltage])
#             prev_fltr_th = fltr_th.item()
#             prev_scint_th = scint_th.item()
#
#             # Clear gradients and update previous values for the next iteration
#             if return_history:
#                 cost_list.append(closure().item())
#                 src_voltage_list.append([sv.item() for sv in src_voltage])
#                 fltr_th_list.append(fltr_th.item())
#                 scint_th_list.append(scint_th.item())
#
#     print(
#         f"The minimum cost value: {closure().item()}, occurs at source voltage = {[sv.item() for sv in src_voltage]} kV, "
#         f"filter thickness = {fltr_th.item()} mm, scintillator thickness = {scint_th.item()} mm")
#
#     if return_history:
#         return cost_list, src_voltage_list, fltr_th_list, scint_th_list
#     else:
#         return [sv.item() for sv in src_voltage], fltr_th.item(), scint_th.item(), closure().item()
#
#
# def parallel_anal_sep_model(num_processes,
#                             fltr_mat_list, scint_mat_list,
#                             init_fltr_th_values, init_scint_th_values, *args, **kwargs):
#     # Create parameter combinations
#     params_combinations = product(fltr_mat_list, scint_mat_list, init_fltr_th_values,
#                                   init_scint_th_values)
#
#     with Pool(processes=num_processes) as pool:
#         result_objects = [
#             pool.apply_async(
#                 anal_sep_model,
#                 args=args,
#                 kwds={
#                     **kwargs,
#                     "fltr_mat": fltr_mat,
#                     "scint_mat": scint_mat,
#                     "init_fltr_th": fltr_th,
#                     "init_scint_th": scint_th
#                 }
#             )
#             for fltr_mat, scint_mat, fltr_th, scint_th in params_combinations
#         ]
#
#         # Gather results
#         print('result_objects', result_objects)
#         results = [r.get() for r in result_objects]
#
#     return results
#


class src_spec_model(torch.nn.Module):
    def __init__(self, src_config: src_spec_params, device=None, dtype=None)-> None:
        """

        Parameters
        ----------
        src_config: src_spec_params
            Source model configuration.
        device
        dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.src_config = src_config

        # Min-Max Normalization
        self.lower = src_config.src_vol_bound.lower
        self.scale = src_config.src_vol_bound.upper-src_config.src_vol_bound.lower
        normalized_voltage = (src_config.voltage-self.lower)/self.scale
        # Instantiate parameters
        self.normalized_voltage = Parameter(torch.tensor(normalized_voltage, **factory_kwargs))

    def get_voltage(self):
        """Read voltage.

        Returns
        -------
        voltage: float
            Read voltage.
        """

        return self.normalized_voltage*self.scale+self.lower

    def forward(self):
        """Calculate source spectrum.

        Returns
        -------
        src_spec: torch.Tensor
            Source spectrum.

        """

        return interp_src_spectra(self.src_config.src_vol_list, self.src_config.src_spec_list, self.get_voltage())


class fltr_resp_model(torch.nn.Module):
    def __init__(self, fltr_config: fltr_resp_params, device=None, dtype=None)-> None:
        """Filter module

        Parameters
        ----------
        fltr_config: fltr_resp_params
            Filter model configuration.
        device
        dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.fltr_config = fltr_config

        # Min-Max Normalization
        self.lower = [fltr_config.fltr_th_bound[i].lower for i in range(fltr_config.num_fltr)]
        self.scale = [fltr_config.fltr_th_bound[i].upper-fltr_config.fltr_th_bound[i].lower for i in range(fltr_config.num_fltr)]
        normalized_fltr_th = [(fltr_config.fltr_th[i] - self.lower[i]) / self.scale[i] for i in range(fltr_config.num_fltr)]
        # Instantiate parameters
        self.normalized_fltr_th = torch.nn.ParameterList([Parameter(torch.tensor(normalized_fltr_th[i], **factory_kwargs)) for i in range(fltr_config.num_fltr)])

    def get_fltr_th(self):
        """Get filter thickness.

        Returns
        -------
        fltr_th_list: list
            List of filter thickness. Length is equal to num_fltr.

        """
        return [self.normalized_fltr_th[i]*self.scale[i]+self.lower[i]
                             for i in range(self.fltr_config.num_fltr)]

    def forward(self, energies):
        """Calculate filter response.

        Parameters
        ----------
        energies : numpy.ndarray
            List of X-ray energies of a poly-energetic source in units of keV.

        Returns
        -------
        fltr_resp: torch.Tensor
            Filter response.

        """

        return gen_fltr_res(energies, self.fltr_config.fltr_mat, self.get_fltr_th())

class scint_cvt_model(torch.nn.Module):
    def __init__(self, scint_config: scint_cvt_func_params, device=None, dtype=None)-> None:
        """Scintillator convertion model

        Parameters
        ----------
        scint_config: scint_cvt_func_params
            Sinctillator model configuration.
        device
        dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.scint_config = scint_config

        # Min-Max Normalization
        self.lower = scint_config.scint_th_bound.lower
        self.scale = scint_config.scint_th_bound.upper-scint_config.scint_th_bound.lower
        normalized_scint_th = (scint_config.scint_th-self.lower)/self.scale
        # Instantiate parameter
        self.normalized_scint_th = Parameter(torch.tensor(normalized_scint_th, **factory_kwargs))

    def get_scint_th(self):
        """

        Returns
        -------
        scint_th: float
            Sintillator thickness.

        """
        return self.normalized_scint_th*self.scale+self.lower

    def forward(self, energies):
        """Calculate scintillator convertion function.

        Parameters
        ----------
        energies: numpy.ndarray
            List of X-ray energies of a poly-energetic source in units of keV.

        Returns
        -------
        scint_cvt_func: torch.Tensor
            Scintillator convertion function.

        """

        return gen_scint_cvt_func(energies, self.scint_config.scint_mat, self.get_scint_th())

class spec_distrb_energy_resp(torch.nn.Module):
    def __init__(self,
                 energies,
                 src_config_list: [src_spec_params],
                 fltr_config_list: [fltr_resp_params],
                 scint_config_list: [scint_cvt_func_params], device=None, dtype=None):
        """Total spectrally distributed energy response model.

        Parameters
        ----------
        energies: numpy.ndarray
            List of X-ray energies of a poly-energetic source in units of keV.
        src_config_list: list of src_spec_params
            List of source model configurations.
        fltr_config_list: list of fltr_resp_params
            List of filter model configurations.
        scint_config_list: list of scint_cvt_func_params
            List of scintillator model configurations.
        device
        dtype
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.energies = torch.Tensor(energies) if energies is not torch.Tensor else energies
        self.src_spec_list = torch.nn.ModuleList([src_spec_model(src_config, **factory_kwargs) for src_config in src_config_list])
        self.fltr_resp_list = torch.nn.ModuleList([fltr_resp_model(fltr_config, **factory_kwargs) for fltr_config in fltr_config_list])
        self.scint_cvt_list = torch.nn.ModuleList([scint_cvt_model(scint_config, **factory_kwargs) for scint_config in scint_config_list])

    def forward(self, F:torch.Tensor, mc: Model_combination):
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
        src_func = self.src_spec_list[mc.src_ind]()
        fltr_func = self.fltr_resp_list[mc.fltr_ind](self.energies)
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
        for name, param in self.named_parameters():
            print(f"Name: {name} | Size: {param.size()} | Values : {param.data} | Requires Grad: {param.requires_grad}")

    def print_ori_parameters(self):
        """
        Print all scaled-back parameters of the model.
        """
        for src_i, src_spec in enumerate(self.src_spec_list):
            print('Voltage %d:'%(src_i), src_spec.get_voltage())

        for fltr_i, fltr_resp in enumerate(self.fltr_resp_list):
            print('Filter Thickness %d:'%(fltr_i), fltr_resp.get_fltr_th())

        for scint_i, scint_cvt in enumerate(self.scint_cvt_list):
            print('Scintillator Thickness %d:'%(scint_i), scint_cvt.get_scint_th())
def weighted_mse_loss(input, target, weight):
    return 0.5*torch.mean(weight * (input - target) ** 2)

def param_based_spec_estimate(energies,
                              y,
                              F,
                              src_config:[src_spec_params],
                              fltr_config:[fltr_resp_params],
                              scint_config:[scint_cvt_func_params],
                              model_combination:[Model_combination],
                              learning_rate=0.1,
                              iterations=5000,
                              tolerance=1e-6,
                              optimizer_type='Adam',
                              return_history=False):
    """

    Parameters
    ----------
    energies : numpy.ndarray
        List of X-ray energies of a poly-energetic source in units of keV.
    y : list
        Transmission Data :math:`y`.  (#datasets, #samples, #views, #rows, #columns).
        Should be the exponential term instead of the projection after taking negative log.
    F : list
        Forward matrix. (#datasets, #samples, #views, #rows, #columns, #energy_bins)
    src_config : list of src_spec_params
    fltr_config : list of fltr_resp_params
    scint_config : list of src_spec_params
    model_combination : list of Model_combination
    learning_rate : int
    iterations : int
    tolerance : float
        Stop when all parameters update is less than tolerance.
    optimizer_type : str
        'Adam' or 'NNAT_LBFGS'
    return_history : bool
        Save history of parameters.

    Returns
    -------

    """

    # Check Variables
    if return_history:
        src_voltage_list = []
        fltr_th_list = []
        scint_th_list = []
        cost_list = []

    # Construct our model by instantiating the class defined above
    model = spec_distrb_energy_resp(energies, src_config, fltr_config, scint_config, device='cpu', dtype=torch.float32)
    model.print_parameters()

    loss = torch.nn.MSELoss()
    if optimizer_type == 'Adam':
        iter_prt = 50
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'NNAT_LBFGS':
        iter_prt = 10
        optimizer = NNAT_LBFGS(model.parameters(), lr=learning_rate, device='cpu')
    else:
        warnings.warn(f"The optimizer type {optimizer_type} is not supported.")
        sys.exit("Exiting the script due to unsupported optimizer type.")
    y = [torch.tensor(np.concatenate([sig.reshape((-1, 1)) for sig in yy]), dtype=torch.float32) for yy in y]
    weights = [1.0/yy for yy in y]
    F = [torch.tensor(FF, dtype=torch.float32) for FF in F]
    for iter in range(1, iterations + 1):
        if iter % iter_prt == 0:
            print('Iteration:', iter)
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            cost = 0
            for yy, FF, ww, mc in zip(y, F, weights, model_combination):
                trans_val = model(FF, mc)
                # print(trans_val.shape)
                # print(yy.shape)
                # print(ww.shape)
                # sub_cost = loss(trans_val, yy)
                #sub_cost = weighted_mse_loss(trans_val, yy, ww)
                sub_cost = loss(-torch.log(trans_val), -torch.log(yy))
                cost += sub_cost
            if cost.requires_grad and optimizer_type != 'NNAT_LBFGS':
                cost.backward()
            return cost

        cost = closure()
        if optimizer_type == 'NNAT_LBFGS':
            cost.backward()

        with (torch.no_grad()):
            if iter == 1:
                print('Initial cost: %e' % (closure().item()))

        # Before the update, clone the current parameters
        old_params = {k: v.clone() for k, v in model.state_dict().items()}

        if optimizer_type == 'Adam':
            optimizer.step()
        elif optimizer_type == 'NNAT_LBFGS':
            options = {'closure': closure, 'current_loss': cost,
                       'max_ls': 10, 'damping': True}
            cost, grad_new, _, _, closures_new, grads_new, desc_dir, fail = optimizer.step(
                options=options)

        with (torch.no_grad()):
            # Clamp all parameters to be between 0 and 1
            for param in model.parameters():
                param.data.clamp_(0, 1)

            if iter % iter_prt == 0:
                # model.print_parameters()
                print('Cost:', cost.item())
                model.print_ori_parameters()

            # After the update, check if the update is too small
            small_update = True
            for k, v in model.state_dict().items():
                if torch.norm(v - old_params[k]) > tolerance:
                    small_update = False
                    break

            if small_update:
                print(f"Stopping at epoch {iter} because updates are too small.")
                print('Cost:', cost.item())
                model.print_ori_parameters()
                break
    return 0
