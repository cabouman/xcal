import sys
import warnings
import numpy as np

import torch
import torch.optim as optim
from torch.nn.parameter import Parameter

from torch.multiprocessing import Pool
import torch.multiprocessing as mp
import logging

from xspec._utils import *
from xspec.defs import *
from xspec.dict_gen import gen_fltr_res, gen_scint_cvt_func
from xspec.opt._pytorch_lbfgs.functions.LBFGS import FullBatchLBFGS as NNAT_LBFGS
from itertools import product



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



class src_spec_model(torch.nn.Module):
    def __init__(self, src_config: src_spec_params, device=None, dtype=None) -> None:
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
        self.scale = src_config.src_vol_bound.upper - src_config.src_vol_bound.lower
        normalized_voltage = (src_config.voltage - self.lower) / self.scale
        # Instantiate parameters
        if src_config.require_gradient:
            self.normalized_voltage = Parameter(torch.tensor(normalized_voltage, **factory_kwargs))
        else:
            self.normalized_voltage = torch.tensor(normalized_voltage, **factory_kwargs)

    def get_voltage(self):
        """Read voltage.

        Returns
        -------
        voltage: float
            Read voltage.
        """

        return torch.clamp(self.normalized_voltage, 0, 1) * self.scale + self.lower

    def forward(self):
        """Calculate source spectrum.

        Returns
        -------
        src_spec: torch.Tensor
            Source spectrum.

        """

        return interp_src_spectra(self.src_config.src_vol_list, self.src_config.src_spec_list, self.get_voltage())


class fltr_resp_model(torch.nn.Module):
    def __init__(self, fltr_config: fltr_resp_params, device=None, dtype=None) -> None:
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
        self.lower = fltr_config.fltr_th_bound.lower
        self.scale = fltr_config.fltr_th_bound.upper - fltr_config.fltr_th_bound.lower
        normalized_fltr_th = (fltr_config.fltr_th - self.lower) / self.scale
        # Instantiate parameters
        if fltr_config.require_gradient:
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
        return torch.clamp(self.normalized_fltr_th, 0, 1) * self.scale + self.lower


    def get_fltr_mat(self):
        """Get filter thickness.

        Returns
        -------
        fltr_mat: Material
            Filter material.

        """
        return self.fltr_config.fltr_mat

    def forward(self, energies):
        """Calculate filter response.

        Parameters
        ----------
        energies : numpy.ndarray
            List of X-ray energies of a poly-energetic source in units of keV.

        fltr_ind_list: list of int

        Returns
        -------
        fltr_resp: torch.Tensor
            Filter response.

        """

        return gen_fltr_res(energies, self.fltr_config.fltr_mat, self.get_fltr_th())


class scint_cvt_model(torch.nn.Module):
    def __init__(self, scint_config: scint_cvt_func_params, device=None, dtype=None) -> None:
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
        self.scale = scint_config.scint_th_bound.upper - scint_config.scint_th_bound.lower
        normalized_scint_th = (scint_config.scint_th - self.lower) / self.scale
        # Instantiate parameter
        if scint_config.require_gradient:
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
        return torch.clamp(self.normalized_scint_th, 0, 1) * self.scale + self.lower

    def get_scint_mat(self):
        """

        Returns
        -------
        scint_th: float
            Sintillator thickness.

        """
        return self.scint_config.scint_mat

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
        """Total spectrally distributed energy response model based on torch.nn.Module.

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
        self.src_spec_list = torch.nn.ModuleList(
            [src_spec_model(src_config, **factory_kwargs) for src_config in src_config_list])
        self.fltr_resp_list = torch.nn.ModuleList(
            [fltr_resp_model(fltr_config, **factory_kwargs) for fltr_config in fltr_config_list])
        self.scint_cvt_list = torch.nn.ModuleList(
            [scint_cvt_model(scint_config, **factory_kwargs) for scint_config in scint_config_list])
        self.logger = logging.getLogger(str(mp.current_process().pid))

    def print_method(self,*args, **kwargs):
        message = ' '.join(map(str, args))
        self.logger.info(message)

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
        src_func = self.src_spec_list[mc.src_ind]()
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
            print('Voltage %d:' % (src_i), src_spec.get_voltage().item())

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
                                   src_config: [src_spec_params],
                                   fltr_config: [fltr_resp_params],
                                   scint_config: [scint_cvt_func_params],
                                   model_combination: [Model_combination],
                                   learning_rate=0.02,
                                   max_iterations=5000,
                                   stop_threshold=1e-3,
                                   optimizer_type='NNAT_LBFGS',
                                   loss_type='wmse',
                                   return_history=False):
    """Other arguments are same as param_based_spec_estimate.

    Parameters
    ----------
    fltr_config : list of fltr_resp_params
        Each fltr_resp_params.fltr_mat should be specified to a Material instead of None.
    scint_config
        Each scint_cvt_func_params.scint_mat should be specified to a Material instead of None.

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
    model = spec_distrb_energy_resp(energies, src_config, fltr_config, scint_config, device='cpu', dtype=torch.float32)
    model.print_parameters()

    loss = torch.nn.MSELoss()
    if optimizer_type == 'Adam':
        iter_prt = 50
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'NNAT_LBFGS':
        iter_prt = 10
        optimizer = NNAT_LBFGS(model.parameters(), lr=learning_rate)
    else:
        warnings.warn(f"The optimizer type {optimizer_type} is not supported.")
        sys.exit("Exiting the script due to unsupported optimizer type.")
    y = [torch.tensor(np.concatenate([sig.reshape((-1, 1)) for sig in yy]), dtype=torch.float32) for yy in y]
    weights = [1.0 / yy for yy in y]
    F = [torch.tensor(FF, dtype=torch.float32) for FF in F]
    for iter in range(1, max_iterations + 1):
        if iter % iter_prt == 0:
            print('Iteration:', iter)

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            cost = 0
            for yy, FF, ww, mc in zip(y, F, weights, model_combination):
                trans_val = model(FF, mc)
                if loss_type == 'mse':
                    sub_cost = 0.5*loss(trans_val, yy)
                elif loss_type == 'wmse':
                    sub_cost = weighted_mse_loss(trans_val, yy, ww)
                elif loss_type == 'attmse':
                    sub_cost = 0.5*loss(-torch.log(trans_val), -torch.log(yy))
                else:
                    raise ValueError('loss_type should be \'mse\' or \'wmse\' or \'attmse\'. ','Given', loss_type)
                cost += sub_cost
            if cost.requires_grad and optimizer_type != 'NNAT_LBFGS':
                cost.backward()
            return cost

        cost = closure()
        if torch.isnan(cost):
            model.print_parameters()
            return iter, closure().item(), model
        if optimizer_type == 'NNAT_LBFGS':
            cost.backward()

        has_nan = check_gradients_for_nan(model)
        if has_nan:
            return iter, closure().item(), model

        with (torch.no_grad()):
            if iter == 1:
                print('Initial cost: %e' % (closure().item()))

        # Before the update, clone the current parameters
        old_params = {k: v.clone() for k, v in model.state_dict().items()}

        if optimizer_type == 'Adam':
            optimizer.step()
        elif optimizer_type == 'NNAT_LBFGS':
            options = {'closure': closure, 'current_loss': cost,
                       'max_ls': 200, 'damping': False}
            cost, grad_new, _, _, closures_new, grads_new, desc_dir, fail = optimizer.step(
                options=options)

        with (torch.no_grad()):
            if iter % iter_prt == 0:
                print('Cost:', cost.item())
                model.print_ori_parameters()

            # After the update, check if the update is too small
            small_update = True
            for k, v in model.state_dict().items():
                if torch.norm(v - old_params[k]) > stop_threshold:
                    small_update = False
                    break

            if small_update:
                print(f"Stopping at epoch {iter} because updates are too small.")
                print('Cost:', cost.item())
                model.print_ori_parameters()
                break
    return iter, cost.item(), model


def init_logging(filename):
    worker_id = mp.current_process().pid
    logger = logging.getLogger(str(worker_id))
    logger.setLevel(logging.INFO)

    if filename is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(f"{filename}_{worker_id}.log")

    formatter = logging.Formatter('%(asctime)s  - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def param_based_spec_estimate(energies,
                              y,
                              F,
                              src_config: [src_spec_params],
                              Fltr_config: [fltr_resp_params],
                              Scint_config: [scint_cvt_func_params],
                              model_combination: [Model_combination],
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
    energies : numpy.ndarray
        List of X-ray energies of a poly-energetic source in units of keV.
    y : list
        Transmission Data :math:`y`.  (#datasets, #samples, #views, #rows, #columns).
        Normalized transmission data, background should be close to 1.
    F : list
        Forward matrix. (#datasets, #samples, #views, #rows, #columns, #energy_bins)
    src_config : list of src_spec_params
        Specify all sources used across datasets.
    Fltr_config : list of fltr_resp_params
        Specify all filters used across datasets. For each filter, Fltr_config provides possible material list instead of specific material.
        The function will find out the best filter material among fltr_resp_params.psb_fltr_mat.
    Scint_config : list of scint_cvt_func_params
        Specify all scintillators used across datasets. For each scintillator, Scint_config provides possible material list instead of specific material.
        The function will find out the best scintillator material among scint_cvt_func_params.psb_scint_mat.
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

    fltr_config_list = [[fc for fc in fcm.next_psb_fltr_mat()] for fcm in Fltr_config]
    scint_config_lsit = [[sc for sc in scm.next_psb_scint_mat()] for scm in Scint_config]
    model_params_list = list(product(*fltr_config_list, *scint_config_lsit))
    model_params_list = [nested_list(l, [len(d) for d in [fltr_config_list, scint_config_lsit]]) for l in
                         model_params_list]

    with Pool(processes=num_processes, initializer=init_logging, initargs=(logpath,)) as pool:
        result_objects = [
            pool.apply_async(
                param_based_spec_estimate_cell,
                args=args[:4] + (fltr_config, scint_config,) + args[6:]
            )
            for fltr_config, scint_config in model_params_list
        ]

        # Gather results
        print('result_objects', result_objects)
        results = [r.get() for r in result_objects]

    return results
