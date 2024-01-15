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
from xspec.chem_consts._periodictabledata import atom_weights, density,ptableinverse
from xspec.opt._pytorch_lbfgs.functions.LBFGS import FullBatchLBFGS as NNAT_LBFGS
from itertools import product

class ClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
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
    Z = 74 # Tungsten
    target_material = ptableinverse[Z]
    PhilibertConstant = 4.0e5
    PhilibertExponent = 1.65
    #sin_psi = torch.sin(takeOffAngle * torch.pi / 180.0)
    h_local = 1.2 * atom_weights[target_material] / (Z**2)
    h_factor = h_local / (1.0 + h_local)

    kVp_e165 = voltage ** PhilibertExponent
    kappa = torch.zeros((energies.shape))
    if not isinstance(energies, torch.Tensor):
        energies = torch.tensor(energies)
    kappa[:-1] = (PhilibertConstant / (kVp_e165 - energies ** PhilibertExponent)[:-1])
    kappa[-1] = np.inf
    mu = torch.tensor(get_mass_absp_c_vs_E(ptableinverse[Z], energies)) # cm^-1

    return (1+mu/kappa/sin_psi)**-1*(1+h_factor*mu/kappa/sin_psi)**-1

def takeoff_angle_conversion_factor(voltage, sin_psi_cur, sin_psi_new, energies):
    # Assuming takeOffAngle_cur is already defined
    if not isinstance(sin_psi_cur, torch.Tensor):
        sin_psi_cur = torch.tensor(sin_psi_cur)
    if not isinstance(sin_psi_new, torch.Tensor):
        sin_psi_new = torch.tensor(sin_psi_new)
    return philibert_absorption_correction_factor(voltage, sin_psi_new, energies)/philibert_absorption_correction_factor(voltage, sin_psi_cur, energies)


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
        self.toa_lower = angle_sin(source.takeoff_angle_bound.lower)
        self.toa_scale = angle_sin(source.takeoff_angle_bound.upper) - angle_sin(source.takeoff_angle_bound.lower)
        normalized_sin_psi = (angle_sin(source.takeoff_angle) - self.toa_lower) / self.toa_scale

        # Instantiate parameters
        if source.optimize_voltage:
            self.normalized_voltage = Parameter(torch.tensor(normalized_voltage, **factory_kwargs))
        else:
            self.normalized_voltage = torch.tensor(normalized_voltage, **factory_kwargs)

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

        return np.arcsin(clamp_with_grad(self.normalized_sin_psi, 0, 1).detach().numpy() * self.toa_scale + self.toa_lower)*180.0/np.pi

    def get_sin_psi(self):
        """Read takeoff_angle.

        Returns
        -------
        voltage: float
            Read takeoff_angle.
        """

        return clamp_with_grad(self.normalized_sin_psi, 0, 1) * self.toa_scale + self.toa_lower
    def forward(self, energies):
        """Calculate source spectrum.

        Returns
        -------
        src_spec: torch.Tensor
            Source spectrum.

        """
        src_spec = interp_src_spectra(self.source.src_voltage_list, self.source.src_spec_list, self.get_voltage())
        sin_psi_cur = angle_sin(self.source.takeoff_angle_cur)
        src_spec = src_spec * takeoff_angle_conversion_factor(self.get_voltage(), sin_psi_cur, self.get_sin_psi(), energies)
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
        fltr_mat: Material
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
        self.ot =optimizer_type
        self.energies = torch.Tensor(energies) if energies is not torch.Tensor else energies
        self.src_spec_list = torch.nn.ModuleList(
            [Source_Model(source, **factory_kwargs) for source in sources])
        if sources[0].optimize_takeoff_angle:
            for smm in self.src_spec_list[1:]:
                smm._parameters['normalized_sin_psi'] = self.src_spec_list[0]._parameters['normalized_sin_psi']
        self.fltr_resp_list = torch.nn.ModuleList(
        [Filter_Model(filter, **factory_kwargs) for filter in filters])
        self.scint_cvt_list = torch.nn.ModuleList(
            [Scintillator_Model(scintillator, **factory_kwargs) for scintillator in scintillators])
        self.logger = logging.getLogger(str(mp.current_process().pid))

    def print_method(self,*args, **kwargs):
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
                if self.src_spec_list[mc.src_ind].source.optimize_takeoff_angle:
                    self.src_spec_list[mc.src_ind]._parameters['normalized_sin_psi'].data.clamp_(min=1e-6, max=1-1e-6)
                for fii in mc.fltr_ind_list:
                     self.fltr_resp_list[fii]._parameters['normalized_fltr_th'].data.clamp_(min=1e-6, max=1-1e-6)
                self.scint_cvt_list[mc.scint_ind]._parameters['normalized_scint_th'].data.clamp_(min=1e-6, max=1-1e-6)
            
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
            print('Source %d: Voltage: %.2f; Take-off Angle: %.2f' % (src_i, src_spec.get_voltage().item(), src_spec.get_takeoff_angle().item()))

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
        ot ='Adam'
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
    
    print('Initial optimizer:', optimizer)
    print('Initial optimizer_type:', model.ot)
        
    y = [torch.tensor(np.concatenate([sig.reshape((-1, 1)) for sig in yy]), dtype=torch.float32) for yy in y]
    num_sp_datasets=len(y)
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
            if cost<=0.6*num_sp_datasets or iter>5000:
          
                if optimizer != optimizerN:
                    print('Start use NNAT_LBFGS')
                    print('Current cost value:',cost.item())
                    iter_prt = 5
                    ot = 'NNAT_LBFGS'
                    optimizer = optimizerN
                    model.update_optimizer_type(ot)
                else:
                    LBFGS_iter+=1
                    
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            cost = 0
            for yy, FF, ww, mc in zip(y, F, weight, model_combination):
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
                if torch.norm(v.clamp(0,1) - old_params[k].clamp(0,1)) > stop_threshold:
                    small_update = False
                    break

            if small_update or LBFGS_iter>max_iterations-5000:
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
        handler = logging.FileHandler(f"{filename}_{worker_id%num_processes}.log")

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
    possible_scintilators_combinations = [[sc for sc in scm.next_psb_scint_mat()] for scm in scintillators]

    # Combine possible filters and scintillators
    model_params_list = list(product(*possible_filters_combinations, *possible_scintilators_combinations))
    # Regroup filters and scintillators
    model_params_list = [nested_list(model_params, [len(d) for d in [possible_filters_combinations,
                                                          possible_scintilators_combinations]]) for model_params in
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
        print('Source %d: Voltage: %.2f; Take-off Angle: %.2f' % (src_i, src_spec.get_voltage().item(), src_spec.get_takeoff_angle().item()))

    for fltr_i, fltr_resp in enumerate(best_res.fltr_resp_list):
        print(
            f'Filter {fltr_i}: Material: {fltr_resp.get_fltr_mat()}, Thickness: {fltr_resp.get_fltr_th()}')

    for scint_i, scint_cvt in enumerate(best_res.scint_cvt_list):
        print(f'Scintillator {scint_i}: Material:{scint_cvt.get_scint_mat()} Thickness:{scint_cvt.get_scint_th()}')
    return results
