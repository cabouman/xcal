import sys
import warnings
import numpy as np

import torch
import torch.optim as optim
from torch.nn.parameter import Parameter

from multiprocessing import Pool

from xspec._utils import is_sorted, min_max_normalize_scalar, min_max_denormalize_scalar, concatenate_items, split_list
from xspec._utils import Bound, Material
from xspec.opt._pytorch_lbfgs.functions.LBFGS import FullBatchLBFGS as NNAT_LBFGS
from itertools import product


def anal_cost(energies, y, F, src_response, fltr_mat, scint_mat, th_fl, th_sc, signal_weight=None, torch_mode=True):
    signal = np.concatenate([sig.reshape((-1, 1)) for sig in y])
    forward_mat = np.concatenate([fwm.reshape((-1, fwm.shape[-1])) for fwm in F])
    if signal_weight is None:
        signal_weight = np.ones(signal.shape)
    signal_weight = np.concatenate([sig.reshape((-1, 1)) for sig in signal_weight])

    if torch_mode:
        if not isinstance(src_response, torch.Tensor):
            src_response = torch.tensor(src_response)
        signal = torch.tensor(signal)
        forward_mat = torch.tensor(forward_mat)
        signal_weight = torch.tensor(signal_weight)

    # Calculate filter response
    fltr_params = [
        {'formula': fltr_mat['formula'], 'density': fltr_mat['density'], 'thickness_list': [th_fl]},
    ]
    fltr_response, fltr_info = gen_filts_specD(energies, composition=fltr_params, torch_mode=torch_mode)

    # Calculate scintillator response
    scint_params = [
        {'formula': scint_mat['formula'], 'density': scint_mat['density'], 'thickness_list': [th_sc]},
    ]
    scint_response, scint_info = gen_scints_specD(energies, composition=scint_params, torch_mode=torch_mode)

    # Calculate total system response as a product of source, filter, and scintillator responses.
    sys_response = (src_response * fltr_response * scint_response).T

    if torch_mode:
        Fx = torch.trapz(forward_mat * sys_response.T, torch.tensor(energies), axis=1).reshape((-1, 1))
        e = signal - Fx / torch.trapz(sys_response, torch.tensor(energies), axis=0)
    else:
        Fx = np.trapz(forward_mat * sys_response.T, energies, axis=1).reshape((-1, 1))
        e = signal - Fx / np.trapz(sys_response, energies, axis=0)

    return cal_cost(e, signal_weight, torch_mode)


def project_onto_constraints(x, lower_bound, upper_bound):
    # return torch.clamp(x, min=lower_bound, max=upper_bound)
    return torch.clamp(x, min=torch.tensor(lower_bound), max=torch.tensor(upper_bound))


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


def anal_sep_model(energies, signal_train_list, spec_F_train_list, src_response_dict=None, fltr_mat=None,
                   scint_mat=None,
                   init_src_vol=[50.0], init_fltr_th=1.0, init_scint_th=0.1, src_vol_bound=None, fltr_th_bound=(0, 10),
                   scint_th_bound=(0.01, 1),
                   learning_rate=0.1, iterations=5000, tolerance=1e-6, optimizer_type='Adam', return_history=False):
    """

    Parameters
    ----------
    energies
    signal_train_list
    spec_F_train_list
    src_response_dict
    fltr_mat
    scint_mat
    init_src_vol
    init_fltr_th
    init_scint_th
    src_vol_bound
    fltr_th_bound
    scint_th_bound
    learning_rate
    iterations
    tolerance
    optimizer_type
    return_history

    Returns
    -------

    """

    if return_history:
        src_voltage_list = []
        fltr_th_list = []
        scint_th_list = []
        cost_list = []

    torch.autograd.set_detect_anomaly(True)

    src_spec_list = [ssl['spectrum'] for ssl in src_response_dict]
    src_kV_list = [ssl['source_voltage'] for ssl in src_response_dict]

    if not is_sorted(src_kV_list):
        raise ValueError("Warning: source voltage in src_response_dict are not sorted!")

    # Sorted list of values
    src_spec_list = torch.tensor(src_spec_list, dtype=torch.float32)
    src_kV_list = torch.tensor(src_kV_list, dtype=torch.int32)

    if 'thickness_bound' in fltr_mat:
        if fltr_mat['thickness_bound'] is not None:
            fltr_th_bound = fltr_mat['thickness_bound']
    if 'thickness_bound' in scint_mat:
        if scint_mat['thickness_bound'] is not None:
            scint_th_bound = scint_mat['thickness_bound']

    src_vol_range = [(src_kV_list[0], src_kV_list[-1]) for sv in init_src_vol]
    src_vol_lower_range_list = [lb for lb, _ in src_vol_range]
    src_vol_upper_range_list = [ub for _, ub in src_vol_range]

    if src_vol_bound is None:
        src_vol_bound = src_vol_range
    src_vol_lower_bound_list = min_max_normalize_scalar([lb for lb, _ in src_vol_bound],
                                                        src_vol_lower_range_list,
                                                        src_vol_upper_range_list).tolist()
    src_vol_upper_bound_list = min_max_normalize_scalar([ub for _, ub in src_vol_bound],
                                                        src_vol_lower_range_list,
                                                        src_vol_upper_range_list).tolist()

    init_params, length = concatenate_items(init_src_vol, init_fltr_th, init_scint_th)
    init_lower_ranges, _ = concatenate_items(src_vol_lower_range_list, fltr_th_bound[0], scint_th_bound[0])
    init_upper_ranges, _ = concatenate_items(src_vol_upper_range_list, fltr_th_bound[1], scint_th_bound[1])
    init_lower_bounds, _ = concatenate_items(src_vol_lower_bound_list, 0, 0)
    init_upper_bounds, _ = concatenate_items(src_vol_upper_bound_list, 1, 1)
    norm_params = torch.tensor(min_max_normalize_scalar(init_params,
                                                        init_lower_ranges,
                                                        init_upper_ranges), requires_grad=True)

    if optimizer_type == 'Adam':
        optimizer = optim.Adam([norm_params], lr=learning_rate)
    elif optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS([norm_params], lr=learning_rate, max_iter=100, tolerance_grad=1e-8,
                                tolerance_change=1e-11)
    elif optimizer_type == 'NNAT_LBFGS':
        optimizer = NNAT_LBFGS([norm_params], lr=learning_rate, device='cpu')
    else:
        warnings.warn(f"The optimizer type {optimizer_type} is not supported.")
        sys.exit("Exiting the script due to unsupported optimizer type.")

    prev_cost = None
    for i in range(1, iterations + 1):
        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            cost = 0
            norm_src_voltage, norm_fltr_th, norm_scint_th = split_list(norm_params, length)
            src_voltage = min_max_denormalize_scalar(norm_src_voltage,
                                                     src_vol_lower_range_list,
                                                     src_vol_upper_range_list)
            fltr_th = min_max_denormalize_scalar(norm_fltr_th,
                                                 fltr_th_bound[0],
                                                 fltr_th_bound[1])
            scint_th = min_max_denormalize_scalar(norm_scint_th,
                                                  scint_th_bound[0],
                                                  scint_th_bound[1])
            for signal_train, sv, spec_F_train in zip(signal_train_list, src_voltage, spec_F_train_list):
                src_response = interp_src_spectra(src_kV_list, src_spec_list, sv)

                cost += anal_cost(energies, signal_train, spec_F_train, src_response,
                                  fltr_mat,
                                  scint_mat,
                                  fltr_th,
                                  scint_th, signal_weight=[1.0 / sig for sig in signal_train])
            if cost.requires_grad and optimizer_type != 'NNAT_LBFGS':
                cost.backward()
            return cost

        cost = closure()
        if optimizer_type == 'NNAT_LBFGS':
            cost.backward()

        with (torch.no_grad()):
            if i == 1:
                print('Initial cost: %e' % (closure().item()))

        if optimizer_type == 'Adam':
            # cost = closure()
            optimizer.step()
        elif optimizer_type == 'LBFGS':
            optimizer.step(closure)
        elif optimizer_type == 'NNAT_LBFGS':
            options = {'closure': closure, 'current_loss': cost,
                       'max_ls': 100, 'damping': True}
            cost, grad_new, _, _, closures_new, grads_new, desc_dir, fail = optimizer.step(
                options=options)

        with (torch.no_grad()):
            if i % 2 == 0:
                print(
                    'Iteration:{0}: before update cost: {1:e}, source voltage: {2} filter {3} thickness: {4:e}, scintillator {5} thickness: {6:e}'
                    .format(i, closure().item(), [sv.item() for sv in src_voltage], fltr_mat['formula'], fltr_th.item(),
                            scint_mat['formula'],
                            scint_th.item()))

            # Project the updated x back onto the feasible set

            norm_params.data = project_onto_constraints(norm_params.data,
                                                        init_lower_bounds,
                                                        init_upper_bounds)

            norm_src_voltage, norm_fltr_th, norm_scint_th = split_list(norm_params, length)

            src_voltage = min_max_denormalize_scalar(norm_src_voltage,
                                                     src_vol_lower_range_list,
                                                     src_vol_upper_range_list)
            fltr_th = min_max_denormalize_scalar(norm_fltr_th,
                                                 fltr_th_bound[0],
                                                 fltr_th_bound[1])
            scint_th = min_max_denormalize_scalar(norm_scint_th,
                                                  scint_th_bound[0],
                                                  scint_th_bound[1])

            # Check the stopping criterion based on changes in x and y
            if prev_cost is not None and \
                    torch.abs(closure() - prev_cost) / prev_cost < tolerance and \
                    torch.mean(torch.abs(src_voltage - prev_src_voltage) / prev_src_voltage) < tolerance and \
                    torch.abs(fltr_th - prev_fltr_th) < tolerance and \
                    torch.abs(scint_th - prev_scint_th) / prev_scint_th < tolerance:
                print(f"Stopping after {i} iterations")
                break

            prev_cost = closure().item()
            prev_src_voltage = torch.tensor([sv.item() for sv in src_voltage])
            prev_fltr_th = fltr_th.item()
            prev_scint_th = scint_th.item()

            # Clear gradients and update previous values for the next iteration
            if return_history:
                cost_list.append(closure().item())
                src_voltage_list.append([sv.item() for sv in src_voltage])
                fltr_th_list.append(fltr_th.item())
                scint_th_list.append(scint_th.item())

    print(
        f"The minimum cost value: {closure().item()}, occurs at source voltage = {[sv.item() for sv in src_voltage]} kV, "
        f"filter thickness = {fltr_th.item()} mm, scintillator thickness = {scint_th.item()} mm")

    if return_history:
        return cost_list, src_voltage_list, fltr_th_list, scint_th_list
    else:
        return [sv.item() for sv in src_voltage], fltr_th.item(), scint_th.item(), closure().item()


def parallel_anal_sep_model(num_processes,
                            fltr_mat_list, scint_mat_list,
                            init_fltr_th_values, init_scint_th_values, *args, **kwargs):
    # Create parameter combinations
    params_combinations = product(fltr_mat_list, scint_mat_list, init_fltr_th_values,
                                  init_scint_th_values)

    with Pool(processes=num_processes) as pool:
        result_objects = [
            pool.apply_async(
                anal_sep_model,
                args=args,
                kwds={
                    **kwargs,
                    "fltr_mat": fltr_mat,
                    "scint_mat": scint_mat,
                    "init_fltr_th": fltr_th,
                    "init_scint_th": scint_th
                }
            )
            for fltr_mat, scint_mat, fltr_th, scint_th in params_combinations
        ]

        # Gather results
        print('result_objects', result_objects)
        results = [r.get() for r in result_objects]

    return results

    class src_spec_params(self):
        def __int__(self, energies, src_vol_list, src_spec_list, src_vol_bound, voltage=None):
            """A data structure to store and check source spectrum parameters.

            Parameters
            ----------
            energies : numpy.ndarray
                1D numpy array of X-ray energies of a poly-energetic source in units of keV.
            src_vol_list : list
                A list of source voltage corresponding to src_spect_list.
            src_spec_list: list
                A list of source spectrum corresponding to src_vol_list.
            src_vol_bound: class Bound
                Source voltage lower and uppder bound.
            voltage: float or int
                Source voltage. Default is None. Can be set for initial value.

            Returns
            -------

            """
            # Check if voltages in src_vol_list is sorted from small to large
            if not is_sorted(src_vol_list):
                raise ValueError("Warning: source voltage in src_response_dict are not sorted!")
            else:
                self.src_vol_list = src_vol_list

            # Check if the integral of each spectrum is close to 1
            # (considering a very small tolerance for floating-point errors)
            for vol, spectrum in zip(src_vol_list, src_spec_list):
                spectrum_intg = np.trapz(spectrum, energies)
                if not abs(spectrum_intg - 1.0) < 1e-10:  # The tolerance can be adjusted
                    raise ValueError(f"Spectrum at voltage {vol} does not sum to 1. It sums to {spectrum_intg}")
            self.src_spec_list = src_spec_list

            # Check if src_vol_bound is an instance of Bound
            if not isinstance(src_vol_bound, Bound):
                raise ValueError(
                    "Expected an instance of Bound for src_vol_bound, but got {}.".format(type(src_vol_bound).__name__))
            else:
                self.src_vol_bound = src_vol_bound

            # Check voltage
            if voltage is None:
                self.voltage = 0.5 * (src_vol_bound.lower + src_vol_bound.upper)
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

    class fltr_resp_params(self):
        def __int__(self, num_fltr, fltr_mat, fltr_th_bound, fltr_th=None):
            """A data structure to store and check filter response parameters.

            Parameters
            ----------
            num_fltr: int
                Number of filters.
            fltr_mat: class Material or list
                If num_fltr is 1, fltr_mat is an instance of class Material, containing chemical formula and density.
                Otherwise, it whould be a list of instances of class Material.
                Length should be equal to num_fltr.
            fltr_th_bound: class Bound or list
                If num_fltr is 1, fltr_th_bound is an instance of class Bound, containing lower bound and uppder bound.
                Otherwise, it whould be a list of instances of class Bound for filter thickness.
                Length should be equal to num_fltr.
            fltr_th: float or list
                If num_fltr is 1, fltr_th is a non-negative float for filter thickness.
                Otherwise, it whould be a list of filter thickness, which length should be equal to num_fltr.
                Default is None.

            Returns
            -------

            """
            # Check if num_fltr is a positive integer
            if isinstance(num_fltr, int):
                if num_fltr > 0:
                    self.num_fltr = num_fltr
                else:
                    raise ValueError("num_fltr must be positive integer, got: {}".format(num_fltr))
            else:
                raise ValueError("num_fltr must be an integer, got: {}".format(type(num_fltr).__name__))

            if num_fltr == 1:
                fltr_mat = fltr_mat if isinstance(fltr_mat, list) else [fltr_mat]
                fltr_th_bound = fltr_th_bound if isinstance(fltr_th_bound, list) else [fltr_th_bound]

            # Check fltr_mat is an instance of Material
            for fm in fltr_mat:
                if not isinstance(fm, Material):  # The tolerance can be adjusted
                    raise ValueError(
                        "Expected an instance of class Material for fm, but got {}.".format(type(fm).__name__))
            self.fltr_mat = fltr_mat

            # Check if fltr_th_bound is an instance of Bound
            for ftb in fltr_th_bound:
                if not isinstance(ftb, Bound):
                    raise ValueError(
                        "Expected an instance of Bound for ftb, but got {}.".format(type(ftb).__name__))
            self.fltr_th_bound = fltr_th_bound

            # Check fltr_th
            if fltr_th is None:
                fltr_th = [0.5 * (ftb.lower + ftb.upper) for ftb in fltr_th_bound]
            else:
                if isinstance(fltr_th, list):  # if 'fltr_th' is already a list
                    # Convert all elements to float and raise ValueError if any conversion fails
                    try:
                        fltr_th = [float(ft) for ft in fltr_th]
                    except ValueError:
                        raise ValueError("All elements in 'fltr_th' must be convertible to float")
                elif self.num_fltr == 1:  # if there's only one filter, 'fltr_th' can be a single value
                    try:
                        fltr_th = [float(fltr_th)]  # convert single value to float and wrap it in a list
                    except ValueError:
                        raise ValueError("'fltr_th' must be convertible to float")
                else:
                    raise ValueError("'fltr_th' must be a list for multiple filter thickness")

            # Check if fltr_th within fltr_th_bound
            for ft, ftb in zip(fltr_th, fltr_th_bound):
                if not ftb.is_within_bound(ft):
                    raise ValueError(f"Expected 'ft' to be inside ftb, but got {ft}.")
            self.fltr_th = fltr_th

    class scint_cvt_func_params(self):
        def __int__(self, scint_mat, scint_th_bound, scint_th=None):
            """A data structure to store and check scintillator response parameters.

            Parameters
            ----------
            scint_mat
            scint_th_bound
            scint_th

            Returns
            -------

            """
            # Check scint_mat is an instance of Material
            if not isinstance(scint_mat, Material):  # The tolerance can be adjusted
                raise ValueError(
                    "Expected an instance of class Material for scint_mat, but got {}.".format(
                        type(scint_mat).__name__))
            self.scint_mat = scint_mat

            if not isinstance(scint_th_bound, Bound):
                raise ValueError(
                    "Expected an instance of Bound for scint_th_bound, but got {}.".format(
                        type(scint_th_bound).__name__))
            if scint_th_bound.lower <= 0.001:
                raise ValueError(
                    f"Expected lower bound of scint_th is greater than 0.001, but got {scint_th_bound.lower}.")
            self.scint_th_bound = scint_th_bound

            if scint_th is None:
                self.scint_th = 0.5 * (scint_th_bound.lower + scint_th_bound.upper)
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

    def param_based_spec_estimate(energies, y, F, src_config, fltr_config, scint_config, optim_config):
        # Check Variables

        #
        return 0
