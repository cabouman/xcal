import sys
import atexit
import warnings
import numpy as np
import copy

import torch
import torch.optim as optim
from torch.multiprocessing import Pool
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')
mp.set_start_method("spawn", force=True)

import logging
from xcal.opt._pytorch_lbfgs.functions.LBFGS import FullBatchLBFGS as NNAT_LBFGS
from xcal._utils import *
from xcal.models import get_merged_params_list, get_concatenated_params_list, denormalize_parameter_as_tuple, clamp_with_grad
def weighted_mse_loss(input, target, weight):
    return 0.5 * torch.mean(weight * (input - target) ** 2)


def calc_forward_matrix(homogenous_vol_masks, lac_vs_energies, forward_projector, slices=None):
    """
    Calculate the forward matrix for a combination of multiple solid objects using a given forward projector.

    Args:
        homogenous_vol_masks (list of numpy.ndarray): Each 3D array in the list represents a mask for a homogenous,
            pure object.
        lac_vs_energies (list of numpy.ndarray): Each 1D array contains the linear attenuation coefficient (LAC)
            curve and the corresponding energies for the materials represented in `homogenous_vol_masks`.
        forward_projector (object): An instance of a class that implements a forward projection method. This
            instance should have a method, forward(mask), that takes a 3D volume mask as input and computes the photon's line
            path length.
        slices (tuple of slice objects, optional): Slices to apply to the forward projection output to reduce memory
            usage. Each element in the tuple corresponds to a dimension of the 3D volume output of the forward_projector
            (views, rows, and columns), and specifies the portion of the data to include in the calculation. If not
            provided, the entire volume will be used.

    Returns:
        numpy.ndarray: The calculated forward matrix for spectral estimation. This matrix represents the exponential
        attenuation of photons through the combined materials, with dimensions corresponding to the input volumes and
        the energy levels specified in `lac_vs_energies`.
    """


    linear_att_intg_list = []
    for mask, lac_vs_energies in zip(homogenous_vol_masks, lac_vs_energies):
        linear_intg = forward_projector.forward(mask)
        if slices is None:
            linear_att_intg = linear_intg[np.newaxis, :] * lac_vs_energies[:, np.newaxis, np.newaxis, np.newaxis]
        else:
            linear_att_intg = linear_intg[(np.newaxis,) + slices] * lac_vs_energies[:, np.newaxis, np.newaxis, np.newaxis]
        linear_att_intg_list.append(linear_att_intg)

    tot_lai = np.sum(np.array(linear_att_intg_list), axis=0)
    forward_matrix = np.exp(- tot_lai.transpose((1, 2, 3, 0)))

    return forward_matrix

def fit_cell(energies,
             nrads,
             forward_matrices,
             spec_models,
             params,
             weights=None,
             learning_rate=0.02,
             max_iterations=5000,
             stop_threshold=1e-3,
             optimizer_type='NNAT_LBFGS',
             loss_type='transmission'):
    """Arguments are same as param_based_spec_estimate.

    """

    logger = logging.getLogger(str(mp.current_process().pid))

    def print(*args, **kwargs):
        message = ' '.join(map(str, args))
        logger.info(message)

    def print_params(params):
        for key, value in sorted(params.items()):
            if isinstance(value, tuple):
                dv = denormalize_parameter_as_tuple(value)
                dd = torch.clamp(dv[0], dv[1], dv[2])
                print(f"{key}: {dd.numpy()}")
            else:
                print(f"{key}: {value}")
        print()


    spec_models = [[copy.deepcopy(cm) for cm in component_models] for component_models in spec_models]
    params = copy.deepcopy(params)
    parameters = []
    for component_models in spec_models:
        for cm in component_models:
            cm.set_params(params)
            parameters += list(cm.parameters())
    parameters = list(set(parameters))
    loss = torch.nn.MSELoss()

    if optimizer_type == 'Adam':
        ot = 'Adam'
        iter_prt = 50
        optimizer = optim.Adam(parameters, lr=learning_rate)
    elif optimizer_type == 'NNAT_LBFGS':
        ot = 'NNAT_LBFGS'
        iter_prt = 5
        optimizer = NNAT_LBFGS(parameters, lr=learning_rate)
    else:
        warnings.warn(f"The optimizer type {optimizer_type} is not supported.")
        sys.exit("Exiting the script due to unsupported optimizer type.")

    cost = np.inf
    print('Start Estimation.')
    for iter in range(1, max_iterations + 1):
        if iter % iter_prt == 0:
            print('Iteration:', iter)

        def closure():
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            cost = 0
            for yy, FF, ww, component_models in zip(nrads, forward_matrices, weights, spec_models):
                spec = component_models[0](energies)
                for cm in component_models[1:]:
                    spec = spec*cm(energies)
                spec /= torch.trapz(spec, energies)
                trans_value = torch.trapz(FF * spec, energies, axis=-1).reshape((-1, 1))

                if loss_type == 'transmission':
                    sub_cost = weighted_mse_loss(trans_value, yy, ww)
                elif loss_type == 'attmse':
                    sub_cost = 0.5 * loss(-torch.log(trans_value), -torch.log(yy))
                elif loss_type == 'least_square':
                    sub_cost = 0.5 * loss(trans_value, yy)
                else:
                    raise ValueError('loss_type should be \'mse\' or \'wmse\' or \'attmse\'. ', 'Given', loss_type)
                cost += sub_cost
            if cost.requires_grad and ot != 'NNAT_LBFGS':
                cost.backward()
            return cost

        cost = closure()

        if torch.isnan(cost):
            print('Meet NaN!!')
            for component_models in spec_models:
                for cm in component_models:
                    print(cm.get_params())
            return iter, closure().item(), params

        if ot == 'NNAT_LBFGS':
            cost.backward()

        for component_models in spec_models:
            for cm in component_models:
                has_nan = check_gradients_for_nan(cm)
                if has_nan:
                    return iter, closure().item(), params

        with (torch.no_grad()):
            if iter == 1:
                print('Initial cost: %e' % (closure().item()))

        # Before the update, clone the current parameters
        old_params = [parameter.data.clone() for parameter in parameters]

        if ot == 'Adam':
            optimizer.step()
        elif ot == 'NNAT_LBFGS':
            options = {'closure': closure, 'current_loss': cost,
                       'max_ls': 100, 'damping': False}
            cost, grad_new, _, _, closures_new, grads_new, desc_dir, fail = optimizer.step(options=options)

        with (torch.no_grad()):
            if iter % iter_prt == 0:
                print('Cost:', cost.item())
                print_params(params)
            # After the update, check if the update is too small
            small_update = True
            for parameter,old_param in zip(parameters,old_params):
                if torch.norm(parameter.data.clamp(0, 1) - old_param.clamp(0, 1)) > stop_threshold:
                    small_update = False
                    break

            if small_update:
                print(f"Stopping at epoch {iter} because updates are too small.")
                print('Cost:', cost.item())
                print_params(params)
                break
    return iter, cost.item(), params


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


class Estimate():
    def __init__(self, energies):
        """The Estimate class provides a structured approach for parameter estimation by separating input arguments into data and optimization domains, thereby reducing duplicate input. The Estimate class provides estimation of both discrete and continuous parameters within a unified framework.

        Args:
            energies (numpy.ndarray): X-ray energies of a poly-energetic source in units of keV.
        """
        self.energies = torch.tensor(energies, dtype=torch.float32)
        self.nrads = []
        self.forward_matrices = []
        self.spec_models = []
        self.weights = []


    def add_data(self, nrad, forward_matrix, component_models, weight=None):
        """Add data for parameter estimation, which allows adding multiple datasets scanned with different X-ray system setting.

        Args:

            nrad (numpy.ndarray): Normalized radiograph with dimensions [N_views, N_rows, N_cols].
            forward_matrix (numpy.ndarray): Forward matricx corresponds to nrad with dimensions [N_views, N_rows, N_cols, N_energiy_bins]. We provide ``xcal.calc_forward_matrix.rst`` to calculate a forward matrix from a 3D mask for a homogenous object.
            component_models (object): An instance of Base_Spec_Model.
            weight (numpy.ndarray): Weight corresponds to the normalized radiograph.

        Returns:

        """
        self.nrads.append(torch.tensor(nrad.reshape((-1, 1)), dtype=torch.float32))
        self.num_sp_datasets = len(self.nrads)
        self.forward_matrices.append(torch.tensor(forward_matrix, dtype=torch.float32))
        self.spec_models.append(component_models)

        if weight is None:
            weight = 1.0 / self.nrads[-1]
        else:
            weight = torch.tensor(weight.reshape((-1, 1)), dtype=torch.float32)
        self.weights.append(weight)

    def fit(self, learning_rate=0.001, max_iterations=5000, stop_threshold=1e-4,
            optimizer_type='Adam', loss_type='transmission', logpath=None,
             num_processes=1):
        """Estimate both discrete and continuous parameters.

        Args:
            learning_rate (float, optional): [Default=0.001] Learning rate for the optimization process.
            max_iterations (int, optional): [Default=5000] Maximum number of iterations for the optimization.
            stop_threshold (float, optional): [Default=1e-4] Scalar valued stopping threshold in percent.
                If stop_threshold=0.0, then run max iterations.
            optimizer_type (str, optional): [Default='Adam'] Type of optimizer to use. If we do not have
                accurate initial guess use 'Adam', otherwise, 'NNAT_LBFGS' can provide a faster convergence.
            loss_type (str, optional): [Default='transmission'] Calculate loss function in 'transmission' or 'attenuation' space.
            logpath (optional): [Default=None] Path for logging, if required.
            num_processes (int, optional): [Default=1] Number of processes to use for parallel computation.
        Returns:

        """
        # Calculate params_list
        concatenate_params_list = [get_concatenated_params_list([cm._params_list for cm in concatenate_models]) for
                                   concatenate_models in self.spec_models]
        merged_params_list = get_merged_params_list(concatenate_params_list)

        # Use multiprocessing pool to parallelize the optimization process
        with Pool(processes=num_processes, initializer=init_logging, initargs=(logpath, num_processes)) as pool:
            # Apply optimization function to each combination of model parameters
            result_objects = [
                pool.apply_async(
                    fit_cell,
                    args=(self.energies, self.nrads, self.forward_matrices, self.spec_models, params,
                    self.weights, learning_rate,
                    max_iterations, stop_threshold,
                    optimizer_type, loss_type)
                )
                for params in merged_params_list
            ]

            # Gather results from all parallel optimizations
            print('Number of cases for different discrete parameters:', len(result_objects))
            results = [r.get() for r in result_objects]  # Retrieve results from async calls
        cost_list = [res[1] for res in results]
        optimal_cost_ind = np.argmin(cost_list)
        best_params = results[optimal_cost_ind][2]
        self.params = best_params
        self.results = results
        for component_models in self.spec_models:
            for cm in component_models:
                cm.set_params(best_params)
    def get_spec_models(self):
        """ Obtain optimized spectral models.

        Returns:
            list: A list of compenent lists. Each compenent list contains all used components to scan the corresponding radiograph.
        """
        return self.spec_models

    def get_spectra(self):
        """ Obtain optimized system responses corresponding to list of added nrad.

        Returns:
            list: A list of system responses.
        """
        spec_list = []
        for sms in self.spec_models:
            est_sp = torch.ones(self.energies.shape)
            for sm in sms:
                est_sp*=sm(self.energies)
            spec_list.append(est_sp)
        return spec_list

    def get_params(self):
        """
        Read estimated parameters as a dictionary.

        Returns:
            dict: Dictionary containing estimated parameters.
        """
        display_estimates = {}
        for key, value in self.params.items():
            if isinstance(value, tuple):
                dv = denormalize_parameter_as_tuple(value)
                display_estimates[key] =  clamp_with_grad(dv[0], dv[1], dv[2])
            else:
                display_estimates[key] = value
        return display_estimates

    def get_all_estimates(self):
        """
        Generates a list of tuples, each containing a combination of discrete
        and continuous parameters. This function explores all possible combinations
        of given parameters to facilitate comprehensive analysis or optimization tasks.

        Each tuple in the list comprises three elements:
        1. Stopped iterations: The number of iterations after which the evaluation stopped.
        2. Cost value: The cost or objective function value associated with the parameter combination.
        3. A dictionary of estimated parameters: Keys are parameter names, and values are the corresponding discrete or continuous values for that combination.

        Returns:
            List[Tuple[int, float, Dict[str, Union[int, float]]]]: A list of tuples, each representing
            a unique combination of parameters and their evaluation metrics.
        """
        return self.results



def least_squares_estimation(energies, A_np, y_np, x_init_np, weights_np=None, num_iterations=1000, learning_rate=1e-3, smoothness_lambda=0.01, non_neg_lambda=0.01, change_lambda=0.01, change_scale=10000, change_threshold=0.001, stop_threshold=1e-5):
    """
    Perform least squares estimation using Adam optimizer to solve y = Ax, ensuring that x is non-negative
    and sums to one (treated as a probability distribution), with optional weighted loss, smoothness regularization,
    and a stopping threshold when updates to x become very small. Prints loss every 50 iterations.

    Args:
        A_np (np.ndarray): The matrix coefficients of the linear model as a numpy array. Shape should be (m, n).
        y_np (np.ndarray): The output vector as a numpy array. Shape should be (m,).
        x_init_np (np.ndarray): Initial estimate of the parameter vector x as a numpy array. Shape should be (n,).
        weights_np (np.ndarray, optional): Weights for each observation, affecting their contribution to the loss.
                                          Shape should be (m,). If None, equal weighting is assumed.
        num_iterations (int): Number of iterations for the optimization.
        learning_rate (float): Learning rate for the optimization.
        smoothness_lambda (float): Regularization parameter for promoting smoothness in the solution.
        non_neg_lambda (float): Regularization parameter for enforcing non-negativity and sum-to-one constraint.
        change_lambda (float): Regularization parameter for limiting changes from the initial estimate.
        stop_threshold (float): Threshold for stopping the optimization when the change in x is small.

    Returns:
        np.ndarray: The estimated parameters x, non-negative and summing to one. Shape will be (n,).
    """
    # Convert numpy arrays to PyTorch tensors
    energies = torch.tensor(energies, dtype=torch.float32)
    A = torch.tensor(A_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    x_init = torch.tensor(x_init_np, dtype=torch.float32)
    x = torch.tensor(x_init_np, dtype=torch.float32, requires_grad=True)
    if weights_np is not None:
        weights = torch.tensor(weights_np, dtype=torch.float32)
    else:
        weights = torch.ones(y.shape[0], dtype=torch.float32)

    # Define the optimizer using Adam
    optimizer = torch.optim.Adam([x], lr=learning_rate)

    # Initialize x_old for the first iteration
    x_old = torch.zeros_like(x)

    # Optimization loop
    print('Start Estimation.')
    for iteration in range(num_iterations):
        optimizer.zero_grad()  # Clear previous gradients
        y_pred = torch.trapz(A * x, energies, axis=-1).reshape((-1, 1))
        loss = torch.mean(weights * (y_pred - y) ** 2)  # Weighted mean squared error loss

        # Add smoothness regularization if required
        if smoothness_lambda > 0:
            smoothness_loss = torch.sum((x[:-1] - x[1:])**2)
            loss += smoothness_lambda * smoothness_loss

        # Add non-negativity and sum-to-one constraints
        non_neg_loss = torch.sum(torch.relu(-x) ** 2) + (torch.sum(x) - 1) ** 2
        loss += non_neg_lambda * non_neg_loss

        # Add change regularization term
        change_penalty = torch.sum(torch.max(torch.abs(x - x_init) - change_scale * x_init, torch.tensor(change_threshold))- torch.tensor(change_threshold)) 
        loss += change_lambda * change_penalty

        loss.backward()  # Perform backpropagation
        optimizer.step()  # Update the parameters

        # Apply non-negativity constraint and normalize to sum to one
        with torch.no_grad():  # Update without tracking gradient

            # Check if the update is smaller than the stop threshold
            if torch.norm(x - x_old) < stop_threshold:
                break

            x_old = x.clone()  # Update x_old with the new values

        # Print loss every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}: Loss = {loss.item()}")
            with torch.no_grad():
                print(f"forward loss: {torch.mean(weights * (y_pred - y) ** 2)}; non-negative loss: {non_neg_lambda * non_neg_loss}; change penalty loss: {change_lambda * change_penalty}")

    # Return the estimated x as a numpy array
    return x.detach().numpy()
