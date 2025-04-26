import itertools

from multiprocessing import Pool
import contextlib
from functools import partial
from psutil import cpu_count

import numpy as np
from xcal._utils import huber_func
from xcal.dict_gen import *

class Huber:
    """Solve nonlinear model Huber prior using ICD update.


    The MAP surrogate cost function becomes,



    """

    def __init__(self, c=1.0, l_star=0.001, max_iter=3000, threshold=1e-7):
        """

        Parameters
        ----------

        c : float
            Scalar value :math:`>0` that specifies the Huber threshold parameter.
        l_star : float
            Scalar value :math:`>0` that specifies the Huber regularization.
        max_iter : int
            Maximum number of iterations.
        threshold : float
            Scalar value for stop update threshold.

        """
        self.c = c
        self.l_star = l_star
        self.max_iter = max_iter
        self.threshold = threshold
        self.mbi = 0  # Max beta index
        self.Omega = None

    def set_mbi(self, mbi):
        self.mbi = mbi

    def cost(self):
        huber_list = [huber_func(omega, self.c) for omega in self.Omega]
        cost = 0.5 * np.mean(self.e ** 2 * self.weight) + self.l_star * np.sum(huber_list)
        return cost

    def solve(self, X, y, weight=None, spec_dict=None):
        """

        Parameters
        ----------
        X : numpy.ndarray
            Dictionary matrix.
        y : numpy.ndarray
            Measurement data.

        Returns
        -------
        beta : numpy.ndarray
            Coefficients.

        """
        m, n = np.shape(X)
        Omega = np.ones(n) / n
        c = self.c
        print('%d Dictionary' % n, Omega)

        if weight is None:
            weight = np.ones(y.shape)
        weight = weight.reshape((-1, 1))

        e = y.reshape((-1, 1)) - np.mean(X, axis=-1, keepdims=True)

        for k in range(self.max_iter):
            permuted_ind = np.arange(n)
            permuted_ind = permuted_ind[permuted_ind != self.mbi]
            tol_update = 0
            for i in permuted_ind:
                omega_tmp = Omega[i]
                omega_mbi = Omega[self.mbi]

                if np.abs(Omega[i]) < 1e-9:
                    b1 = self.l_star / 2
                else:
                    b1 = self.l_star * np.clip(Omega[i], -c, c) / (2 * Omega[i])

                if np.abs(omega_mbi) < 1e-9:
                    b2 = self.l_star / 2
                else:
                    b2 = self.l_star * np.clip(-omega_mbi, -c, c) / (-2 * omega_mbi)
                theta_1 = -(e * weight).T @ (X[:, i:i + 1] - X[:, self.mbi:self.mbi + 1]) / m + 2 * b1 * Omega[
                    i] - 2 * b2 * omega_mbi
                theta_2 = ((X[:, i:i + 1] - X[:, self.mbi:self.mbi + 1]) * weight).T @ (
                        X[:, i:i + 1] - X[:, self.mbi:self.mbi + 1]) / m + 2 * b1 + 2 * b2
                update = -theta_1 / theta_2
                Omega[i] = np.clip(omega_tmp + update, 0, omega_tmp + Omega[self.mbi])
                Omega[self.mbi] = 1 - np.sum(Omega[permuted_ind])

                tol_update += np.abs(Omega[i] - omega_tmp)
                e = e - (X[:, i:i + 1] - X[:, self.mbi:self.mbi + 1]) * (Omega[i] - omega_tmp)

            if tol_update < self.threshold:
                print('Stop at iteration:', k, '  Total update:', tol_update)
                break

        print('mbi, Omega_mbi:', self.mbi, Omega[self.mbi])
        print('Omega', Omega)
        self.Omega = Omega
        self.e = e
        self.weight = weight
        return Omega


class L1:
    """Solve L1 prior using pair-wise ICD update.

    """

    def __init__(self, l_star=0.001, nnc='on-coef', max_iter=3000, threshold=1e-7):
        """

        Parameters
        ----------

        l_star : float
            Scalar value :math:`>0` that specifies the Snap regularization.
        nnc : bool
            Whether to use positivity constraint.
        max_iter : int
            Maximum number of iterations.
        threshold : float
            Scalar value for stop update threshold.

        """

        self.l_star = l_star
        self.nnc = nnc
        self.max_iter = max_iter
        self.threshold = threshold
        self.cost_list = []

    def cost(self):
        """Return current value of MAP cost function.

        Returns
        -------
        cost : float
            Return current value of MAP cost function.
        """
        l1_list = [np.abs(omega) for omega in self.Omega]
        cost = 0.5 * np.mean(self.e ** 2 * self.weight) + self.l_star * (np.sum(l1_list) - 1)
        return cost

    def forward_cost(self):
        """Return current value of MAP cost function.

        Returns
        -------
        cost : float
            Return current value of MAP cost function.
        """
        return 0.5 * np.mean(self.e ** 2 * self.weight)

    def prior_cost(self):
        """Return current value of MAP cost function.

        Returns
        -------
            cost : float
                Return current value of MAP cost function.
        """
        l1_list = [np.abs(omega) for omega in self.Omega]
        cost = self.l_star * (np.sum(l1_list) - 1)
        return cost

    def solve(self, FD, y, weight=None, Omega_init=None, e_init=None, spec_dict=None):
        """Use pairwise ICD to solve MAP cost function with L1 prior.

        Parameters
        ----------
        FD : numpy.ndarray
            Forward matrix multiplys subset Dictionary matrix.
        y : numpy.ndarray
            Measurement data.

        Returns
        -------
        omega : numpy.ndarray
            Coefficients.

        """
        self.cost_list = []
        m, n = np.shape(FD)
        if Omega_init is not None:
            Omega = Omega_init
            print('Omega init:', Omega)
        else:
            Omega = np.ones(n) / n
        print('%d Dictionary' % n, Omega)

        if weight is None:
            weight = np.ones(y.shape)
        weight = weight.reshape((-1, 1))

        if e_init is None:
            e = y.reshape((-1, 1)) - np.mean(FD, axis=-1, keepdims=True)
        else:
            e = e_init
        l_star = self.l_star
        self.Omega = Omega.copy()
        self.e = e
        self.weight = weight
        self.cost_list.append(self.cost())
        numbers = np.arange(n)
        pairs = np.array(list(itertools.combinations(numbers, 2)))
        print('Cost before pairwise ICD:', self.cost(), self.forward_cost(), self.prior_cost())
        for K in range(self.max_iter):
            previous_omega = Omega.copy()
            # Generate a random permutation of indices
            perm = np.random.permutation(len(pairs))

            for pair_ind in pairs[perm]:
                k = pair_ind[0]
                g = pair_ind[1]
                omega_tmp_k = Omega[k].copy()
                omega_tmp_g = Omega[g].copy()
                theta_1 = -(e * weight).T @ (FD[:, k:k + 1] - FD[:, g:g + 1]) / m
                theta_2 = ((FD[:, k:k + 1] - FD[:, g:g + 1]) * weight).T @ (
                        FD[:, k:k + 1] - FD[:, g:g + 1]) / m
                update = -theta_1 / theta_2

                if self.nnc == 'on-coef':
                    Omega[k] = np.clip(omega_tmp_k + update, 0, omega_tmp_k + omega_tmp_g)
                    Omega[g] = np.clip(omega_tmp_g - update, 0, omega_tmp_k + omega_tmp_g)
                elif self.nnc == 'on-spec':
                    lt2 = 2 * l_star / theta_2
                    if -omega_tmp_k > omega_tmp_g:
                        osmall = omega_tmp_g
                        olarge = -omega_tmp_k
                    else:
                        osmall = -omega_tmp_k
                        olarge = omega_tmp_g

                    if update > olarge + lt2:
                        update -= lt2
                    elif update < osmall - lt2:
                        update += lt2
                    elif update <= olarge + lt2 and update >= olarge:
                        update = olarge
                    elif update >= osmall - lt2 and update <= osmall:
                        update = osmall

                    estimated_spec = spec_dict @ Omega
                    update_bound = np.clip(estimated_spec, 0, np.inf) / (spec_dict[:, g] - spec_dict[:, k])
                    update_bound_nan_mask = np.isnan(update_bound)
                    update_bound_sign = np.copysign(np.ones_like(update_bound), update_bound)
                    if (update_bound_sign == 1).any():
                        update_ub = np.min(update_bound[np.logical_and(update_bound_sign == 1, ~update_bound_nan_mask)])
                        update_ub = np.max(
                            [0, update_ub - self.threshold / 1000])  # Avoid negative value from percision.
                    else:
                        update_ub = 0
                    if (update_bound_sign == -1).any():
                        update_lb = np.max(
                            update_bound[np.logical_and(update_bound_sign == -1, ~update_bound_nan_mask)])
                        update_lb = np.min([0, update_lb + self.threshold / 1000])
                    else:
                        update_lb = 0
                    update = np.clip(update, update_lb, update_ub)

                    Omega[k] = Omega[k] + update
                    Omega[g] = Omega[g] - update

                e = e - (FD[:, k:k + 1] - FD[:, g:g + 1]) * (Omega[k] - omega_tmp_k)
            avg_update = np.sum(np.abs(previous_omega - Omega)) / n
            total_update = np.sum(np.abs(self.Omega - Omega))
            self.Omega = Omega.copy()
            self.e = e
            self.weight = weight
            if total_update >= self.threshold:
                if K % 100 == 0:
                    print('Iteration %d, Cost:' % K, self.cost())
                self.cost_list.append(self.cost())

            else:
                print('Stop at iteration:', K, '  Average update:', avg_update)
                break

        print('Omega', Omega)

        return Omega


def cal_cost(e, weight, torch_mode=False):
    """Return current value of MAP cost function.

    Returns
    -------
        cost : float
            Return current value of MAP cost function.
    """
    if torch_mode:
        cost = 0.5 * torch.mean(e ** 2 * weight)
    else:
        cost = 0.5 * np.mean(e ** 2 * weight)
    return cost


class Snap:
    """Solve Snap prior using pair-wise ICD update.

    """

    def __init__(self, l_star=0.001, nnc='on-coef', max_iter=3000, threshold=1e-7):
        """

        Parameters
        ----------

        l_star : float
            Scalar value :math:`>0` that specifies the Snap regularization.
        nnc : bool
            Whether to use positivity constraint.
        max_iter : int
            Maximum number of iterations.
        threshold : float
            Scalar value for stop update threshold.

        """

        self.l_star = l_star
        self.nnc = nnc
        self.max_iter = max_iter
        self.threshold = threshold
        self.cost_list = []

    def cost(self):
        """Return current value of MAP cost function.

        Returns
        -------
            cost : float
                Return current value of MAP cost function.
        """
        snap_list = [omega ** 2 for omega in self.Omega]
        cost = 0.5 * np.mean(self.e ** 2 * self.weight) + self.l_star * (1 - np.sum(snap_list)) / 2
        return cost

    def solve(self, FD, y, weight=None, Omega_init=None, e_init=None, spec_dict=None):
        """Use pairwise ICD to solve MAP cost function with Snap prior.

        Parameters
        ----------
        FD : numpy.ndarray
            Forward matrix multiplys subset Dictionary matrix.
        y : numpy.ndarray
            Measurement data.

        Returns
        -------
        omega : numpy.ndarray
            Coefficients.

        """
        self.cost_list = []
        m, n = np.shape(FD)
        if Omega_init is not None:
            Omega = Omega_init
            print('Omega init:', Omega)
        else:
            Omega = np.ones(n) / n
        print('%d Dictionary' % n, Omega)

        if weight is None:
            weight = np.ones(y.shape)
        weight = weight.reshape((-1, 1))

        if e_init is None:
            e = y.reshape((-1, 1)) - np.mean(FD, axis=-1, keepdims=True)
        else:
            e = e_init
        l_star = self.l_star
        self.Omega = Omega.copy()
        self.e = e
        self.weight = weight
        self.cost_list.append(self.cost())
        numbers = np.arange(n)
        pairs = np.array(list(itertools.combinations(numbers, 2)))
        print('Cost before pairwise ICD:', self.cost())
        for K in range(self.max_iter):
            # print('ICD Iteration:', K)
            avg_update = 0
            # Generate a random permutation of indices
            perm = np.random.permutation(len(pairs))

            for pair_ind in pairs[perm]:
                k = pair_ind[0]
                g = pair_ind[1]
                omega_tmp_k = Omega[k].copy()
                omega_tmp_g = Omega[g].copy()
                theta_1 = -(e * weight).T @ (FD[:, k:k + 1] - FD[:, g:g + 1]) / m - l_star * (
                        omega_tmp_k - omega_tmp_g)
                theta_2 = ((FD[:, k:k + 1] - FD[:, g:g + 1]) * weight).T @ (
                        FD[:, k:k + 1] - FD[:, g:g + 1]) / m - 2 * l_star
                if theta_2 < 0:
                    update = -np.sign(theta_1)
                elif np.abs(theta_2) < 1e-9:
                    update = np.sign(theta_1)
                else:
                    update = -theta_1 / theta_2
                if self.nnc == 'on-coef':
                    Omega[k] = np.clip(omega_tmp_k + update, 0, omega_tmp_k + omega_tmp_g)
                    Omega[g] = np.clip(omega_tmp_g - update, 0, omega_tmp_k + omega_tmp_g)
                elif self.nnc == 'on-spec':
                    estimated_spec = spec_dict @ Omega
                    update_bound = np.clip(estimated_spec, 0, np.inf) / (spec_dict[:, g] - spec_dict[:, k])
                    update_bound_nan_mask = np.isnan(update_bound)
                    update_bound_sign = np.copysign(np.ones_like(update_bound), update_bound)
                    # print(update_bound)
                    if (update_bound_sign == 1).any():
                        update_ub = np.min(update_bound[np.logical_and(update_bound_sign == 1, ~update_bound_nan_mask)])
                        update_ub = np.max(
                            [0, update_ub - self.threshold / 1000])  # Avoid negative value from percision.
                    else:
                        update_ub = 0
                    if (update_bound_sign == -1).any():
                        update_lb = np.max(
                            update_bound[np.logical_and(update_bound_sign == -1, ~update_bound_nan_mask)])
                        update_lb = np.min([0, update_lb + self.threshold / 1000])
                    else:
                        update_lb = 0

                    update = np.clip(update, update_lb, update_ub)
                    Omega[k] = Omega[k] + update
                    Omega[g] = Omega[g] - update

                    estimated_spec1 = spec_dict @ Omega
                    if (estimated_spec1 < 0).any():
                        print('Found negative value on the estimate spectrum.')
                        print(k, g)
                        print(update, update_lb, update_ub)
                        print(estimated_spec[estimated_spec1 < 0])
                        print(estimated_spec1[estimated_spec1 < 0])

                e = e - (FD[:, k:k + 1] - FD[:, g:g + 1]) * (Omega[k] - omega_tmp_k)
            avg_update = np.sum(np.abs(self.Omega - Omega)) / n
            total_update = np.sum(np.abs(self.Omega - Omega))
            self.Omega = Omega.copy()
            self.e = e.copy()
            self.weight = weight
            if total_update >= self.threshold:
                if K % 100 == 0:
                    print('Iteration %d, Cost:' % K, self.cost())
                self.cost_list.append(self.cost())

            else:
                print('Stop at iteration:', K, '  Average update:', avg_update)
                break

        print('Omega', Omega)

        return Omega

def cal_first_derivative(a, b, c, d, e, x):
    numerator = b * (d + 2 * c * x) - a * (2 * e + d * x)
    denominator = (b + a * x)**3
    result = numerator / denominator
    return result

def update_xi_matrix(omega_src, omega_fltr, omega_scint):
    # get length of each omega
    l_omega_src = len(omega_src)
    l_omega_fltr = len(omega_fltr)
    l_omega_scint = len(omega_scint)

    # reshape arrays to be able to use broadcasting
    omega_src = omega_src.flatten()[:, None, None, None]
    omega_fltr = omega_fltr.flatten()[None, :, None, None]
    omega_scint = omega_scint.flatten()[None, None, :, None]

    # create identity matrix with extra dimensions
    src_eye = np.eye(l_omega_src)[:, None, None, :]
    fltr_eye = np.eye(l_omega_fltr)[None, :, None, :]
    scint_eye = np.eye(l_omega_scint)[None, None, :, :]

    # compute xi's using broadcasting and in-place multiplication
    xi_src = src_eye * omega_fltr * omega_scint
    xi_fltr = omega_src * fltr_eye * omega_scint
    xi_scint = omega_src * omega_fltr * scint_eye

    # reshape arrays and transpose
    xi_src = xi_src.reshape(-1, l_omega_src)
    xi_fltr = xi_fltr.reshape(-1, l_omega_fltr)
    xi_scint = xi_scint.reshape(-1, l_omega_scint)

    xi_list = [xi_src, xi_fltr, xi_scint]

    return xi_list

class LS_sep_model:
    """Solve least square using pair-wise ICD update.

    """

    def __init__(self, max_iter=3000, threshold=1e-7):
        """

        Parameters
        ----------

        max_iter : int
            Maximum number of iterations.
        threshold : float
            Scalar value for stop update threshold.

        """

        self.max_iter = max_iter
        self.threshold = threshold
        self.cost_list = []

    def cost(self, Z, omega):
        """Return current value of MAP cost function.

        Returns
        -------
            cost : float
                Return current value of MAP cost function.
        """
        cost = 0.5 * np.mean((self.e/(Z@omega)) ** 2 * self.weight)
        return cost

    def solve(self, FD, y, Z, Omega_init, weight=None, e_init=None):
        """Use pairwise ICD to solve MAP cost function with Snap prior.

        Parameters
        ----------
        FD : numpy.ndarray
            Forward matrix multiplys subset Dictionary matrix.
        y : numpy.ndarray
            Measurement data.


        Returns
        -------
        omega : numpy.ndarray
            Coefficients.

        """
        self.cost_list = []
        m, n = np.shape(FD)

        omega_src, omega_fltr, omega_scint = Omega_init
        print('omega_src:', omega_src)
        print('omega_fltr:', omega_fltr)
        print('omega_scint:', omega_scint)
        Omega = (omega_src[:, np.newaxis, np.newaxis] \
                 * omega_fltr[np.newaxis, :, np.newaxis] \
                 * omega_scint[np.newaxis, np.newaxis, :]).reshape((-1, 1))

        if weight is None:
            weight = np.ones(y.shape)
        weight = weight.reshape((-1, 1))

        print('Z.shape:', Z.shape)
        yZ_FD= y@Z-FD
        print('yZ_FD.shape:', yZ_FD.shape)

        if e_init is None:
            e = y.reshape((-1, 1))@(Z@Omega) - FD@Omega
        else:
            e = e_init

        self.Omega = Omega.copy()
        self.e = e.copy()
        self.weight = weight
        self.cost_list.append(self.cost(Z, Omega))

        pairs_src = np.array(list(itertools.combinations(np.arange(len(omega_src)), 2)))
        pairs_fltr = np.array(list(itertools.combinations(np.arange(len(omega_fltr)), 2)))
        pairs_scint = np.array(list(itertools.combinations(np.arange(len(omega_scint)), 2)))

        pairs_list = [pairs_src, pairs_fltr, pairs_scint]


        print('Cost before pairwise ICD:', self.cost(Z, Omega))
        for K in range(self.max_iter):

            for ind, pairs in enumerate(pairs_list):
                omega_src, omega_fltr, omega_scint = Omega_init
                xi_list = update_xi_matrix(omega_src, omega_fltr, omega_scint)

                xi = xi_list[ind]
                omega = Omega_init[ind]
                # Generate a random permutation of indices
                perm = np.random.permutation(len(pairs))
                yZ_FD_xi = yZ_FD @ xi

                for pair_ind in pairs[perm]:
                    k = pair_ind[0]
                    g = pair_ind[1]
                    # print(k, g)
                    omega_tmp_k = omega[k].copy()
                    omega_tmp_g = omega[g].copy()

                    theta_1 = (e * weight).T @ (yZ_FD_xi[:, k:k + 1] - yZ_FD_xi[:, g:g + 1]) / m
                    theta_2 = ((yZ_FD_xi[:, k:k + 1] - yZ_FD_xi[:, g:g + 1]) * weight).T @ (
                            yZ_FD_xi[:, k:k + 1] - yZ_FD_xi[:, g:g + 1]) / m

                    a = Z@xi[:, k] - Z@xi[:, g]
                    b = Z@xi @ omega
                    c = theta_2
                    d = theta_1*2
                    e1 = ((yZ_FD_xi@omega).reshape((-1,1)) * weight).T @ (yZ_FD_xi@omega).reshape((-1,1)) / m

                    if np.abs(a*d-2*b*c) < 1e-6:
                        update = np.inf
                    else:
                        update = (b*d-2*a*e1) / (a*d-2*b*c)

                    if update < - omega_tmp_k or update > omega_tmp_g:
                        if cal_first_derivative(a, b, c, d, e1, 0)>0:
                            update = - omega_tmp_k
                        elif cal_first_derivative(a, b, c, d, e1, 0)< 0:
                            update = omega_tmp_g

                    omega[k] = np.clip(omega_tmp_k + update, 0, omega_tmp_k + omega_tmp_g)
                    omega[g] = np.clip(omega_tmp_g - update, 0, omega_tmp_k + omega_tmp_g)
                    Omega_init[ind] = omega
                    # print('After 1 pair ICD:', Omega, self.cost(Z, Omega))

                    e = e + (yZ_FD_xi[:, k:k + 1] - yZ_FD_xi[:, g:g + 1]) * (omega[k] - omega_tmp_k)
            omega_src, omega_fltr, omega_scint = Omega_init
            Omega = (omega_src[:, np.newaxis, np.newaxis] \
                     * omega_fltr[np.newaxis, :, np.newaxis] \
                     * omega_scint[np.newaxis, np.newaxis, :]).reshape((-1, 1))
            total_update = np.sum(np.abs(self.Omega - Omega))
            self.Omega = Omega.copy()
            self.e = e.copy()
            self.weight = weight
            if total_update >= self.threshold:
                if K % 100 == 0:
                    print('Iteration %d, Cost:' % K, self.cost(Z, Omega))
                self.cost_list.append(self.cost(Z, Omega))

            else:
                print('Stop at iteration:', K, '  Total update:', total_update)
                break

        print('Omega', Omega_init)

        return Omega_init

def _compute_criteria_and_beta(yhat_F, spec_dict, energies, y_yhat, FDsq, y_F_Dk, yhat_sq, S, auto_stop, ysq, signal):
    """

    Parameters
    ----------
    yhat_F
    spec_dict
    energies
    y_yhat
    FDsq
    y_F_Dk
    yhat_sq
    S
    auto_stop
    ysq
    signal

    Returns
    -------

    """

    # Compute yhat_F_Dk
    yhat_F_Dk = np.trapz(yhat_F * spec_dict, energies, axis=0)

    # Compute rho1 and rho2
    rho1 = y_yhat + FDsq - y_F_Dk - yhat_F_Dk
    rho2 = yhat_sq + FDsq - 2 * yhat_F_Dk

    # Calculate beta
    if len(S) == 0:
        beta = np.zeros(rho1.shape)
    else:
        beta = rho1 / rho2

    # Clip beta
    beta = np.clip(beta, 0, 1)
    if auto_stop:
        beta_mask = (beta >= 1)
    else:
        beta_mask = (beta > 1)
    # Calculate criteria
    criteria = ysq + beta ** 2 * yhat_sq + (1 - beta) ** 2 * FDsq - 2 * beta * y_yhat - 2 * (
            1 - beta) * y_F_Dk + 2 * beta * (1 - beta) * yhat_F_Dk
    criteria /= 2 * signal.shape[0]

    return criteria, beta, beta_mask




# Orthogonal match pursuit with different optimization models.
def dictSE(signal, energies, forward_mat, spec_dict, src_info, fltr_info_dict, scints_info_dict,
           sparsity, optimizer, num_candidate=1, max_num_supports=None,
           signal_weight=None, tol=1e-6,
           return_component=False, auto_stop=True, apply_phys_constraint=True, apc_method='traverse',
           verbose=0):
    """A spectral calibration algorithm using dictionary learning.

    This function requires users to provide a large enough spectrum dictionary to estimate the source spectrum.
    By specify sparsity, it will use a greedy algorithm to add selected spectrum to a support from the large dictionary.

    Parameters
    ----------
    signal : numpy.ndarray
        Transmission Data :math:`y` of size (#sets, #views, #rows, #columns). Should be the exponential term instead of the projection after taking negative log.
    energies : numpy.ndarray
        List of X-ray energies of a poly-energetic source in units of keV.
    forward_mat : numpy.ndarray
        A numpy array of forward matrix (#sets * #views * #rows * #columns, #energies)
    spec_dict : numpy.ndarray
        The spectrum dictionary contains N, the number of column, different X-ray spectrum.
        The number of rows M, should be same as the length of energies, should be normalized to integrate to 1.
    sparsity : int
        The max number of nonzero coefficients.
    optimizer : Python object
        Should be one of the optimizers defined above. [RidgeReg(), LassoReg(), ElasticNetReg(), QGGMRF()]
    signal_weight: numpy.ndarray
        Weight for transmission Data, representing uncertainty, has same size as signal.
    tol : float
        The stop threshold.
    return_component : bool
        If true return coefficient.
    verbose: int
        Possible values are {0,1}, where 0 is quiet, 1 prints full information.


    Returns
    -------
    estimated_spec : 1D numpy.ndarray
        The estimated source-spectrum.
    omega : 1D numpy.ndarray
        Coefficients to do linear combination of the basis spectra in dictionary.


    # Examples
    # --------
    # .. code-block:: python
    #
    #
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from phasetorch.spectrum import als_bm832,omp_spec_cali,LassoReg,RidgeReg
    #     from phasetorch._refindxdata import calculate_beta_vs_E
    #     from phasetorch._utils import get_wavelength
    #
    #     if __name__ == '__main__': #Required for parallel compute
    #
    #         # ALS
    #         energies, sp_als =als_bm832()    #Generate dictionary source spectrum
    #         sp_als/= np.trapz(sp_als,energies) # normalized spectrum response.
    #         x, y = np.meshgrid(np.arange(0, 128.0)*0.001, np.arange(0, 128.0)*0.001) #Grid shape 128 x 128. Pixel 0.001mm.
    #         projs = np.zeros_like(x)[np.newaxis] #Ground-truth projections. Unitary dimension indicates one view.
    #         mask = (0.032**2 - (x-0.064)**2) > 0 #Mask for non-zero projections of 0.032mm cylinder.
    #         projs[0][mask] = 2*np.sqrt((0.032**2 - (x-0.064)**2)[mask]) #Generate path lengths.
    #
    #         # Spectrum Calibration with two different materials.
    #         beta_vs_energy = np.array([calculate_beta_vs_E(2.702, 'Al', energies).astype('float32'),
    #                          calculate_beta_vs_E(6.11, 'V', energies).astype('float32')])
    #         beta_projs =projs[np.newaxis,np.newaxis,:,:,:]*beta_vs_energy[:,:, np.newaxis, np.newaxis, np.newaxis]
    #         wnum = 2 * np.pi / get_wavelength(energies)
    #
    #         # Poly-energic
    #         polyE_projs = np.trapz(np.exp(-2*wnum*beta_projs.transpose((0,2,3,4,1)))*sp_als,energies, axis=-1)
    #
    #         # Add noise
    #         Trans_noisy = polyE_projs  + np.sqrt(polyE_projs)*0.1*0.01*np.random.standard_normal(size=polyE_projs.shape)
    #         Trans_noisy = np.clip(Trans_noisy,0,1)
    #
    #         estimated_spec, errs, coef = dictSE(signal = Trans_noisy[:, :,63:64,:],
    #                                                  energies=energies,
    #                                                  beta_projs=beta_projs[:,:,:,63:64,:],
    #                                                  spec_dict=np.array([np.roll(sp_als,i) for i in range(500)]).T,
    #                                                  sparsity=10,
    #                                                  optimizer=LassoReg(l1=0.001),
    #                                                  tol=1e-06,
    #                                                  return_component=True)
    #         plt.plot( energies,sp_als)
    #         plt.plot( energies,estimated_spec)
    """

    signal = np.concatenate([sig.reshape((-1, 1)) for sig in signal])
    forward_mat = np.concatenate([fwm.reshape((-1, fwm.shape[-1])) for fwm in forward_mat])

    if signal_weight is None:
        signal_weight = np.ones(signal.shape)
    signal_weight = np.concatenate([sig.reshape((-1, 1)) for sig in signal_weight])
    if max_num_supports is None:
        max_num_supports = num_candidate

    src_len = len(src_info)
    kV_list = list(set([si[0] for si in src_info]))
    kV_list.sort()
    kv_len = len(kV_list)
    fltr_len = len(fltr_info_dict)
    scint_len = len(scints_info_dict)
    scint_list = list(set([si[0] for si in scints_info_dict]))
    scint_list.sort()

    y = signal.copy()
    omega = np.zeros((spec_dict.shape[1], 1))

    if apply_phys_constraint and apc_method == 'traverse':
        S_list = [[] for iii in range(len(scint_list)) for jjj in range(len(kV_list))]
        omega_list = [omega for iii in range(len(scint_list)) for jjj in range(len(kV_list))]
        FDS_list = [np.zeros((len(signal), 0)) for iii in range(len(scint_list)) for jjj in range(len(kV_list))]
    else:
        S_list = [[]]
        omega_list = [omega]
        FDS_list = [np.zeros((len(signal), 0))]
    res_S_list = []
    res_omega_list = []
    res_estimated_spec_list = []
    res_cost_list = []

    if verbose > 0:
        print(forward_mat.shape)
        print(np.diag(signal_weight).shape)

    # Pre-calculate matrices
    ysq = np.sum(y * signal_weight * y)
    y_F = ((y * signal_weight).T @ forward_mat).reshape((-1, 1))
    y_F_Dk = np.trapz(y_F * spec_dict, energies, axis=0)
    Fsq = np.einsum('ik,k,kj->ij', forward_mat.T, signal_weight.flatten(), forward_mat)

    D_Fsq = np.trapz(spec_dict.T[:, :, np.newaxis] * Fsq[np.newaxis, :, :], energies, axis=1)
    if verbose > 0:
        print('D_Fsq shape:', D_Fsq.shape)
    FDsq = np.trapz(D_Fsq * spec_dict.T, energies, axis=1)
    if verbose > 0:
        print('FDsq shape:', FDsq.shape)

    while len(S_list) > 0:
        print('\nIteration %d:' % (len(S_list[0]) + 1))
        print(S_list)
        appeared_spec_mask = np.zeros(spec_dict.shape[1], dtype=bool)

        new_S_list = []
        new_omega_list = []
        new_FDS_list = []
        new_cost_list = []

        for tst_ind, S, omega, pre_FDS in zip(np.arange(len(S_list)), S_list, omega_list, FDS_list):
            print('Test ID:', tst_ind)
            if len(S) > 0 and apply_phys_constraint:
                dict_ind = S[0]
                kV = kV_list[dict_ind //(src_len//kv_len) // fltr_len // scint_len]
                scint_formula = scints_info_dict[dict_ind % scint_len][0]
                print('kV:', kV, ' scintillator material:', scint_formula)

                scint_mask = np.array([scint_info[0] == scint_formula for scint_info in scints_info_dict])
                kV_mask = np.array([skv[0] == kV for skv in src_info])
                dict_mask = np.zeros((src_len, fltr_len, scint_len), dtype=bool)
                dict_mask[:, :, :] = scint_mask[None, None, :]
                dict_mask[:, :, :] *= kV_mask[:, None, None]
                dict_mask = dict_mask.flatten()

            elif len(S) == 0 and apc_method == 'traverse' and apply_phys_constraint:
                kV = kV_list[tst_ind // len(scint_list)]
                scint_formula = scint_list[tst_ind % len(scint_list)]

                scint_mask = np.array([scint_info[0] == scint_formula for scint_info in scints_info_dict])
                kV_mask = np.array([skv[0] == kV for skv in src_info])
                dict_mask = np.zeros((src_len, fltr_len, scint_len), dtype=bool)
                dict_mask[:, :, :] = scint_mask[None, None, :]
                dict_mask[:, :, :] *= kV_mask[:, None, None]
                dict_mask = dict_mask.flatten()
            else:
                dict_mask = np.ones(spec_dict.shape[1], dtype=bool)

            appeared_spec_mask[S] = True
            estimated_spec = spec_dict @ omega
            print('S:', S)
            print('omega:', omega[S])

            # Find new index.
            if len(S) == 0:
                yhat = np.zeros(signal.shape)
                e = signal - yhat
            else:
                yhat = pre_FDS @ omega[S]
                e = signal - yhat

            # Compute required matrices
            y_yhat = np.sum(y * signal_weight * yhat)
            yhat_sq = np.sum(yhat * signal_weight * yhat)
            yhat_F = ((yhat * signal_weight).T @ forward_mat).reshape((-1, 1))

            criteria, beta, beta_mask = _compute_criteria_and_beta(yhat_F, spec_dict, energies, y_yhat, FDsq, y_F_Dk,
                                                                   yhat_sq, S,
                                                                   auto_stop, ysq, signal)

            criteria = np.ma.array(criteria, mask=np.logical_or.reduce((appeared_spec_mask, beta_mask, ~dict_mask)))

            if criteria.mask.all():
                res_S_list.append(S)
                res_omega_list.append(omega)
                res_estimated_spec_list.append(estimated_spec)
                res_cost_list.append(cal_cost(e, signal_weight))
                continue
            elif criteria.count() >= num_candidate:
                candidate_list = np.argsort(criteria)[:num_candidate]
            else:
                candidate_list = np.argsort(criteria)[:criteria.count()]

            for cl in candidate_list:
                k = [cl]

                new_S = S + k
                new_omega = omega.copy()

                if verbose > 0:
                    print(k)
                    print(beta)
                    print(criteria[k])

                # Build new support

                new_omega[S, 0] = new_omega[S, 0] * beta[k[0]]
                new_omega[k[0], 0] = 1 - beta[k[0]]

                print(new_S)
                FDk = np.trapz(forward_mat[:, :, np.newaxis] * spec_dict[np.newaxis, :, k], energies, axis=1)
                FDS = np.concatenate([pre_FDS, FDk], axis=1)

                # Find best coefficient with new support given solver.
                Omega_init = new_omega[new_S, 0]
                e = signal - FDS @ new_omega[new_S]

                new_omega[new_S, 0] = optimizer.solve(FDS, signal, weight=signal_weight, Omega_init=Omega_init,
                                                      e_init=e,
                                                      spec_dict=spec_dict[:, new_S])
                e = signal - FDS @ new_omega[new_S]

                if len(new_S) == sparsity or np.linalg.norm(e) <= tol:
                    res_S_list.append(new_S)
                    res_omega_list.append(new_omega)
                    res_estimated_spec_list.append(estimated_spec)
                    res_cost_list.append(cal_cost(e, signal_weight))
                else:
                    new_S_list.append(new_S)
                    new_omega_list.append(new_omega)
                    new_FDS_list.append(FDS)
                    cost = optimizer.cost()
                    new_cost_list.append(cost)
                    print('cost:', cost)

        if len(new_cost_list) > 0:
            print('new_cost_list:', new_cost_list)
            print('new_S_list:', new_S_list)
            ncl_id = np.argsort(new_cost_list)[:max_num_supports]
            ncl_id.sort()
            print(ncl_id)
            S_list = [new_S_list[nnid] for nnid in ncl_id]
            omega_list = [new_omega_list[nnid] for nnid in ncl_id]
            FDS_list = [new_FDS_list[nnid] for nnid in ncl_id]
            print('S_list:', S_list)
        else:
            S_list = []
    if return_component:
        return res_estimated_spec_list, res_S_list, res_omega_list, res_cost_list
    else:
        return res_estimated_spec_list


def cal_fw_mat(solid_vol_masks, lac_vs_energies_list, energies, fw_projector):
    """Calculate the forward matrix of multiple solid objects combination with given forward projector.

    Parameters
    ----------
    solid_vol_masks : A list of 3D numpy.ndarray.
        Each mask in the list represents a solid pure object.
    lac_vs_energies_list : A list of linear attenuation coefficient(LAC) vs Energies curves corresponding to solid_vol_masks.
    energies : numpy.ndarray
        List of X-ray energies of a poly-energetic source in units of keV.
    fw_projector : A numpy class.
        Forward projector with a function forward(3D mask, {chemical formula, density(g/cc)}).

    Returns
    -------
    spec_fw_mat : A numpy.ndarray.
        Forward matrix of spectral escal_fw_mattimation.

    """
    linear_att_intg_list = []
    for mask, lac_vs_energies in zip(solid_vol_masks, lac_vs_energies_list):
        linear_intg = fw_projector.forward(mask)
        linear_att_intg_list.append(
            linear_intg[np.newaxis, :, :, :] * lac_vs_energies[:, np.newaxis, np.newaxis, np.newaxis])

    tot_lai = np.sum(np.array(linear_att_intg_list), axis=0)
    fw_mat = np.exp(- tot_lai.transpose((1, 2, 3, 0)))

    return fw_mat


def uncertainty_analysis(signal, back_ground_area, energies, forward_mat, spec_dict, sparsity, num_candidate=1,
                         num_sim=100, num_cores=None, npt_scale=1, anal_mode='add_noise_to_signal'):
    """

    Parameters
    ----------
    signal : numpy.ndarray
        Transmission Data :math:`y` of size (#sets, #views, #rows, #columns). Should be the exponential term instead of the projection after taking negative log.
    back_ground_area : int numpy array. [[row_index, col_index],[height, width]]
        row and column range of background to estimate variance.
    num_sim: int
        Number of simulations that user want to run.
    energies : numpy.ndarray
        List of X-ray energies of a poly-energetic source in units of keV.
    forward_mat : numpy.ndarray
        A numpy array of forward matrix (#sets * #views * #rows * #columns, #energies)
    spec_dict : numpy.ndarray
        The spectrum dictionary contains N, the number of column, different X-ray spectrum.
        The number of rows M, should be same as the length of energies, should be normalized to integrate to 1.
    sparsity : int
        The max number of nonzero coefficients.
    num_cores : Number of cores that user want to use for simulations.
    anal_mode : 'add_noise_to_signal' or 'use_est_as_gt'

    Returns
    -------

    """
    npt_set = []
    back_ground_area = np.array(back_ground_area).astype('int')
    row_index, col_index, height, width = back_ground_area.flatten()

    if num_cores is None:
        num_cores = cpu_count(logical=False)
    lst = np.arange(num_sim)

    for sig in signal:
        npt_set.append(np.std(sig[:, row_index:row_index + height, col_index:col_index + width]))
    npt_set = np.array(npt_set)
    print('Estimated Standard deviation of each set:', npt_set)

    if anal_mode == 'add_noise_to_signal':
        with contextlib.closing(Pool(num_cores)) as pool:
            result_list = pool.map(
                partial(dictse_wrapper, signal=signal, npt_set=npt_scale * npt_set, energies=energies,
                        spec_F_train=forward_mat, spec_dict=spec_dict, num_cores=num_cores,
                        sparsity=sparsity, num_candidate=num_candidate), lst)

    elif anal_mode == "use_est_as_gt":
        Snapprior = Snap(l_star=0, max_iter=500, threshold=1e-5, nnc='on-coef')
        estimated_spec, omega, S, cost_list = dictSE(signal, energies, forward_mat,
                                                     spec_dict.reshape(-1, spec_dict.shape[-1]), sparsity,
                                                     optimizer=Snapprior, nnc='on-coef',
                                                     signal_weight=[1.0 / sig for sig in signal], auto_stop=True,
                                                     return_component=True,
                                                     verbose=0)
        ideal_proj = [np.trapz(fwm * estimated_spec.flatten(), energies, axis=-1).reshape(sig.shape) for sig, fwm in
                      zip(signal, forward_mat)]
        with contextlib.closing(Pool(num_cores)) as pool:
            result_list = pool.map(
                partial(dictse_wrapper, signal=ideal_proj, npt_set=npt_scale * npt_set, energies=energies,
                        spec_F_train=forward_mat, spec_dict=spec_dict, num_cores=num_cores,
                        sparsity=sparsity, num_candidate=num_candidate), lst)
    else:
        print('No analysis mode call:', anal_mode)
    return result_list


def dictse_wrapper(ii, signal, npt_set, energies, spec_F_train, spec_dict, num_cores, sparsity, num_candidate):
    signal_train = [sig + np.sqrt(sig) * np.random.normal(0, npt, size=(num_cores,) + sig.shape)[ii % num_cores] for
                    sig, npt in zip(signal, npt_set)]
    Snapprior = Snap(l_star=0, max_iter=500, threshold=1e-5, nnc='on-coef')
    return dictSE(signal_train, energies, spec_F_train,
                  spec_dict.reshape(-1, spec_dict.shape[-1]), sparsity, optimizer=Snapprior,
                  num_candidate=num_candidate,
                  nnc='on-coef', signal_weight=[1.0 / sig for sig in signal], auto_stop=True, return_component=True,
                  verbose=0)


# Orthogonal match pursuit with different optimization models.
def dictSE_sep_model(signal, energies, forward_mat,
                     src_dict, fltr_dict, scint_dict, optimizer,
                     signal_weight=None, return_component=False, verbose=0):
    signal = np.concatenate([sig.reshape((-1, 1)) for sig in signal])
    forward_mat = np.concatenate([fwm.reshape((-1, fwm.shape[-1])) for fwm in forward_mat])

    if signal_weight is None:
        signal_weight = np.ones(signal.shape)
    signal_weight = np.concatenate([sig.reshape((-1, 1)) for sig in signal_weight])

    src_len = src_dict.shape[1]
    fltr_len = fltr_dict.shape[1]
    scint_len = scint_dict.shape[1]

    y = signal.copy()
    omega_src = np.zeros((src_dict.shape[1], 1))
    omega_fltr = np.zeros((fltr_dict.shape[1], 1))
    omega_scint = np.zeros((scint_dict.shape[1], 1))

    S_src = []
    S_fltr = []
    S_scint = []
    S = [S_src, S_fltr, S_scint]
    cost_list = []

    if verbose > 0:
        print(forward_mat.shape)
        print(np.diag(signal_weight).shape)

    assert src_dict.shape[0] == fltr_dict.shape[0] == scint_dict.shape[0], "Number of energy bins are not equal among source responses, flter responses, and scintillator responses."

    # Pre-calculate matrices
    spec_dict = (src_dict[:, :, np.newaxis, np.newaxis] \
                * fltr_dict[:, np.newaxis, :, np.newaxis] \
                * scint_dict[:, np.newaxis, np.newaxis, :]).reshape((src_dict.shape[0], -1))
    Z = np.trapz(spec_dict, energies, axis=0).reshape((1, spec_dict.shape[-1]))

    yZ = y.reshape((-1, 1)) @ Z
    print('yZ shape:', yZ.shape)
    S_list = []
    omega_sfs_list = []
    for i_src in range(src_len):
        for i_fltr in range(fltr_len):
            for i_scint in range(scint_len):
                if i_src < src_len-1:
                    S_src = [i_src, i_src+1]
                elif src_len == 1:
                    S_src = [i_src]
                else:
                    continue
                if i_fltr < fltr_len-1:
                    S_fltr = [i_fltr, i_fltr+1]
                elif fltr_len == 1:
                    S_fltr = [i_fltr]
                else:
                    continue
                if i_scint < scint_len-1:
                    S_scint = [i_scint, i_scint+1]
                elif scint_len == 1:
                    S_scint = [i_scint]
                else:
                    continue
                S = [S_src, S_fltr, S_scint]
                S_list.append(S)
                print('Neiborhood area:', S)

                omega_src = np.zeros((src_dict.shape[1], 1))
                omega_fltr = np.zeros((fltr_dict.shape[1], 1))
                omega_scint = np.zeros((scint_dict.shape[1], 1))
                omega_src[S_src] = 1/len(S_src)
                omega_fltr[S_fltr] = 1/len(S_fltr)
                omega_scint[S_scint] = 1/len(S_scint)

                omega = (omega_src[:, np.newaxis, np.newaxis] \
                         * omega_fltr[np.newaxis, :, np.newaxis] \
                         * omega_scint[np.newaxis, np.newaxis, :]).reshape((-1, 1))

                # Find best coefficient with new support given solver.
                SS = [a*fltr_len*scint_len+b*scint_len+c for a in S_src for b in S_fltr for c in S_scint]
                FDS = np.trapz(forward_mat[:, :, np.newaxis] * spec_dict[np.newaxis, :, SS], energies, axis=1)

                e = (yZ[:,SS] - FDS) @ omega[SS]
                print('Cost before ICD:', cal_cost(e/(Z[:,SS]@ omega[SS]), signal_weight))
                omega_src[S_src], omega_fltr[S_fltr], omega_scint[S_scint] = optimizer.solve(FDS, signal, Z[:, SS],
                                                                                             Omega_init=[omega_src[S_src], omega_fltr[S_fltr], omega_scint[S_scint]],
                                                                                             weight=signal_weight,
                                                                                             e_init=e)
                omega_sfs = [omega_src, omega_fltr, omega_scint]
                omega_sfs_list.append(omega_sfs)

                new_omega = (omega_src[:, np.newaxis, np.newaxis] \
                         * omega_fltr[np.newaxis, :, np.newaxis] \
                         * omega_scint[np.newaxis, np.newaxis, :]).reshape((-1, 1))
                e = (yZ[:, SS] - FDS) @ new_omega[SS]
                updated_cost = cal_cost(e / (Z[:, SS] @ new_omega[SS]), signal_weight)
                print('Cost after ICD:', updated_cost)
                cost_list.append(updated_cost)
                #cost_list.append(optimizer.cost_list)


    best_cdi = np.argmin(cost_list)
    omega_sfs = omega_sfs_list[best_cdi]
    omega_src, omega_fltr, omega_scint = omega_sfs
    omega = (omega_src[:, np.newaxis, np.newaxis] \
            * omega_fltr[np.newaxis, :, np.newaxis] \
            * omega_scint[np.newaxis, np.newaxis, :]).reshape((-1, 1))
    estimated_spec = spec_dict @ omega

    if return_component:
        return estimated_spec, S_list, omega_sfs_list, cost_list, #S_list, omega_sfs_list
    else:
        return estimated_spec


def gen_system_response(energies, src_response, M_fl, M_sc, th_fl, th_sc):
    # Calculate filter response
    fltr_params = [
        {'formula': M_fl['formula'], 'density': M_fl['density'], 'thickness_list': [th_fl]},
    ]
    fltr_response, fltr_info = gen_filts_specD(energies, composition=fltr_params)
    #     print('Filter response:', fltr_response.T.shape)

    # Calculate scintillator response
    scint_params = [
        {'formula': M_sc['formula'], 'density': M_sc['density'], 'thickness_list': [th_sc]},
    ]
    scint_response, scint_info = gen_scints_specD(energies, composition=scint_params)
    #     print('Scintillator response:', scint_response.T.shape)

    # Calculate total system response as a product of source, filter, and scintillator responses.
    sys_response = (src_response * fltr_response * scint_response).T

    return sys_response


