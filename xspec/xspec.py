"""
Author: Wenrui Li
Email: li3120@purdue.edu
Description:
    This script includes dictionary learning algorithm to do spectrum calibration.

Date: Dec 1st, 2022.
"""


import numpy as np
import matplotlib.pyplot as plt

from xspec._utils import get_wavelength, binwised_spec_cali_cost, huber_func




class Huber:
    """Slove nonliner model Huber prior using ICD update.


    The MAP surrogate cost function becomes,



    """
    def __init__(self, c=1.0, l_star=0.001, max_iter=3000, threshold=1e-7):
        """

        Parameters
        ----------

        c : float
            Scalar value :math:`>0` that specifies the Huber threshold parameter.
        l : float
            Scalar value :math:`>0` that specifies the Huber regularization.
        sigma : float
            Scalar value :math:`>0` that specifies the :math:`||\\sum beta|| =1`.
        max_iter : int
            Maximum number of iterations.
        threshold : float
            Scalar value for stop update threshold.

        """
        self.c = c
        self.l_star = l_star
        self.max_iter = max_iter
        self.threshold = threshold
        self.mbi = 0 # Max beta index
        self.Omega=None
        
    def set_mbi(self, mbi):
        self.mbi = mbi
        
    def cost(self):
        huber_list = [huber_func(omega,self.c) for omega in self.Omega]
        cost = 0.5*np.mean(self.e**2*self.weight)+self.l_star*np.sum(huber_list) 
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
        Omega = np.ones(n)/n
        c = self.c
        print('%d Dictionary'%n, Omega)
        
        if weight is None:
            weight = np.ones(y.shape)
        weight = weight.reshape((-1, 1))
        
        e = y.reshape((-1, 1))-np.mean(X,axis=-1,keepdims=True)

        for k in range(self.max_iter):
            permuted_ind = np.arange(n)
            permuted_ind = permuted_ind[permuted_ind!=self.mbi]
            tol_update = 0
            for i in permuted_ind:
                omega_tmp = Omega[i]
                omega_mbi = Omega[self.mbi]

                if np.abs(Omega[i])<1e-9:
                    b1 = self.l_star/2
                else:
                    b1 = self.l_star*np.clip(Omega[i],-c,c)/(2*Omega[i])

                if np.abs(omega_mbi)<1e-9:
                    b2 = self.l_star/2
                else:
                    b2 = self.l_star*np.clip(-omega_mbi,-c,c)/(-2*omega_mbi)
                theta_1 = -(e*weight).T @ (X[:, i:i + 1]-X[:,self.mbi:self.mbi+1]) / m + 2*b1 * Omega[i] -2*b2*omega_mbi
                theta_2 = ((X[:, i:i + 1]-X[:,self.mbi:self.mbi+1])*weight).T @ (X[:, i:i + 1]-X[:,self.mbi:self.mbi+1]) / m + 2 * b1 +2*b2
                update = -theta_1 / theta_2
                Omega[i] = np.clip(omega_tmp + update, 0, omega_tmp+Omega[self.mbi])
                Omega[self.mbi] = 1 - np.sum(Omega[permuted_ind])

                tol_update += np.abs(Omega[i]-omega_tmp)
                e = e - (X[:, i:i + 1]-X[:,self.mbi:self.mbi+1]) * (Omega[i]-omega_tmp)
            
            if tol_update < self.threshold:
                print('Stop at iteration:', k,'  Total update:', tol_update)
                break
                
        
        print('mbi, Omega_mbi:',self.mbi,Omega[self.mbi])
        print('Omega',Omega)
        self.Omega=Omega
        self.e = e
        self.weight = weight
        return Omega


class Snap:
    """Slove Snap prior using pair-wise ICD update.

    """

    def __init__(self, l_star=0.001, max_iter=3000, threshold=1e-7):
        """

        Parameters
        ----------

        l_star : float
            Scalar value :math:`>0` that specifies the Snap regularization.
        max_iter : int
            Maximum number of iterations.
        threshold : float
            Scalar value for stop update threshold.

        """

        self.l_star = l_star
        self.max_iter = max_iter
        self.threshold = threshold
        self.mbi = 0  # Max beta index
        self.cost_list = []

    def _set_mbi(self, mbi):
        self.mbi = mbi

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
        l = self.l_star
        self.Omega = Omega
        self.e = e
        self.weight = weight
        self.cost_list.append(self.cost())
        for k in range(self.max_iter):
            permuted_ind = np.random.permutation(n)
            permuted_ind = permuted_ind[permuted_ind != self.mbi]
            tol_update = 0
            for i in permuted_ind:
                omega_tmp = Omega[i]
                omega_mbi = Omega[self.mbi]
                theta_1 = -(e * weight).T @ (FD[:, i:i + 1] - FD[:, self.mbi:self.mbi + 1]) / m - l * (
                            omega_tmp - omega_mbi)
                theta_2 = ((FD[:, i:i + 1] - FD[:, self.mbi:self.mbi + 1]) * weight).T @ (
                        FD[:, i:i + 1] - FD[:, self.mbi:self.mbi + 1]) / m - 2 * l
                if theta_2 < 0:
                    update = -np.sign(theta_1)
                else:
                    update = -theta_1 / theta_2
                Omega[i] = np.clip(omega_tmp + update, 0, omega_tmp + omega_mbi)
                Omega[self.mbi] = np.clip(omega_mbi - update, 0, omega_tmp + omega_mbi)
                tol_update += np.abs(Omega[i] - omega_tmp)
                e = e - (FD[:, i:i + 1] - FD[:, self.mbi:self.mbi + 1]) * (Omega[i] - omega_tmp)
            self.Omega = Omega
            self.e = e
            self.weight = weight
            self.mbi = np.argmax(Omega)
            if tol_update > self.threshold / 10:
                self.cost_list.append(self.cost())

            if tol_update < self.threshold:
                print('Stop at iteration:', k, '  Total update:', tol_update)
                break

        print('mbi, omega_mbi:', self.mbi, Omega[self.mbi])
        print('Omega', Omega)

        return Omega

# Orthogonal match pursuit with different optimization models.
def dictSE(signal, energies, forward_mat, spec_dict, sparsity, optimizor, signal_weight=None,
           tol=1e-6, return_component=False, verbose=0):
    """A spectral calibration algorithm using dictionary learning.

    This function requires users to provide a large enough spectrum dictionary to estimate the source spectrum.
    By specify sparsity, it will use a greedy algorithm to add selected spectrum to a support from the large dictionary.

    Parameters
    ----------
    signal : numpy.ndarray
        Transmission Data :math:`y` of size (#sets, #energies, #views, #rows, #columns). Should be the exponential term instead of the projection after taking negative log.
    energies : numpy.ndarray
        List of X-ray energies of a poly-energetic source in units of keV.
    forward_mat : numpy.ndarray
        A numpy array of forward matrix (#sets * #views * #rows * #columns, #energies)
    spec_dict : numpy.ndarray
        The spectrum dictionary contains N, the number of column, different X-ray spectrum.
        The number of rows M, should be same as the length of energies, should be normalized to integrate to 1.
    sparsity : int
        The max number of nonzero coefficients.
    optimizor : Python object
        Should be one of the optimizors defined above. [RidgeReg(), LassoReg(), ElasticNetReg(), QGGMRF()]
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


    Examples
    --------
    .. code-block:: python


        import numpy as np
        import matplotlib.pyplot as plt
        from phasetorch.spectrum import als_bm832,omp_spec_cali,LassoReg,RidgeReg
        from phasetorch._refindxdata import calculate_beta_vs_E
        from phasetorch._utils import get_wavelength

        if __name__ == '__main__': #Required for parallel compute

            # ALS
            energies, sp_als =als_bm832()    #Generate dictionary source spectrum
            sp_als/= np.trapz(sp_als,energies) # normalized spectrum response.
            x, y = np.meshgrid(np.arange(0, 128.0)*0.001, np.arange(0, 128.0)*0.001) #Grid shape 128 x 128. Pixel 0.001mm.
            projs = np.zeros_like(x)[np.newaxis] #Ground-truth projections. Unitary dimension indicates one view.
            mask = (0.032**2 - (x-0.064)**2) > 0 #Mask for non-zero projections of 0.032mm cylinder.
            projs[0][mask] = 2*np.sqrt((0.032**2 - (x-0.064)**2)[mask]) #Generate path lengths.

            # Spectrum Calibration with two different materials.
            beta_vs_energy = np.array([calculate_beta_vs_E(2.702, 'Al', energies).astype('float32'),
                             calculate_beta_vs_E(6.11, 'V', energies).astype('float32')])
            beta_projs =projs[np.newaxis,np.newaxis,:,:,:]*beta_vs_energy[:,:, np.newaxis, np.newaxis, np.newaxis]
            wnum = 2 * np.pi / get_wavelength(energies)

            # Poly-energic
            polyE_projs = np.trapz(np.exp(-2*wnum*beta_projs.transpose((0,2,3,4,1)))*sp_als,energies, axis=-1)

            # Add noise
            Trans_noisy = polyE_projs  + np.sqrt(polyE_projs)*0.1*0.01*np.random.standard_normal(size=polyE_projs.shape)
            Trans_noisy = np.clip(Trans_noisy,0,1)

            estimated_spec, errs, coef = dictSE(signal = Trans_noisy[:, :,63:64,:],
                                                     energies=energies,
                                                     beta_projs=beta_projs[:,:,:,63:64,:],
                                                     spec_dict=np.array([np.roll(sp_als,i) for i in range(500)]).T,
                                                     sparsity=10,
                                                     optimizor=LassoReg(l1=0.001),
                                                     tol=1e-06,
                                                     return_component=True)
            plt.plot( energies,sp_als)
            plt.plot( energies,estimated_spec)
    """

    S = []
    signal = signal.reshape((-1, 1))
    if signal_weight is None:
        signal_weight = np.ones(signal.shape)
    signal_weight = signal_weight.reshape((-1, 1))
    FDS = np.zeros((len(signal), 0))
    e = signal.copy()
    y = signal.copy()
    omega = np.zeros((spec_dict.shape[1], 1))
    errs = []
    err_list = []
    cost_list = []
    ICD_cost_list = []
    estimated_spec_list = []
    criteria_list = []

    selected_spec_mask = np.zeros(spec_dict.shape[1], dtype=bool)

    # wavelengths = get_wavelength(energies)
    # wnum = 2 * np.pi / wavelengths
    # forward_mat = np.exp(-2 * wnum * beta_projs.transpose((0, 2, 3, 4, 1))).reshape((-1, len(energies)))

    if verbose > 0:
        print(forward_mat.shape)
        print(np.diag(signal_weight).shape)

    # Pre-calculate matrices
    ysq = np.sum(y * signal_weight * y)
    y_F = ((y * signal_weight).T @ forward_mat).reshape((-1, 1))
    y_F_Dk = np.trapz(y_F * spec_dict, energies, axis=0)
    Fsq = forward_mat.T @ np.diag(signal_weight.flatten()) @ forward_mat

    D_Fsq = np.trapz(spec_dict.T[:, :, np.newaxis] * Fsq[np.newaxis, :, :], energies, axis=1)
    if verbose > 0:
        print('D_Fsq shape:', D_Fsq.shape)
    FDsq = np.trapz(D_Fsq * spec_dict.T, energies, axis=1)
    if verbose > 0:
        print('FDsq shape:', FDsq.shape)
    FDK = np.trapz(forward_mat[:, :, np.newaxis] * spec_dict[np.newaxis, :, :], energies, axis=1).reshape(
        (-1, spec_dict.shape[-1]))
    if verbose > 0:
        print('FDK shape:', FDK.shape)

    while len(S) < sparsity and np.linalg.norm(e) > tol:
        print('\nIteration %d:' % (len(S) + 1))

        # Find new index.
        if len(S) == 0:
            yhat = np.zeros(signal.shape)

        # Compute required matrices
        y_yhat = np.sum(y * signal_weight * yhat)
        yhat_sq = np.sum(yhat * signal_weight * yhat)
        yhat_F = ((yhat * signal_weight).T @ forward_mat).reshape((-1, 1))
        yhat_F_Dk = np.trapz(yhat_F * spec_dict, energies, axis=0)

        rho1 = y_yhat + FDsq - y_F_Dk - yhat_F_Dk
        rho2 = yhat_sq + FDsq - 2 * yhat_F_Dk

        # Calculate \beta_k.
        if len(S) == 0:
            beta = np.zeros(rho1.shape)
        else:
            beta = rho1 / rho2
        beta_mask = (beta >= 1)

        # Substitute \beta to criteria function.
        criteria = np.zeros(beta.shape)
        criteria = ysq + beta ** 2 * yhat_sq + (1 - beta) ** 2 * FDsq - 2 * beta * y_yhat - 2 * (
                    1 - beta) * y_F_Dk + 2 * beta * (1 - beta) * yhat_F_Dk
        criteria = np.ma.array(criteria, mask=np.logical_or(selected_spec_mask, beta_mask))

        criteria_list.append(criteria)
        if criteria.mask.all():
            break
        k = [np.argmin(criteria)]
        if verbose > 0:
            print(k)
            print(beta)
            print(criteria)

        # Build new support
        omega[S, 0] = omega[S, 0] * beta[k[0]]
        omega[k[0], 0] = 1 - beta[k[0]]
        S = S + k
        print(S)
        selected_spec_mask[k] = True
        FDk = np.trapz(forward_mat[:, :, np.newaxis] * spec_dict[np.newaxis, :, k], energies, axis=1)
        FDS = np.concatenate([FDS, FDk], axis=1)

        # Find best coefficient with new support given solver.
        Omega_init = omega[S, 0]
        e = signal - FDS @ omega[S]
        omega[S, 0] = optimizor.solve(FDS, signal, weight=signal_weight, Omega_init=Omega_init, e_init=e,
                                      spec_dict=spec_dict[:, S])
        ICD_cost_list.append(optimizor.cost_list)
        # Compute new residual
        e = signal - FDS @ omega[S]
        yhat = FDS @ omega[S]
        err_list.append(e)
        errs.append(np.sqrt(np.mean(e ** 2 * signal_weight)))

        cost = optimizor.cost()
        cost_list.append(cost)

        print('e:', np.sqrt(np.mean(e ** 2 * signal_weight)))
        print('cost:', cost)
        optimizor._set_mbi(np.argmax(omega[S, 0].flatten()))
        #optimizor.set_mbi(len(S))
        estimated_spec = spec_dict @ omega
        estimated_spec_list.append(estimated_spec)

    if return_component:
        return estimated_spec_list, errs, omega, S, cost_list, err_list, criteria_list, ICD_cost_list
    else:
        return estimated_spec, omega



def cal_fw_mat(solid_vol_masks, materials, fw_projector):
    """Calculate the forward matrix of multiple solid objects combination with given forward projector.

    Parameters
    ----------
    solid_vol_masks: A list of 3D numpy.ndarray.
        Each mask in the list represents a solid pure object.
    materials: A list of dictionary corresponding to solid_vol_masks.
        Each dictionary contains {chemical formula, density(g/cc)} for the corresponding solid pure object.
    fw_projector: A numpy class.
        Forward projector with a function forward(3D mask, {chemical formula, density(g/cc)}).

    Returns
    -------
    spec_fw_mat: A numpy.ndarray.
        Forward matrix of spectral estimation.

    """
    return 1


def binwised_spec_cali(signal, energies, x_init, h_init, beta_projs, 
                       B, h_min, energies_weight=None,beta=1, c=0.01, tol=1e-6,max_iter=200, stop_iter=1e-11,
                       s_update_threshold=1,j_update_threshold=None):
    """
    
    Parameters
    ----------
    signal
    energies
    x_init
    h_init
    beta_projs
    B
    h_min
    beta
    c
    tol

    Returns
    -------

    """
    if energies_weight is None:
        energies_weight = np.ones(energies.shape)
    W = np.diag(energies_weight)
    signal = signal.reshape((-1, 1))
    Ne = len(x_init)
    Nc = len(h_init)
    wavelengths = get_wavelength(energies)
    wnum = 2 * np.pi / wavelengths
    F = np.exp(-2 * wnum * beta_projs.transpose((0, 2, 3, 4, 1))).reshape((-1, len(energies)))
    m, n = np.shape(F)

    x = x_init.copy().reshape((-1,1))

    if j_update_threshold is None:
        j_update_threshold = 1/n
    h = h_init.copy().reshape((-1,1))
    e = signal - F@W @ (x + B @ h)
    iteration=0
    legend=[]
    fig,ax = plt.subplots(figsize=(24,12))
    cost_x_list=[]
    cost_rho_list=[]
    sp_list = []
 
    while np.linalg.norm(e) > tol and iteration<max_iter:
        print()
        total_abs_update = 0
        total_update = 0
        if iteration % int(max_iter*0.1)==0:
            ax.plot(energies, x+B@h)
            legend.append('iter:%d'%iteration)
        iteration+=1
        # Find index to perform direct substitution

        if len(sp_list) ==0:
            j = np.argmax(x[:,0]*energies_weight)
        else:
            mask = np.ones(energies_weight.shape)
            mask[j] = 0
            j = np.argmax(np.abs((x[:,0]-pre_x[:,0]))*energies_weight*mask)
        pre_x = x.copy()
        permuted_ind = np.random.permutation(Ne)
        # Update x serially
        for s in permuted_ind:
            if s == j:
                continue
            delta_1 = []
            delta_2 = []
            dsj = np.zeros(len(x))
            dsj[s]=1/energies_weight[s]
            dsj[j]=-1/energies_weight[j]
            if s > 0:
                we = (energies_weight[s]**2*np.abs(energies[s]-energies[s-1]))
                dd = (x[s] - x[s - 1])/np.abs(energies[s]-energies[s-1])
                dws = dsj[s]-dsj[s-1]
                if np.abs(dd) <1e-8:
                    bs1 = beta / 2/we 
                else:
                    bs1 = beta * np.clip(dd, -c, c) / (2 * dd)/we
                delta_1.append(bs1*dd*dws)
                delta_2.append(bs1*dws**2)

            if s < Ne - 1:
                we = (energies_weight[s]**2*np.abs(energies[s]-energies[s + 1]))
                dd = (x[s] - x[s + 1])/np.abs(energies[s]-energies[s + 1])
                dws = dsj[s]-dsj[s+1]
                if np.abs(dd) <1e-8:
                    bs2 = beta / 2/we 
                else:
                    bs2 = beta * np.clip(dd, -c, c) / (2 * dd)/we
                delta_1.append(bs2*dd*dws)
                delta_2.append(bs2*dws**2)

            if j > 0:
                we = (energies_weight[j]**2*np.abs(energies[j] - energies[j - 1]))
                dd = (x[j - 1] - x[j])/np.abs(energies[j] - energies[j - 1])
                dws = dsj[j-1]-dsj[j]
                if np.abs(dd) <1e-8:
                    bj1 = beta / 2 /we
                else:
                    bj1 = beta * np.clip(dd, -c, c) / (2 * dd)/we
                delta_1.append(bj1*dd*dws)
                delta_2.append(bj1*dws**2)

            if j < Ne - 1:
                we = (energies_weight[j]**2*np.abs(energies[j]-energies[j + 1]))
                dd = (x[j + 1] - x[j])/np.abs(energies[j] - energies[j + 1])
                dws = dsj[j+1]-dsj[j]
                if np.abs(dd) <1e-8:
                    bj2 = beta / 2 /we
                else:
                    bj2 = beta * np.clip(dd, -c, c) / (2 * dd)/we
                delta_1.append(bj2*dd*dws)
                delta_2.append(bj2*dws**2)
            Fsj = F[:, s:s + 1] - F[:, j:j + 1]
            theta_x_1 = -e.T @ Fsj / m + 2 * np.sum(np.array(delta_1))
            theta_x_2 = Fsj.T @ Fsj / m + 2 * np.sum(np.array(delta_2))
            Tmin = max(-x[j]*energies_weight[j]*j_update_threshold,-x[s]*energies_weight[s]*s_update_threshold)
            Tmax = min(x[s]*energies_weight[s]*s_update_threshold,x[j]*energies_weight[j]*j_update_threshold)

            
            update_alpha = np.clip(-theta_x_1 / theta_x_2, Tmin, Tmax).reshape((1,))
            total_abs_update +=np.abs(update_alpha)
            total_update +=update_alpha
            x[s] += update_alpha/energies_weight[s]
            x[j] -= update_alpha/energies_weight[j]
            e -= Fsj * update_alpha
            
        print('A1. Iter:', iteration,'Max index',j, 'Updated Err:', np.linalg.norm(e),
              'True Err:',np.linalg.norm(signal - F@W @ (x + B @ h)),'W (x+Bh)=',np.sum(W@(x+B@h)))
        cost_x,cost_rho = binwised_spec_cali_cost(signal,x,h,F,W,B,beta,c,energies)
        cost_x_list.append(cost_x)
        cost_rho_list.append(cost_rho)
        print('A. iter:',iteration,'cost_x:',cost_x,'cost_rho:',cost_rho)
        for s in range(Nc):
            delta_1 = []
            delta_2 = []
            if s == j:
                continue
            dsj = np.zeros(len(x))
            dsj[s]=1/energies_weight[s]
            dsj[j]=1/energies_weight[j]                
            if j > 0:
                we = (energies_weight[j]**2*np.abs(energies[j] - energies[j - 1]))
                dd = (x[j - 1] - x[j])/np.abs(energies[j] - energies[j - 1])
                dws = dsj[j-1]-dsj[j]
                if np.abs(dd) <1e-8:
                    bj1 = beta / 2/we
                else:
                    bj1 = beta * np.clip(dd, -c, c) / (2 * dd)/we
                delta_1.append(bj1*dd*dws)
                delta_2.append(bj1*dws**2)
            if j < Ne - 1:
                we = (energies_weight[j]**2*np.abs(energies[j]-energies[j + 1]))
                dd = (x[j + 1] - x[j])/np.abs(energies[j] - energies[j + 1])
                dws = dsj[j+1]-dsj[j]
                if np.abs(dd) <1e-8:
                    bj2 = beta / 2/we
                else:
                    bj2 = beta * np.clip(dd, -c, c) / (2 * dd)/we
                delta_1.append(bj2*dd*dws)
                delta_2.append(bj2*dws**2)
            FB = F@ W @ B[:, s:s + 1]
            FBj = FB - F[:, j:j + 1]
            theta_h_1 = -e.T @ FBj / m + 2 * np.sum( np.array(delta_1))
            theta_h_2 = FBj.T @ FBj / m + 2 * np.sum(np.array(delta_2))
            Tmin = max(-x[j]*energies_weight[j]*j_update_threshold,h_min[s] - h[s])
            Tmax = x[j]*energies_weight[j]*j_update_threshold
            update_h = np.clip(-theta_h_1 / theta_h_2, Tmin, Tmax).reshape((1,))
            h[s] += update_h
            x[j] -= update_h/energies_weight[j]
            total_abs_update +=np.abs(update_h)
            total_update +=update_h

            e -= FBj * update_h
        cost_x,cost_rho_s = binwised_spec_cali_cost(signal,x,h,F,W,B,beta,c,energies)
        cost_x_list.append(cost_x)
        cost_rho_list.append(cost_rho_s)
        sp_list.append(x + B @ h)
        print('B1. Iter:', iteration,'Max index',j, 'Updated Err:', np.linalg.norm(e),
              'True Err:',np.linalg.norm(signal - F@W @ (x + B @ h)),'W (x+Bh)=',np.sum(W@(x+B@h)))
        print('B. iter:',iteration,'cost_x:',cost_x,'cost_rho:',cost_rho)
        print('Total Update:',total_update, 'Total abs update:',total_abs_update )
        if total_abs_update < stop_iter:
            break
    ax.plot(energies, x+B@h)
    legend.append('iter:%d'%iteration)
    plt.legend(legend)

    return x, h, cost_x_list, cost_rho_list, sp_list


