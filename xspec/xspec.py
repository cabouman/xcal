"""
Author: Wenrui Li
Email: li3120@purdue.edu
Description:
    This script includes dictionary learning algorithm to do spectrum calibration.

Date: Dec 1st, 2022.
"""


import numpy as np
from scipy.signal import butter,filtfilt,find_peaks

from xspec._utils import get_wavelength


def matching_pursuit(mp,mptype='omp',gamma=0.9,topk=1):
    if mptype=='omp':
        k = np.argmax(mp)
        return [k]
    elif mptype=='domp':
        max_mp = np.max(mp)
        index_list, = np.nonzero(mp>gamma*max_mp)
        return index_list.tolist()
    elif mptype=='gomp':
        idx = np.argpartition(mp, -topk)[-topk:]
        print(idx)
        return idx.tolist()


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
        
    def set_mbi(self, mbi):
        self.mbi = mbi
        
    def solve(self, X, y, spec_dict=None):
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
        beta = np.ones(n)/n
        c = self.c
        print('%d Dictionary'%n, beta)
        e = y.reshape((-1, 1))-np.mean(X,axis=-1,keepdims=True)
        l = spec_dict.shape[0]
        spec_beta = spec_dict @ beta.reshape((-1, 1))

        for k in range(self.max_iter):
            permuted_ind = np.arange(n)
            permuted_ind = permuted_ind[permuted_ind!=self.mbi]
            tol_update = 0
            for i in permuted_ind:
                beta_tmp = beta[i]
                beta_mbi = beta[self.mbi]

                if np.abs(beta[i])<1e-9:
                    b1 = self.l_star/2
                else:
                    b1 = self.l_star*np.clip(beta[i],-c,c)/(2*beta[i])

                if np.abs(beta_mbi)<1e-9:
                    b2 = self.l_star/2
                else:
                    b2 = self.l_star*np.clip(-beta_mbi,-c,c)/(-2*beta_mbi)
                theta_1 = -e.T @ (X[:, i:i + 1]-X[:,self.mbi:self.mbi+1]) / m + 2*b1 * beta[i] -2*b2*beta_mbi
                theta_2 = (X[:, i:i + 1]-X[:,self.mbi:self.mbi+1]).T @ (X[:, i:i + 1]-X[:,self.mbi:self.mbi+1]) / m + 2 * b1 +2*b2
                update = -theta_1 / theta_2
                # Calculate threshold
                spec_dict_diff = spec_dict[:, i:i + 1]-spec_dict[:, self.mbi:self.mbi + 1]
                pos_spec_mask=(spec_dict_diff>1e-10).reshape((l,))
                neg_spec_mask=(spec_dict_diff<-1e-10).reshape((l,))
                if np.sum(pos_spec_mask)==0:
                    T1=-np.inf
                else:
                    T1 = max(-spec_beta[pos_spec_mask]/spec_dict_diff[pos_spec_mask]+beta_tmp)
                if np.sum(neg_spec_mask) == 0:
                    T2 = np.inf
                else:
                    T2 = min(-spec_beta[neg_spec_mask]/spec_dict_diff[neg_spec_mask]+beta_tmp)
                beta[i] = np.clip(beta_tmp + update, T1, T2)
                beta[self.mbi] = 1 - np.sum(beta[permuted_ind])
                spec_beta = spec_beta + (spec_dict[:, i:i + 1]-spec_dict[:,self.mbi:self.mbi+1]) * (beta[i] - beta_tmp)

                tol_update += np.abs(beta[i]-beta_tmp)
                e = e - (X[:, i:i + 1]-X[:,self.mbi:self.mbi+1]) * (beta[i]-beta_tmp)
            
            if tol_update < self.threshold:
                print('Stop at iteration:', k,'  Total update:', tol_update)
                break
                
        
        print('mbi, beta_mbi:',self.mbi,beta[self.mbi])
        print('beta',beta)
        return beta  


# Orthogonal match pursuit with different optimization models.
def omp_spec_cali(signal, energies, beta_projs, spec_dict, sparsity, optimizor,
                mp_type='omp', mp_gamma=1, mp_topk=1, tol=1e-6,
                return_component=False, normalized_output=False, verbose=0):
    """A spectral calibration algorithm using dictionary learning.

    This function requires users to provide a large enough spectrum dictionary to estimate the source spectrum.
    By specify sparsity, it will use a greedy algorithm to add selected spectrum to a support from the large dictionary.

    Parameters
    ----------
    signal : numpy.ndarray
        Transmission Data of size (#sets, #energies, #views, #rows, #columns). Should be the exponential term instead of the projection after taking negative log.
    energies : numpy.ndarray
        List of X-ray energies of a poly-energetic source in units of keV.
    beta_projs : numpy.ndarray
        A numpy array of volumetric values of absorption index of size (#sets, #energies, #views, #rows, #columns)
    spec_dict : numpy.ndarray
        The spectrum dictionary contains N, the number of column, different X-ray spectrum.
        The number of rows M, should be same as the length of energies, should be normalized to integrate to 1.
    sparsity : int
        The max number of nonzero coefficients.
    optimizor : Python object
        Should be one of the optimizors defined above. [RidgeReg(), LassoReg(), ElasticNetReg(), QGGMRF()]
    tol : float
        The stop threshold.
    return_component : bool
        If true return coefficient.
    normalized_output: bool
        If true the output estimate will be normalized to integrate to 1.

    Returns
    -------
    estimated_spec : numpy.ndarray
        The estimated source-spectrum.
    errs : List
        List of errors while increasing sparsity.
    beta : numpy.ndarray
        Coefficients for spectrum dictionary wite size (#spectrums, 1).



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

            estimated_spec, errs, coef = omp_spec_cali(signal = Trans_noisy[:, :,63:64,:],
                                                     energies=energies,
                                                     beta_projs=beta_projs[:,:,:,63:64,:],
                                                     spec_dict=np.array([np.roll(sp_als,i) for i in range(500)]).T,
                                                     sparsity=10,
                                                     optimizor=LassoReg(l1=0.001),
                                                     tol=1e-06,
                                                     return_component=True,
                                                     normalized_output=True)
            plt.plot( energies,sp_als)
            plt.plot( energies,estimated_spec)
    """
    
    S = []
    signal = signal.reshape((-1,1))
    DS = np.zeros((len(signal),0))
    e=signal.copy()
    beta = np.zeros((spec_dict.shape[1],1))
    errs=[]
    yFexp_list=[]
    err_list=[]
    estimated_spec_list = []
    m = np.zeros(spec_dict.shape[1],dtype=bool)

    wavelengths = get_wavelength(energies)
    wnum = 2 * np.pi / wavelengths
    Aexp = np.exp(-2*wnum*beta_projs.transpose((0,2,3,4,1))).reshape((-1,len(energies)))
    if verbose>0:
        print(Aexp.shape)
    Aexp2 = Aexp.T@Aexp
    DAexp2 = np.trapz(spec_dict.T[:,:,np.newaxis]*Aexp2[np.newaxis,:,:],energies,axis=1)
    if verbose>0:
        print('DAexp2 shape:',DAexp2.shape)
    FD2 = np.trapz(DAexp2*spec_dict.T,energies,axis=1)
    if verbose>0:
        print('FD2 shape:',FD2.shape)
    
    while len(S)<sparsity and np.linalg.norm(e)>tol:
        # Find new index
        err_list.append(e)
        yFexp = (e.T@Aexp).reshape((-1,1))
        mp=2*np.trapz(yFexp*spec_dict,energies,axis=0)-FD2/(len(S)+1)   
        mp = np.ma.array(mp, mask=m)
        k = matching_pursuit(mp,mp_type,mp_gamma,mp_topk)
        if verbose>0:
            print(k)
        if len(k)==0:
            break
        # Build new support
        S = S + k
        if verbose>0:
            print(S)
        
        m[k] = True
        #Dk = np.trapz(Aexp*spec_dict[:,k],energies).reshape((-1,1))
        Dk = np.trapz(Aexp[:,:,np.newaxis]*spec_dict[np.newaxis,:,k],energies,axis=1).reshape((-1,len(k)))

        DS=np.concatenate([DS,Dk],axis=1)
        # Find best coefficient with new support
        beta[S,0] = optimizor.solve(DS, signal, spec_dict[:,S])
        # Compute new residual
        e=signal - DS@beta[S]*len(S)/(len(S)+1)
        yFexp_list.append(yFexp)
        errs.append(np.sqrt(np.mean(e**2)))
        optimizor.set_mbi(np.argmax(beta[S,0].flatten()))
        estimated_spec = spec_dict @ beta
        estimated_spec_list.append(estimated_spec)
    if normalized_output:
        estimated_spec /=np.trapz(estimated_spec.flatten(),energies)
    if return_component:
        return estimated_spec_list, errs, beta, S,yFexp_list,err_list
    else:
        return estimated_spec_list, errs