def trapz_weight(x, axis=-1):
    """Modified from numpy.trapz. 
       Return weights for y to integrate along the given axis using the composite trapezoidal rule.
    """
    x = asanyarray(x)
    if x.ndim == 1:
        d = np.diff(x)
        # reshape to correct shape
        shape = [1]
        shape[axis] = d.shape[0]
        d = d.reshape(shape)
    else:
        d = np.diff(x, axis=axis)
    nd = 1
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    d = np.insert(d,0,0)
    d = np.insert(d,len(d),0)
    d = (d[tuple(slice1)] + d[tuple(slice2)]) / 2.0
    return d

def plot_est_spec(energies, weights_list, coef_, method, src_fltr_info_dict, scint_info_dict, S, mutiply_coef=True,save_path=None):
    plt.figure(figsize=(16,12))
    sd_info=[sfid+sid for sfid in src_fltr_info_dict for sid in scint_info_dict]
    est_sp = weights_list@ coef_
    est_legend = ['%s estimated spectrum'%method]
    plt.plot(energies,est_sp)
    plt.title('%s Estimated Spectrum.\n $|\omega|_1=%.3f$'%(method, np.sum( coef_)))

    for i in S:
        if mutiply_coef:
            plt.plot(energies,weights_list.T[i]* coef_[i])
            eneg_ind = np.argmax(weights_list.T[i])
            plt.text(energies[eneg_ind], weights_list.T[i,eneg_ind]* coef_[i]*1.05, r'%.3f'% coef_[i],\
            horizontalalignment='center', verticalalignment='center')
        else:
            plt.plot(energies,weights_list.T[i], alpha=0.2)
            eneg_ind = np.argmax(weights_list.T[i])
            plt.text(energies[eneg_ind], weights_list.T[i,eneg_ind]*1.05, r'%.3f'% coef_[i],\
            horizontalalignment='center', verticalalignment='center')
        est_legend.append('%.2f mm %s, %.2f mm %s'%(sd_info[i][1],sd_info[i][0],sd_info[i][3],sd_info[i][2]))
    plt.legend(est_legend,fontsize=10)
    if save_path is not None:
        plt.savefig(save_path)

        
def plot_est_spec_versa(energies, weights_list, coef_, method, spec_info_dict, S, mutiply_coef=True,save_path=None):
    plt.figure(figsize=(16,12))
    est_sp = weights_list@ coef_
    est_legend = ['%s estimated spectrum'%method]
    plt.plot(energies,est_sp)
    plt.title('%s Estimated Spectrum.\n $|\omega|_1=%.3f$'%(method, np.sum( coef_)))

    for i in S:
        if mutiply_coef:
            plt.plot(energies,weights_list.T[i]* coef_[i],label=spec_info_dict[i])
            eneg_ind = np.argmax(weights_list.T[i])
            plt.text(energies[eneg_ind], weights_list.T[i,eneg_ind]* coef_[i]*1.05, r'%.3f'% coef_[i],\
            horizontalalignment='center', verticalalignment='center')
        else:
            plt.plot(energies,weights_list.T[i], alpha=0.2,label=spec_info_dict[i])
            eneg_ind = np.argmax(weights_list.T[i])
            plt.text(energies[eneg_ind], weights_list.T[i,eneg_ind]*1.05, r'%.3f'% coef_[i],\
            horizontalalignment='center', verticalalignment='center')
    plt.legend(fontsize=10)
    if save_path is not None:
        plt.savefig(save_path)

