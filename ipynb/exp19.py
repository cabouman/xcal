# Basic Packages
import os
import numpy as np
import matplotlib.pyplot as plt
from xspec.paramSE import param_based_spec_estimate
from xspec.defs import *
from xspec._utils import *
import itertools
from xspec._utils import nested_list
import spekpy as sp  # Import SpekPy
import argparse

if __name__ == '__main__':

    filename = __file__.split('.')[0]
    os.makedirs('./output_exp19/', exist_ok=True)
    os.makedirs('./output_exp19/log/', exist_ok=True)
    os.makedirs('./output_exp19/res/', exist_ok=True)

    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='Parse input parameters.')

    # Add arguments
    parser.add_argument('--num_datasets', default=2, type=int, help='Number of datasets.')
    parser.add_argument('--dataset_ind', default=0, type=int, help='Dataset index.')

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments in your script like this:
    num_datasets = args.num_datasets
    dataset_ind = args.dataset_ind

    src_spec_list = []
    src_info = []
    simkV_list = np.linspace(30, 200, 18, endpoint=True).astype('int')
    max_simkV = max(simkV_list)
    energies = np.linspace(1, max_simkV, max_simkV)
    print('\nRunning demo script (1 mAs, 100 cm)\n')
    for simkV in simkV_list:
        for th in [12]:
            s = sp.Spek(kvp=simkV + 1, th=th, dk=1, char=True)  # Create the spectrum model
            k, phi_k = s.get_spectrum(edges=True)  # Get arrays of energy & fluence spectrum
            src_info.append((simkV,))
            src_spec = np.zeros((max_simkV))
            src_spec[:simkV] = phi_k[::2]
            src_spec_list.append(src_spec)

    print('\nFinished!\n')

    # Scintillator model
    scint_params = [
        # {'formula': 'CsI', 'density': 4.51, 'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1),
        #  'thickness_bound': (0.02, 0.5)},
        # {'formula': 'Gd3Al2Ga3O12', 'density': 6.63,
        #  'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound': (0.02, 0.5)},
        # {'formula': 'Lu3Al5O12', 'density': 6.73,
        #  'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound': (0.02, 0.5)},
        # {'formula': 'CdWO4', 'density': 7.9, 'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1),
        #  'thickness_bound': (0.02, 0.5)},
        # {'formula': 'Y3Al5O12', 'density': 4.56,
        #  'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound': (0.02, 0.5)},
        # {'formula': 'Bi4Ge3O12', 'density': 7.13,
        #  'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound': (0.02, 0.5)},
        {'formula': 'Gd2O2S', 'density': 7.32, 'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1),
         'thickness_bound': (0.02, 0.5)}
    ]

    data = read_mv_hdf5('../sim_data/sim_2v2f1s_dataset.hdf5')
    signal_train_list = [d['measurement'] for d in data][dataset_ind:dataset_ind + num_datasets]
    spec_F_train_list = [d['forward_mat'] for d in data][dataset_ind:dataset_ind + num_datasets]

    num_src = 2
    src_vol_bound = Bound(lower=30.0, upper=200.0)
    voltage_list = [100.0, 160.0]
    Src_config = [src_spec_params(energies, simkV_list, src_spec_list, src_vol_bound, voltage_list[i], require_gradient=False) for i in range(num_src)]
    # Src_config = [src_spec_params(energies, simkV_list, src_spec_list, src_vol_bound) for i in range(num_src)]

    num_fltr = 2
    psb_fltr_mat_list = [[Material(formula='Al', density=2.702)], [Material(formula='Cu', density=8.92)]]
    fltr_th_bound = [Bound(lower=0.0, upper=3.0), Bound(lower=0.0, upper=3.0)]

    Fltr_config = [fltr_resp_params(psb_fltr_mat_list[i], fltr_th_bound[i], 2.0, require_gradient=False) for i in range(num_fltr)]
    # Fltr_config = [fltr_resp_params(psb_fltr_mat_list[i], fltr_th_bound[i]) for i in range(num_fltr)]

    psb_scint_mat = [Material(formula=scint_p['formula'], density=scint_p['density']) for scint_p in scint_params]
    scint_th_bound = Bound(lower=0.01, upper=0.5)
    Scint_config = [scint_cvt_func_params(psb_scint_mat, scint_th_bound)]

    model_combination = [Model_combination(src_ind=0, fltr_ind_list=[0], scint_ind=0),
                         Model_combination(src_ind=1, fltr_ind_list=[0,1], scint_ind=0),
                        ]

    fltr_config_list = [[fc for fc in fcm.next_psb_fltr_mat()] for fcm in Fltr_config]
    scint_config_lsit = [[sc for sc in scm.next_psb_scint_mat()] for scm in Scint_config]
    model_params_list = list(itertools.product(*fltr_config_list, *scint_config_lsit))
    print('Total number of models:', len(model_params_list))
    model_params_list = [nested_list(l, [len(d) for d in [fltr_config_list, scint_config_lsit]]) for l in
                         model_params_list]

    learning_rate = 0.01
    optimizer_type = 'NNAT_LBFGS'
    loss_type = 'wmse'

    savefile_name = 'case_mvmf_%d_%d_%s_%s_lr%.0e' % (num_datasets, dataset_ind, optimizer_type, loss_type, learning_rate)

    res = param_based_spec_estimate(energies,
                                    signal_train_list,
                                    spec_F_train_list,
                                    Src_config,
                                    Fltr_config,
                                    Scint_config,
                                    model_combination,
                                    learning_rate=learning_rate,
                                    iterations=200,
                                    tolerance=1e-8,
                                    optimizer_type=optimizer_type,
                                    loss_type=loss_type,
                                    logpath=None,#'./output_exp19/log/%s'%savefile_name,
                                    num_processes=1,
                                    return_history=False)

    np.save('./output_exp19/res/%s.npy'%savefile_name, res, allow_pickle=True)