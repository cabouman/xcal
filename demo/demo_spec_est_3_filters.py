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
    os.makedirs('./output_3_filters/', exist_ok=True)
    os.makedirs('./output_3_filters/log/', exist_ok=True)
    os.makedirs('./output_3_filters/res/', exist_ok=True)

    # Initialize the ArgumentParser object
    parser = argparse.ArgumentParser(description='Parse input parameters.')

    # Add arguments
    parser.add_argument('--dataset_path', type=str, help='Dataset path.')
    # parser.add_argument('--num_fltr', type=int, help='Number of filters.')
    # parser.add_argument('--dataset_ind', type=int, help='Dataset index.')

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments in your script like this:
    dataset_path = args.dataset_path
    dataset_name = dataset_path.split('/')[-1].split('.')[0]

    # dataset_ind = args.dataset_ind

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



    # Generate filter response
    fltr_params = [
        {'formula': 'Al', 'density': 2.702, 'thickness_list': neg_log_space(vmin=0.1, vmax=6, num=10, scale=1),
         'thickness_bound': (0, 10)},
        {'formula': 'Cu', 'density': 8.92, 'thickness_list': neg_log_space(vmin=0.25, vmax=0.6, num=10, scale=1),
         'thickness_bound': (0, 2)},
    ]

    # Scintillator model
    scint_params = [
        {'formula': 'CsI', 'density': 4.51, 'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1),
         'thickness_bound': (0.02, 0.5)},
        {'formula': 'Gd3Al2Ga3O12', 'density': 6.63,
         'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound': (0.02, 0.5)},
        {'formula': 'Lu3Al5O12', 'density': 6.73,
         'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound': (0.02, 0.5)},
        {'formula': 'CdWO4', 'density': 7.9, 'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1),
         'thickness_bound': (0.02, 0.5)},
        {'formula': 'Y3Al5O12', 'density': 4.56,
         'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound': (0.02, 0.5)},
        {'formula': 'Bi4Ge3O12', 'density': 7.13,
         'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound': (0.02, 0.5)},
        {'formula': 'Gd2O2S', 'density': 7.32, 'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1),
         'thickness_bound': (0.02, 0.5)}
    ]

    data = read_mv_hdf5('../sim_data/sim_1v3f1s_dataset.hdf5')
    signal_train_list = [d['measurement'] for d in data]
    spec_F_train_list = [d['forward_mat'] for d in data]


    src_vol_bound = Bound(lower=30.0, upper=200.0)
    Src_config = [Source(energies, simkV_list, src_spec_list, src_vol_bound, 100.0, optimize=False)]
    # Src_config = [Source(energies, simkV_list, src_spec_list, src_vol_bound)]

    num_fltr = len(data)
    psb_fltr_mat_comb =[Material(formula='Al', density=2.702), Material(formula='Cu', density=8.92)]
    fltr_th_bound = Bound(lower=1.0, upper=10.0)
    Fltr_config = [Filter(psb_fltr_mat_comb, fltr_th_bound) for i in range(num_fltr)]

    psb_scint_mat = [Material(formula=scint_p['formula'], density=scint_p['density']) for scint_p in scint_params]
    scint_th_bound = Bound(lower=0.01, upper=0.5)
    Scint_config = [Scintillator(psb_scint_mat, scint_th_bound)]

    model_combination = [Model_combination(src_ind=0, fltr_ind_list=[i], scint_ind=0) for i in range(num_fltr)]

    learning_rate = 0.1
    optimizer_type = 'NNAT_LBFGS'
    loss_type = 'wmse'

    savefile_name = 'case_%s_%s_%s_lr%.0e' % (dataset_name, optimizer_type, loss_type, learning_rate)

    res = param_based_spec_estimate(energies,
                                    signal_train_list,
                                    spec_F_train_list,
                                    Src_config,
                                    Fltr_config,
                                    Scint_config,
                                    model_combination,
                                    learning_rate=learning_rate,
                                    max_iterations=200,
                                    stop_threshold=1e-6,
                                    optimizer_type=optimizer_type,
                                    loss_type=loss_type,
                                    logpath='./output_3_filters/log/%s'%savefile_name,
                                    num_processes=8,
                                    return_history=False)

    print()
    print('Ground Truth Parameter:')
    print('Source Voltage:', data[0]['src_config']['voltage'])
    print('Filter Material:',  [d['fltr_config']['fltr_mat_0_formula'] for d in data])
    print('Filter Thickness:',  [d['fltr_config']['fltr_mat_0_th'] for d in data], 'mm')
    print('Scintillator Material:', data[0]['scint_config']['scint_mat_formula'])
    print('Scintillator Thickness:', data[0]['scint_config']['scint_th'], 'mm')

    np.save('./output_3_filters/res/%s.npy'%savefile_name, res, allow_pickle=True)
