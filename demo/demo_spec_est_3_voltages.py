# Basic Packages
import os

import matplotlib.pyplot as plt
import numpy as np
from xspec.paramSE import param_based_spec_estimate
from xspec.defs import *
from xspec._utils import *
import itertools
from xspec._utils import nested_list
import spekpy as sp  # Import SpekPy
import argparse
from demo_utils import gen_datasets_3_voltages

if __name__ == '__main__':
    filename = __file__.split('.')[0]
    os.makedirs('./output_3_source_voltages/', exist_ok=True)
    os.makedirs('./output_3_source_voltages/log/', exist_ok=True)
    os.makedirs('./output_3_source_voltages/res/', exist_ok=True)


    src_spec_list = []
    src_info = []
    simkV_list = np.linspace(30, 160, 14, endpoint=True).astype('int')
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

    data = gen_datasets_3_voltages()
    num_src_v = len(data)

    signal_train_list = [d['measurement'] for d in data]
    spec_F_train_list = [d['forward_mat'] for d in data]

    # Number of datasets
    num_dataset = len(signal_train_list)

    voltage_list = [80.0, 130.0, 180.0]
    simkV_list = np.linspace(30, 200, 18, endpoint=True).astype('int')
    max_simkV = max(simkV_list)
    anode_angle = 12
    # Pixel size in mm units.
    rsize = 0.01  # mm

    # Energy bins.
    energies = np.linspace(1, max_simkV, max_simkV)

    # Use Spekpy to generate a source spectra dictionary.
    src_spec_list = []
    print('\nRunning demo script (1 mAs, 100 cm)\n')
    for simkV in simkV_list:
        s = sp.Spek(kvp=simkV + 1, th=anode_angle, dk=1, mas=10, char=True)  # Create the spectrum model
        k, phi_k = s.get_spectrum(edges=True)  # Get arrays of energy & fluence spectrum
        phi_k = phi_k * ((rsize / 10) ** 2)

        src_spec = np.zeros((max_simkV))
        src_spec[:simkV] = phi_k[::2]
        src_spec_list.append(src_spec)

    print('\nFinished!\n')

    # Use class Source to store a source's paramter.
    # optimize=False means do not optimize source voltage.
    src_vol_bound = Bound(lower=30.0, upper=200.0)
    Src_config = [Source(energies, simkV_list, src_spec_list, src_vol_bound, voltage=vv, optimize_voltage=False) for vv in
                  voltage_list]
    # Src_config = [Source(energies, simkV_list, src_spec_list, src_vol_bound)]

    # There is 1 filter with 2 possible materials.
    num_fltr = 1
    psb_fltr_mat_comb = [Material(formula='Al', density=2.702), Material(formula='Cu', density=8.92)]
    fltr_th_bound = Bound(lower=0.0, upper=10.0)
    Fltr_config = [Filter(psb_fltr_mat_comb, fltr_th_bound) for i in range(num_fltr)]

    # 7 possible scintillators
    scint_params = [
        {'formula': 'CsI', 'density': 4.51},
        {'formula': 'Gd3Al2Ga3O12', 'density': 6.63},
        {'formula': 'Lu3Al5O12', 'density': 6.73},
        {'formula': 'CdWO4', 'density': 7.9},
        {'formula': 'Y3Al5O12', 'density': 4.56},
        {'formula': 'Bi4Ge3O12', 'density': 7.13},
        {'formula': 'Gd2O2S', 'density': 7.32}
    ]

    psb_scint_mat = [Material(formula=scint_p['formula'], density=scint_p['density']) for scint_p in scint_params]
    scint_th_bound = Bound(lower=0.01, upper=0.5)
    Scint_config = [Scintillator(psb_scint_mat, scint_th_bound)]

    model_combination = [Model_combination(src_ind=i, fltr_ind_list=[0], scint_ind=0) for i in range(num_dataset)]

    learning_rate = 1.0
    optimizer_type = 'NNAT_LBFGS'
    loss_type = 'mse'

    savefile_name = 'case_mv_%s_%s_lr%.0e' % (optimizer_type, loss_type, learning_rate)

    os.makedirs('./output/log/', exist_ok=True)

    res = param_based_spec_estimate(energies,
                                    signal_train_list,
                                    spec_F_train_list,
                                    Src_config,
                                    Fltr_config,
                                    Scint_config,
                                    model_combination,
                                    weight=None,
                                    learning_rate=learning_rate,
                                    max_iterations=200,
                                    stop_threshold=1e-6,
                                    optimizer_type=optimizer_type,
                                    loss_type=loss_type,
                                    logpath='./output_3_source_voltages/log/%s' % savefile_name,
                                    num_processes=8,
                                    return_history=False)

    cost_list = [r[1] for r in res]
    optimal_cost_ind = np.argmin(cost_list)
    best_res = res[optimal_cost_ind][2]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axs[i].plot(energies, best_res.src_spec_list[i]().data, '--', label='Estimated %d' % i)
        axs[i].set_ylim((0, 0.2e3))
        axs[i].legend()
    plt.savefig('./output_3_source_voltages/res/Est_source.png')

    # Creating a figure with 3 subplots
    ll = ['3 mm Al']

    plt.figure(2)
    plt.plot(energies, best_res.fltr_resp_list[0](energies).data, '--', label='Estimate')
    plt.ylim((0,1))
    plt.title(ll[0])
    plt.legend()
    plt.savefig('./output_3_source_voltages/res/Est_filter.png')

    plt.figure(3)
    plt.plot(energies, best_res.scint_cvt_list[0](energies).data, '--', label='Estimated')
    plt.legend()
    plt.savefig('./output_3_source_voltages/res/Est_scintillator.png')
