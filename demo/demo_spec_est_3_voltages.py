# Basic Packages
import os
import matplotlib.pyplot as plt
import numpy as np
from xspec.estimate import Estimate
from xspec.defs import Material
from xspec._utils import *
from xspec.models import *
import spekpy as sp  # Import SpekPy
from demo_utils import gen_datasets_3_voltages

if __name__ == '__main__':
    filename = __file__.split('.')[0]
    os.makedirs('./output_3_source_voltages/', exist_ok=True)
    os.makedirs('./output_3_source_voltages/log/', exist_ok=True)
    os.makedirs('./output_3_source_voltages/res/', exist_ok=True)

    # Generate simulated multi-polychromatic dataset using 3 different source voltages.
    data = gen_datasets_3_voltages()
    num_src_v = len(data)

    normalized_rads = [d['measurement'] for d in data]
    forward_matrices = [d['forward_mat'] for d in data]
    gt_sources = [d['source'] for d in data]
    gt_filter = data[0]['filter']
    gt_scint = data[0]['scintillator']

    # Set source's parameters.
    voltage_list = [80, 130, 180]
    max_simkV = max(voltage_list)
    reference_anode_angle = 11

    # Detector pixel size in mm units.
    dsize = 0.01  # mm

    # Energy bins.
    energies = np.linspace(1.5, max_simkV - 0.5, max_simkV-1)

    # Use Spekpy to generate a source spectra dictionary.
    takeoff_angles = np.linspace(5, 45, 11)
    src_spec_list = []
    for case_i, simkV in enumerate(voltage_list):
        for ta in takeoff_angles:
            # Generate the X-ray spectrum model with Spekpy for each voltage.
            s = sp.Spek(kvp=simkV, th=ta, dk=1, mas=1, char=True)
            k, phi_k = s.get_spectrum(edges=False)  # Retrieve energy bins and fluence spectrum [Photons cm^-2 keV^-1]

            # Adjust the fluence for the detector pixel area.
            phi_k = phi_k * ((dsize / 10) ** 2) # Convert pixel size from mm² to cm²

            # Initialize a zero-filled spectrum array with length max_simkV.
            src_spec = np.zeros(max_simkV - 1)
            src_spec[:simkV - 1] = phi_k  # Assign spectrum values starting from 1.5 keV

            # Add the processed spectrum for this voltage to the list.
            src_spec_list.append(src_spec)

    src_spec_list = np.array(src_spec_list)
    src_spec_list = src_spec_list.reshape((len(voltage_list), len(takeoff_angles), -1))

    sources = [Reflection_Source(voltage=(voltage, None, None), takeoff_angle=(25, 5, 45), single_takeoff_angle=True)
               for
               voltage in voltage_list]
    for src_i, source in enumerate(sources):
        source.set_src_spec_list(energies, src_spec_list, voltage_list, takeoff_angles)

    psb_fltr_mat = [Material(formula='Al', density=2.702), Material(formula='Cu', density=8.92)]
    filter_1 = Filter(psb_fltr_mat, thickness=(5, 0, 10))

    scint_params_list = [
        {'formula': 'CsI', 'density': 4.51},
        {'formula': 'Gd3Al2Ga3O12', 'density': 6.63},
        {'formula': 'Lu3Al5O12', 'density': 6.73},
        {'formula': 'CdWO4', 'density': 7.9},
        {'formula': 'Y3Al5O12', 'density': 4.56},
        {'formula': 'Bi4Ge3O12', 'density': 7.13},
        {'formula': 'Gd2O2S', 'density': 7.32}
    ]
    psb_scint_mat = [Material(formula=scint_p['formula'], density=scint_p['density']) for scint_p in scint_params_list]
    scintillator_1 = Scintillator(materials=psb_scint_mat, thickness=(0.25, 0.01, 0.5))

    spec_models = [[source, filter_1, scintillator_1] for source in sources]
    learning_rate = 0.02
    max_iterations = 5000
    stop_threshold = 1e-5
    optimizer_type = 'NNAT_LBFGS'

    savefile_name = 'case_mv_%s_lr%.0e' % (optimizer_type, learning_rate)

    os.makedirs('./output_3_source_voltages/log/', exist_ok=True)

    Estimator = Estimate(energies)
    for nrad, forward_matrix, concatenate_models in zip(normalized_rads, forward_matrices, spec_models):
        Estimator.add_data(nrad, forward_matrix, concatenate_models, weight=None)

    # Fit data
    Estimator.fit(learning_rate=learning_rate,
                  max_iterations=max_iterations,
                  stop_threshold=stop_threshold,
                  optimizer_type=optimizer_type,
                  loss_type='transmission',
                  logpath=None,
                  num_processes=7)
    res_spec_models = Estimator.get_spec_models()
    res_params = Estimator.get_params()

    # Function to print a section of the table
    def print_params(params):
        with (torch.no_grad()):
            for key, value in sorted(params.items()):
                if isinstance(value, tuple):
                    print(f"{key}: {value[0].numpy()}")
                else:
                    print(f"{key}: {value}")
            print()


    # Print the table sections
    print('Ground Truth Parameters:')
    for gt_source in gt_sources:
        print(gt_source.get_params())
    print(gt_filter.get_params())
    print(gt_scint.get_params())

    print('Estimated Parameters:')
    print_params(res_params)

    # Create a figure and axes objects for the subplot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each subplot
    for i in range(3):
        ax = axs[i]
        with torch.no_grad():
            ax.plot(energies, (gt_sources[i](energies) * gt_filter(energies) * gt_scint(energies)).numpy(),
                    label='Ground Truth')
            ax.plot(energies, (
                        res_spec_models[i][0](energies) * res_spec_models[i][1](energies) * res_spec_models[i][2](
                    energies)).numpy(), '--', label='Estimate')

        # Add legend
        ax.legend()

        # Add subplot title
        ax.set_title(f'{[80, 130, 180][i]} kV Spectral Response')

    # Add common title
    fig.suptitle('Comparison of Ground Truth and Estimate')

    # Show the plot
    plt.show()

