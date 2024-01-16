# Basic Packages
import os
import matplotlib.pyplot as plt
import numpy as np
from xspec import estimate
from xspec.defs import *
from xspec._utils import *
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

    # Number of datasets
    num_dataset = len(normalized_rads)

    # Set source's parameters.
    voltage_list = [80.0, 130.0, 180.0]
    simkV_list = np.linspace(30, 200, 18, endpoint=True).astype('int')
    max_simkV = max(simkV_list)
    reference_anode_angle = 11

    # Detector pixel size in mm units.
    dsize = 0.01  # mm

    # Energy bins.
    energies = np.linspace(1, max_simkV, max_simkV)

    # Use Spekpy to generate a source spectra dictionary.
    src_spec_list = []
    for simkV in simkV_list:
        s = sp.Spek(kvp=simkV + 1, th=reference_anode_angle, dk=1, mas=1, char=True)  # Create the spectrum model
        k, phi_k = s.get_spectrum(edges=True)  # Get arrays of energy & fluence spectrum
        phi_k = phi_k * ((dsize / 10) ** 2)

        src_spec = np.zeros((max_simkV))
        src_spec[:simkV] = phi_k[::2]
        src_spec_list.append(src_spec)

    # optimize_voltage=False means do not optimize source voltages.
    # Assign values to source_params dictionary
    source_params = {
        'num_voltage': len(voltage_list),
        'reference_voltages': simkV_list,
        'reference_anode_angle': reference_anode_angle,
        'reference_spectra': src_spec_list,
        'voltage_1': voltage_list[0],
        'voltage_2': voltage_list[1],
        'voltage_3': voltage_list[2],
        'voltage_1_range': (voltage_list[0]*0.95, voltage_list[0]*1.05),
        'voltage_2_range': (voltage_list[1]*0.95, voltage_list[1]*1.05),
        'voltage_3_range': (voltage_list[2]*0.95, voltage_list[2]*1.05),
        'anode_angle': None, # Initial Value
        'anode_angle_range': (5, 45),
        'optimize_voltage': True,
        'optimize_anode_angle': True,
        'source_voltage_indices': [1, 2, 3]  # Indices used source voltage for each radiograph
    }

    # Set filter parameters
    # There is 1 filter with 2 possible materials.
    psb_fltr_mat_comb = [Material(formula='Al', density=2.702), Material(formula='Cu', density=8.92)]
    # Assign values to filter_params dictionary
    filter_params = {
        'num_filter': 1,
        'possible_material': psb_fltr_mat_comb,
        'material_1': None,
        'thickness_1': 0.1,
        'thickness_1_range': (0.0, 10.0),
        'optimize': True,
        'filter_indices': [[1], [1], [1]]
    }

    # Set scintillator parameters
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
    # Assign values to scintillator_params dictionary
    scintillator_params = {
        'possible_material': psb_scint_mat,
        'material': None,
        'thickness': None,
        'thickness_range': (0.01, 0.5),
        'optimize': True
    }


    learning_rate = 0.02
    optimizer_type = 'NNAT_LBFGS'

    savefile_name = 'case_mv_%s_lr%.0e' % (optimizer_type, learning_rate)

    os.makedirs('./output_3_source_voltages/log/', exist_ok=True)

    res_params = estimate(energies, normalized_rads, forward_matrices, source_params, filter_params, scintillator_params,
                          weight=None,
                          weight_type='unweighted',
                          blank_rads=None,
                          learning_rate=learning_rate,
                          max_iterations=5000,
                          stop_threshold=1e-4,
                          optimizer_type=optimizer_type,
                          logpath=None,  #'./output_3_source_voltages/log/%s' % savefile_name,
                          num_processes=1,
                          return_all_result=False)

    # Define the data for each section
    source_data = [
        ["Source Parameters", "GT", "Estimated"],
        ["Voltage 1(kV)", 80, res_params['voltage_1']],
        ["Voltage 2(kV)", 130, res_params['voltage_2']],
        ["Voltage 3(kV)", 180, res_params['voltage_3']],
        ["Takeoff Angle(°)", 20, res_params['anode_angle']]
    ]

    filter_data = [
        ["Filter 1 Parameters", "GT", "Estimated"],
        ["material", "Al",  res_params['filter_1_mat'].formula],
        ["thickness(mm)", 3, res_params['filter_1_thickness']]
    ]

    scintillator_data = [
        ["Scintillator Parameters", "Parameter", "Value"],
        ["material", "CsI", res_params['scintillator_mat'].formula],
        ["thickness(mm)", 0.33, res_params['scintillator_thickness']]
    ]


    # Function to print a section of the table
    def print_section(data):
        # Determine the maximum width for each column
        col_widths = [max(len(str(row[i])) for row in data) for i in range(len(data[0]))]
        # Create a format string for each row
        row_format = " | ".join(["{:<" + str(width) + "}" for width in col_widths])

        # Print the header row
        print(row_format.format(*data[0]))
        print('-' * sum(col_widths))  # Separator

        # Print each data row
        for row in data[1:]:
            print(row_format.format(*row))
        print()  # Add a blank line after the section


    # Print the table sections
    print()
    print('Final Estimate Result:')
    print_section(source_data)
    print_section(filter_data)
    print_section(scintillator_data)
