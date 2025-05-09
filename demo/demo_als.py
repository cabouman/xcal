"""
Spectral Calibration Demo using Multi-Filtration Dataset

This script demonstrates how to perform model-based spectral calibration
using normalized transmission of known objects scanned under different filtration conditions.

Steps:
1. Load dataset (normalized radiograph/transmission + reconstruction) from HDF5 files in ../data/demo_xcal_data.
2. Perform circle detection and segmentation to obtain the mask of the homogenous object.
3. Compute the forward matrix using estimated masks.
4. Configure the X-ray system model.
5. Run spectral calibration.
6. Plot and display estimated spectra and fitted parameters.
"""

import os
import urllib.request
import tarfile
from pprint import pprint

import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

from xcal.chem_consts import get_lin_att_c_vs_E
from xcal import calc_forward_matrix
from xcal.chem_consts._periodictabledata import density
from xcal.chem_consts._als_utils import als_bm832, detect_outliers,only_center_mask
from xcal.phantom import segment_object
from xcal.defs import Material
from xcal.models import Filter, Scintillator
from xcal.estimate import Estimate
from demo_utils import Synchrotron_Source
import mbirjax

import torch

class fw_projector:
    """A class for forward projection using MBIRJAX."""

    def __init__(self, angles, num_channels, center_offset=0, delta_pixel=1):
        """
        Initializes the forward projector with specified geometric parameters.

        Parameters:
            angles (array): Array of projection angles.
            num_channels (int): Number of detector channels.
            delta_pixel (float, optional): Size of a pixel, defaults to 1.
        """
        self.angles = angles
        self.num_channels = num_channels
        self.delta_pixel = delta_pixel
        self.center_offset = center_offset
        self.sinogram_shape = (len(angles), 1, self.num_channels) # (num_views, num_rows, num_columns)

    def forward(self, mask):
        """
        Computes the projection of a given mask.

        Parameters:
            mask (numpy.ndarray): 3D mask of the object to be projected.

        Returns:
            numpy.ndarray: The computed projection of the mask.
        """
        ct_model_for_generation = mbirjax.ParallelBeamModel(self.sinogram_shape, self.angles)
        ct_model_for_generation.set_params(det_channel_offset=self.center_offset)

        # Print out model parameters
        ct_model_for_generation.print_params()
        projections = ct_model_for_generation.forward_project(mask) * self.delta_pixel
        return projections

if __name__ == '__main__':
    # --------- Step 0: Download ALS dataset ---------
    # Define the target folder and expected files
    data_dir = os.path.expanduser("../data/demo_xcal_data")
    expected_files = [
        "high_fltr_Al.h5", "high_fltr_Mg.h5", "high_fltr_Ti.h5", "high_fltr_V.h5",
        "low_fltr_Al.h5", "low_fltr_Mg.h5", "low_fltr_Ti.h5", "low_fltr_V.h5"
    ]

    # Check if all expected files exist
    if not os.path.isdir(data_dir) or not all(os.path.exists(os.path.join(data_dir, f)) for f in expected_files):
        print("Downloading demo_xcal_data.tgz...")
        url = "https://www.datadepot.rcac.purdue.edu/bouman/data/demo_xcal_data.tgz"
        download_path = os.path.expanduser("../data/demo_xcal_data.tgz")

        # Download the file
        urllib.request.urlretrieve(url, download_path)
        print("Download complete.")

        # Extract the archive
        print("Extracting files...")
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=os.path.dirname(data_dir))

        print("Extraction complete.")
    else:
        print("All demo calibration files already exist.")


    # --------- Step 1: Load data from HDF5 files ---------
    # Files contain: data_norm (normalized radiograph/transmission), recon (MBIRJAX reconstruction)
    print('Loading both low filtration and high filtration datasets.')
    filenames = glob.glob('../data/demo_xcal_data/*.h5')
    filenames.sort(reverse=True) # Reverse sort

    data_norm_list = []
    recon_list = []
    trans_mask_list = []
    for fname in filenames:
        print(fname)
        with h5py.File(fname, 'r') as f:
            data_norm_list.append(f['data_norm'][()])
            recon_list.append(f['recon'][()])
            # Dectect outliers, which are False in mask.
            outliers_mask = detect_outliers(data_norm_list[-1], window_size=51, threshold_std=0.1)
            # Use center points for each row.
            bg_mask = only_center_mask(data_norm_list[-1], 1400)
            mask = outliers_mask * bg_mask
            trans_mask_list.append(mask)

    # --------- Step 2: Circle detection and segmentation ---------
    # Detect circular regions from reconstructions and create binary masks
    # These masks will be used to model material thickness per ray

    # Set up min and max value for segmentation.
    vmin_list = [1, 1, 0.15, 0.05, 0.3, 0.05, 0.06, 0.05]
    vmax_list = [4, 2.2, 0.3, 0.5, 2.0, 1.0, 0.1, 0.2]

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))  # 2 rows, 4 columns
    axes = axes.flatten()  # Flatten row-wise

    recon_mask_list = []
    for i in range(8):
        ax = axes[i]
        img = recon_list[i][0]
        mask = segment_object(img, vmin_list[i], vmax_list[i], 60, 900)
        recon_mask_list.append(mask[np.newaxis, :, :])
        ax.imshow(mask * img, origin='lower', vmin=vmin_list[i], vmax=vmax_list[i])
        ax.set_aspect('equal')
        ax.set_title(f"{filenames[i].split('/')[-1][:-3]}")
        ax.axis('off')

    plt.suptitle('Reconstruction after segmentation with estimated mask', y=0.98, fontsize=20)
    plt.tight_layout()
    plt.show(block=False)
    plt.savefig('./output/als_masked_region.png')

    # --------- Step 3: Calculate forward matrix with estimated masks ---------
    # Use the masks and forward projector to calculate the path length of each material in each projection
    # Result is a list of forward matrices A of shape (num_measurements, num_energies)
    center_offset_list = [-31, -24, -35, -10, -68, -68, 128, 140]
    total_num_views = data_norm_list[0].shape[0]
    angles = -np.linspace(-0.5*np.pi, 1.5*np.pi, total_num_views, endpoint=True)

    energies, sp_als = als_bm832()
    energies, sp_als = energies[1:], sp_als[1:]

    # Scanned Homogeneous Rods
    # Density for homogenous material is stored in the imported density dictionary
    sample_mats = [fn.split('_')[-1][:-3] for fn in filenames]
    mat_density = [density[formula] for formula in sample_mats]
    lac_vs_E_list = [get_lin_att_c_vs_E(den, formula, energies) for den, formula in zip(mat_density, sample_mats)]
    # Use [recon_mask] is because calc_forward_matrix assumes multiple homogenous objects in 3D.
    spec_F_list = []
    for ri, recon_mask in enumerate(recon_mask_list):
        projector = fw_projector(angles[:total_num_views // 2:100], num_channels=data_norm_list[0].shape[-1], center_offset=center_offset_list[ri], delta_pixel=0.00065)
        spec_F = calc_forward_matrix([recon_mask_list[ri]], [lac_vs_E_list[ri]], projector)
        spec_F_list.append(spec_F)
    nrads_list = [nrad[:total_num_views // 2:100] for nrad in data_norm_list]
    trans_mask_list = [tm[:total_num_views // 2:100] for tm in trans_mask_list]

    # --------- Step 4: Configure the X-ray system model ---------
    # Define source, filters, detector, and geometry
    # Example: voltage, filter materials, scintillator, etc.
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

    learning_rate = 0.001
    max_iterations = 5000
    stop_threshold = 1e-4
    optimizer_type = 'Adam'
    Estimator_list = []

    source_1 = Synchrotron_Source(voltage=(100, None, None))
    source_1.set_src_spec_list(np.array([sp_als]), np.array([100]))

    filter_1 = Filter([Material(formula='Si', density=2.329)], thickness=(2.5, 0.1, 5))
    filter_2 = Filter([Material(formula='Al', density=2.7), Material(formula='Cu', density=8.96)
                       ], thickness=(10, 5, 15))
    scintillator_1 = Scintillator(materials=psb_scint_mat, thickness=(0.05, 0.01, 0.2))
    spec_model_1 = [source_1, filter_1, scintillator_1]
    spec_model_2 = [source_1, filter_1, filter_2, scintillator_1]
    spec_models = [spec_model_1, spec_model_2]

    train_spec_models = [spec_models[i // 4] for i in range(len(nrads_list))]

    # --------- Step 5: Perform spectral calibration ---------
    # Solve for the best-fit spectrum and system parameters given the forward model and measurements
    Estimator = Estimate(energies)
    i = 0
    for nrads, forward_matrix, mask, concatenate_models in zip(nrads_list, spec_F_list, trans_mask_list, train_spec_models):
        Estimator.add_data(nrads[mask], forward_matrix[mask], concatenate_models, weight=None)
        i += 1

    # Fit data
    Estimator.fit(learning_rate=learning_rate,
                  max_iterations=max_iterations,
                  stop_threshold=stop_threshold,
                  optimizer_type=optimizer_type,
                  loss_type='least_square',
                  logpath=None,
                  num_processes=6)


    # --------- Step 6: Plot estimated spectrum and display parameters ---------
    # Ensure output directory exists
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # Save Estimator object (if it's serializable this way)
    als_estimator_path = os.path.join(output_dir, 'ALS_Estimator.npy')
    np.save(als_estimator_path, Estimator)  # Save parameters instead of the object directly
    print(f"Saved ALS Estimator parameters to: {als_estimator_path}")
    print()
    print("Estimated Parameters:")
    pprint(Estimator.get_params())

    # Get estimated effective responses
    est_sp = Estimator.get_spectra()
    title_list = ['Low filtration Effective Spectrum', 'High filtration Effective Spectrum']
    savename_list = ['ALS_low_fltr_sp.npy', 'ALS_high_fltr_sp.npy']

    # Plot and save spectra
    fig = plt.figure()
    for i in range(2):
        with torch.no_grad():
            es = est_sp[i * 4].numpy()
            es /= np.trapezoid(es, energies)
            save_path = os.path.join(output_dir, savename_list[i])
            np.save(save_path, [energies, es])
            print(f"Saved estimated spectrum to: {save_path}")
            plt.plot(energies, es, label=title_list[i])

    plt.legend()
    plt.title('Estimated Effective Spectrum')
    plt.show(block=False)
    plt.savefig('./output/als_estimated_spec.png')

    print("Plotting complete. All results saved in the './output' folder.")
