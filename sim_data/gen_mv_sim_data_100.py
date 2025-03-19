# Basic Packages
import os
import numpy as np
import matplotlib.pyplot as plt
import svmbir
import h5py
import random

from xspec.chem_consts import get_lin_att_c_vs_E
from xspec import calc_forward_matrix
from xspec._utils import Gen_Circle
import spekpy as sp  # Import SpekPy
from xspec.defs import Material
from xspec.chem_consts._periodictabledata import density
from xspec.models import *
from tqdm import tqdm
class fw_projector:
    """A class for forward projection using SVMBIR."""

    def __init__(self, angles, num_channels, delta_pixel=1):
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

    def forward(self, mask):
        """
        Computes the projection of a given mask.

        Parameters:
            mask (numpy.ndarray): 3D mask of the object to be projected.

        Returns:
            numpy.ndarray: The computed projection of the mask.
        """
        projections = svmbir.project(mask, self.angles, self.num_channels) * self.delta_pixel
        return projections

if __name__ == '__main__':
    random.seed(142)
    saved_folder = '/home/li3120/scratch/sim_data/TCI_experiments/'
    os.makedirs(saved_folder, exist_ok=True)
    num_test = 100
    min_simkV = 30
    max_simkV = 160

    # Scanned cylinders
    materials = ['V', 'Al', 'Ti', 'Mg']
    mat_density = [density['%s' % formula] for formula in materials]
    energies = np.linspace(1, max_simkV, max_simkV)  # Define energies bins from 1 kV to 160 kV with step size 1 kV.
    lac_vs_E_list = []

    for i in range(len(materials)):
        formula = materials[i]
        den = mat_density[i]
        lac_vs_E_list.append(get_lin_att_c_vs_E(den, formula, energies))

    # FOV is about 2 mm * 2 mm
    nchanl = 1024
    rsize = 0.003  # mm

    # 4 cylinders with 1mm radius are evenly distributed on a circle with 3mm radius.
    Radius = [0.3 for _ in range(len(materials))]
    arrange_with_radius = 0.9
    centers = [[np.sin(rad_angle) * arrange_with_radius, np.cos(rad_angle) * arrange_with_radius]
               for rad_angle in np.linspace(-np.pi / 2, -np.pi / 2 + np.pi * 2, len(materials), endpoint=False)]

    # Each mask represents a homogenous cylinder.
    mask_list = []
    for mat_id, mat in enumerate(materials):
        circle = Gen_Circle((nchanl, nchanl), (rsize, rsize))
        # Use np.newaxis to convert 2D array to 3D.
        mask_list.append(circle.generate_mask(Radius[mat_id], centers[mat_id])[np.newaxis])


    angles = np.linspace(-np.pi/2, np.pi, 40, endpoint=False)
    projector = fw_projector(angles, num_channels=1024, delta_pixel=rsize)

    spec_F = calc_forward_matrix(mask_list, lac_vs_E_list, projector)

    simkV_list = np.linspace(min_simkV, max_simkV, (max_simkV - min_simkV) // 10 + 1, endpoint=True).astype('int')

    ref_takeoff_angle = 11
    # Energy bins.
    energies = np.linspace(1, max_simkV, max_simkV)

    # Use Spekpy to generate a source spectra dictionary.
    src_spec_list = []

    print('\nRunning demo script (10 mAs, 100 cm)\n')
    for simkV in simkV_list:
        s = sp.Spek(kvp=simkV + 1, th=ref_takeoff_angle, dk=1, mas=180-simkV, char=True)  # Create the spectrum model
        k, phi_k = s.get_spectrum(edges=True)  # Get arrays of energy & fluence spectrum
        phi_k = phi_k * ((rsize / 10) ** 2)
        src_spec = np.zeros((max_simkV))
        src_spec[:simkV] = phi_k[::2]
        src_spec_list.append(src_spec)

    print('\nFinished!\n')

    # A dictionary of source spectra with source voltage from 30 kV to 200 kV
    src_spec_list = np.array(src_spec_list)

    voltage_list = [50.0, 100.0, 150.0]  # kV



    scint_params_list = [
        {'formula': 'CsI', 'density': 4.51},
        {'formula': 'Gd3Al2Ga3O12', 'density': 6.63},
        {'formula': 'Lu3Al5O12', 'density': 6.73},
        {'formula': 'CdWO4', 'density': 7.9},
        {'formula': 'Y3Al5O12', 'density': 4.56},
        {'formula': 'Bi4Ge3O12', 'density': 7.13},
        {'formula': 'Gd2O2S', 'density': 7.32}
    ]
    psb_fltr_mat = [Material(formula='Al', density=2.702), Material(formula='Cu', density=8.92)]
    psb_scint_mat = [Material(formula=scint_p['formula'], density=scint_p['density']) for scint_p in scint_params_list]
    takeoff_angle_range = (5, 25)
    Al_filter_th_range = (0, 10)
    Cu_filter_th_range = (0, 0.6)
    scint_th_range = (0.001, 0.5)

    for n in tqdm(range(num_test), desc="Processing"):
        # Randomly select materials and thicknesses
        selected_fltr_mat = random.choice(psb_fltr_mat)
        selected_scint_mat = random.choice(psb_scint_mat)
        selected_takeoff_angle = random.uniform(*takeoff_angle_range)
        selected_Al_filter_th = random.uniform(*Al_filter_th_range)
        selected_Cu_filter_th = random.uniform(*Cu_filter_th_range)
        selected_scint_th = random.uniform(*scint_th_range)

        sources = [Reflection_Source_Analytical(voltage=(voltage, None, None), takeoff_angle=(selected_takeoff_angle, None, None), single_takeoff_angle=True) for
                   voltage in voltage_list]
        for src_i, source in enumerate(sources):
            source.set_src_spec_list(src_spec_list, simkV_list, ref_takeoff_angle)

        if selected_fltr_mat.formula == 'Al':
            filter_1 = Filter([selected_fltr_mat], thickness=(selected_Al_filter_th, None, None))
            filter_thickness_str = f"Thickness: {selected_Al_filter_th:.2f} mm (Al)"
        elif selected_fltr_mat.formula == 'Cu':
            filter_1 = Filter([selected_fltr_mat], thickness=(selected_Cu_filter_th, None, None))
            filter_thickness_str = f"Thickness: {selected_Cu_filter_th:.2f} mm (Cu)"

        scintillator_1 = Scintillator(materials=[selected_scint_mat], thickness=(selected_scint_th, None, None))

        gt_spec_list = [(source(energies) * filter_1(energies) * scintillator_1(energies)).numpy() for source in sources]
        # Generate title string
        title_str = (f"Takeoff Angle: {selected_takeoff_angle:.2f} degrees\n"
                     f"Filter Material: {selected_fltr_mat.formula},\n"
                     f"{filter_thickness_str}\n"
                     f"Scintillator Material: {selected_scint_mat.formula},\n"
                     f"Thickness: {selected_scint_th:.3f} mm")
        # Create a figure with two subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        # Subplot 1: Ground Truth X-ray Spectral Response
        for spec_i, gt_spec in enumerate(gt_spec_list):
            axs[0].plot(energies, gt_spec / np.trapz(gt_spec, energies), label='%d kV' % voltage_list[spec_i])
        axs[0].legend(loc='upper left')
        axs[0].set_title('Ground Truth: Total X-ray Spectral Response')
        axs[0].set_xlabel('Energy [keV]')
        axs[0].set_ylabel('Normalized Response')

        datasets = []
        label_list = ['50 kV', '100 kV', '150 kV']

        for case_i, gt_spec in zip(np.arange(len(gt_spec_list)), gt_spec_list):
            spec_F_train_list = []
            trans_list = []

            # Add poisson noise before reaching detector/scintillator.
            trans = np.trapz(spec_F * gt_spec, energies, axis=-1)
            trans_0 = np.trapz(gt_spec, energies, axis=-1)
            trans_noise = np.random.poisson(trans).astype(np.float64)
            trans_noise /= trans_0

            # Add poisson noise before reaching detector/scintillator.
            trans = np.trapz(spec_F * gt_spec, energies, axis=-1)
            trans_0 = np.trapz(gt_spec, energies, axis=-1)
            trans_noise = np.random.poisson(trans).astype(np.float64)
            trans_noise /= trans_0

            # Store noiseless transmission data and forward matrix.
            trans_list.append(trans_noise)
            spec_F_train = spec_F.reshape((-1, spec_F.shape[-1]))
            spec_F_train_list.append(spec_F_train)
            spec_F_train_list = np.array(spec_F_train_list)
            trans_list = np.array(trans_list)
            axs[1].plot(trans_list[0][16, 0], label=label_list[case_i])

            d = {
                'measurement': trans_list,
                'forward_mat': spec_F_train_list,
                'source': sources[case_i],
                'filter': filter_1,
                'scintillator': scintillator_1,
            }
            datasets.append(d)
        axs[1].legend(loc='lower left')
        axs[1].set_title('Specific Transmission Curve')
        axs[1].set_xlabel('Index')
        axs[1].set_ylabel('Transmission')
        # Add the title string as a textbox in the figure
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.87, 0.8, title_str, transform=fig.transFigure, fontsize=10,
                 verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.savefig(f'{saved_folder}sp_and_trans%d.png'%n)
        with open(f'{saved_folder}sim%d_3v1f1s_dataset.npy'%n, 'wb') as f:
            np.save(f, datasets, allow_pickle=True)
