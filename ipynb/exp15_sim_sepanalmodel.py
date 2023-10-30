# Basic Packages
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

import svmbir

import spekpy as sp # Import SpekPy
import xspec
from xspec.chem_consts import get_lin_att_c_vs_E
from xspec.chem_consts._periodictabledata import density
from xspec.dict_gen import gen_filts_specD, gen_scints_specD



def find_element_change_indexes(lst):
    start_indexes = [0]
    current_element = lst[0]

    for i in range(1, len(lst)):
        if lst[i] != current_element:
            start_indexes.append(i)
            current_element = lst[i]

    return start_indexes

def find_bin_index(number, sorted_list):
    left, right = 0, len(sorted_list) - 1

    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] <= number:
            left = mid + 1
        else:
            right = mid - 1

    return right

def nrmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    y_range = np.sqrt(np.mean((y_true ) ** 2))
    return rmse / y_range


def neg_log_space(vmin, vmax, num, scale=1):
    """
    vmin, vmax must be positive.
    """
    return np.abs(-np.log(np.linspace(np.exp(-vmin / vmax / scale), np.exp(-vmax / vmax / scale), num=num))) * vmax *scale

def get_dict_info(dict_index, src_info, fltr_info_dict, scints_info_dict):
    src_len = len(src_info)
    fltr_len = len(fltr_info_dict)
    scint_len = len(scints_info_dict)

    print(src_info[dict_index // (fltr_len * scint_len)])
    print(fltr_info_dict[dict_index // scint_len % fltr_len])
    print(scints_info_dict[dict_index % scint_len])


class Gen_Circle:
    def __init__(self, canvas_shape, pixel_size):
        """
        Initialize the Circle class.

        Parameters:
        canvas_shape (tuple): The shape of the canvas, in pixels.
        pixel_size (tuple): The size of a pixel, in the same units as the canvas.
        """
        self.canvas_shape = canvas_shape
        self.pixel_size = pixel_size
        self.canvas_center = ((canvas_shape[0] - 1) / 2.0, (canvas_shape[1] - 1) / 2.0,)

    def generate_mask(self, radius, center=None):
        """
        Generate a binary mask for the circle.

        Parameters:
        radius (int): The radius of the circle, in pixels.
        center (tuple): The center of the circle.

        Returns:
        ndarray: A 2D numpy array where points inside the circle are marked as True and points outside are marked as False.
        """
        if center is None:
            center = ((self.canvas_shape[0] - 1) / 2.0, (self.canvas_shape[1] - 1) / 2.0)

        # Generate a grid of coordinates in the shape of the mask.
        Y, X = np.ogrid[:self.canvas_shape[0], :self.canvas_shape[1]]
        X = X - self.canvas_center[1]
        Y = Y - self.canvas_center[0]

        # Scale the coordinates by the pixel size.
        X = X * self.pixel_size[1]
        Y = Y * self.pixel_size[0]

        # Calculate the distance from the center to each coordinate.
        dist_from_center = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

        # Create a mask where points with a distance less than or equal to the radius are marked as True.
        mask = dist_from_center <= radius

        # Calculate the radius of the largest circle that can be inscribed in the canvas.
        inscribed_circle_radius = min(self.canvas_shape) // 2

        # Check if the mask is outside the inscribed circle.
        if radius > inscribed_circle_radius:
            warnings.warn("The generated mask falls outside the largest inscribed circle in the canvas.")

        return mask


# Customerize forward projector.
class pt_fw_projector:
    def __init__(self, angles, num_channels, delta_pixel=1, geometry='parallel'):
        """

        Parameters
        ----------
        energies
        N_views
        psize
        xcenter
        geometry
        arange
        """
        self.angles = angles
        self.num_channels = num_channels
        self.delta_pixel = delta_pixel
        self.geometry = geometry

    def forward(self, mask):
        """

        Parameters
        ----------
        mask : numpy.ndarray
            3D mask for pure solid object.

        Returns
        -------
        lai : numpy.ndarray
            Linear attenuation integral, of size M measurement * N energy bins.

        """

        projections = svmbir.project(mask, self.angles, self.num_channels) * self.delta_pixel

        return projections



    
if __name__ == '__main__':  
    os.makedirs('./output_exp15',exist_ok=True)
    filename = __file__.split('.')[0]

    ## A. Generate simulated cylinder
    # rsize = [0.03125 / 6, 0.03125 / 24.0]  # mm
    rsize = [0.03125 / 12.0, 0.03125 / 12.0, 0.03125 / 12.0]  # mm
    nchanl = 1024
    materials = ['Al', 'Ti', 'V']
    Radius = [1, 1, 1]
    centers = [0, 0]

    # Simulated sinogram parameters
    num_views = 9
    tilt_angle = np.pi / 2  # Tilt range of +-90deg
    # Generate the array of view angles
    angles = np.linspace(-tilt_angle, tilt_angle, num_views, endpoint=False)

    mask_scan = []
    for mat_id, mat in enumerate(materials):
        mask_list = []
        circle = Gen_Circle((nchanl, nchanl), (rsize[mat_id], rsize[mat_id]))
        mask_list.append(circle.generate_mask(Radius[mat_id], centers)[np.newaxis])
        mask_scan.append(mask_list)

    plt.clf()
    src_spec_list = []
    src_info = []
    simkV_list = np.linspace(30, 160, 14, endpoint=True).astype('int')
    max_simkV = max(simkV_list)
    energies = np.linspace(1, max_simkV, max_simkV)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=80)
    print('\nRunning demo script (1 mAs, 100 cm)\n')
    for simkV in simkV_list:
        for th in [12]:
            s = sp.Spek(kvp=simkV + 1, th=th, dk=1, char=True)  # Create the spectrum model
            k, phi_k = s.get_spectrum(edges=True)  # Get arrays of energy & fluence spectrum

            ## Plot the x-ray spectrum
            axs[0, 0].plot(k[::2], phi_k[::2] / np.trapz(phi_k[::2], k[::2]),
                           label='Char: kvp:%d Anode angle:%d' % (simkV, th))
            src_info.append((simkV,))
            src_spec = np.zeros((max_simkV))
            src_spec[:simkV] = phi_k[::2]
            src_spec_list.append(src_spec)

    print('\nFinished!\n')
    axs[0, 0].set_xlabel('Energy  [keV]', fontsize=8)
    axs[0, 0].set_ylabel('Differential fluence  [unit space$^{-1}$ mAs$^{-1}$ keV$^{-1}$]', fontsize=8)
    axs[0, 0].set_title('X-ray Source spectrum generated by spekpy')
    axs[0, 0].legend(fontsize=8)
    src_spec_list = np.array(src_spec_list)

    # Generate filter response
    fltr_params = [
        {'formula': 'Al', 'density': 2.702, 'thickness_list': neg_log_space(vmin=0.1, vmax=6, num=10, scale=1), 'thickness_bound':(0,10)},
        {'formula': 'Cu', 'density': 8.92, 'thickness_list': neg_log_space(vmin=0.25, vmax=0.6, num=10, scale=1), 'thickness_bound':(0,2)},
    ]

    fltr_dict, fltr_info_dict = gen_filts_specD(energies, composition=fltr_params)
    for i, fltr in enumerate(fltr_dict):
        axs[0, 1].plot(energies[:max_simkV], fltr[:max_simkV],
                       label='Material: %s, Thickness: %.4f' % (fltr_info_dict[i][0],
                                                                fltr_info_dict[i][1]))
    axs[0, 1].set_xlabel('Energy  [keV]', fontsize=8)
    axs[0, 1].set_ylabel('Differential fluence  [unit space$^{-1}$ mAs$^{-1}$ keV$^{-1}$]', fontsize=8)
    axs[0, 1].set_title('Filter response')
    axs[0, 1].legend(fontsize=8)

    # Scintillator model
    scint_params = [
        {'formula': 'CsI', 'density': 4.51, 'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound':(0.02,0.5)},
        {'formula': 'Gd3Al2Ga3O12', 'density': 6.63,
         'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound':(0.02,0.5)},
        {'formula': 'Lu3Al5O12', 'density': 6.73,
         'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound':(0.02,0.5)},
        {'formula': 'CdWO4', 'density': 7.9, 'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound':(0.02,0.5)},
        {'formula': 'Y3Al5O12', 'density': 4.56,
         'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound':(0.02,0.5)},
        {'formula': 'Bi4Ge3O12', 'density': 7.13,
         'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound':(0.02,0.5)},
        {'formula': 'Gd2O2S', 'density': 7.32, 'thickness_list': neg_log_space(vmin=0.02, vmax=0.35, num=10, scale=1), 'thickness_bound':(0.02,0.5)}
    ]

    scints_dict, scints_info_dict = gen_scints_specD(energies, composition=scint_params)
    for i, scints in enumerate(scints_dict):
        axs[1, 0].plot(energies[:max_simkV], scints[:max_simkV],
                       label='Material: %s, Thickness: %.2f' % (scints_info_dict[i][0],
                                                                scints_info_dict[i][1]))
    axs[1, 0].set_xlabel('Energy  [keV]', fontsize=8)
    axs[1, 0].set_ylabel('Differential fluence  [unit space$^{-1}$ mAs$^{-1}$ keV$^{-1}$]', fontsize=8)
    axs[1, 0].set_title('Scintillator response')

    spec_dict = src_spec_list[:, np.newaxis, np.newaxis, :] \
                * fltr_dict[np.newaxis, :, np.newaxis, :] \
                * scints_dict[np.newaxis, np.newaxis, :, :]

    spec_dict = spec_dict.reshape(-1, spec_dict.shape[-1]).T
    spec_dict_norm = spec_dict / np.trapz(spec_dict, energies, axis=0)
    print(spec_dict.shape)
    plt.savefig('./output_exp15/%s_dict.png'%filename)

    for rand_seed_num in [71, 81]:
        print('Simulation Case with random seed = ', rand_seed_num)
        plt.clf()
        np.random.seed(rand_seed_num)
        ref_src_spec_list = []
        ref_src_info = []

        fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=80)
        plt.rcParams.update({'font.size': 8})
        print('\nRunning demo script (1 mAs, 100 cm)\n')
        ## Generate spectrum for 100 kV potential, 10 deg. anode angle & 6 mm Al filtr.
        # for th in (np.exp(np.linspace(0,1.6,5))*6).astype('int'):
        simkV_list_indices = np.random.choice(np.arange(len(simkV_list)))
        simkV_list_indices = 3
        simkV = simkV_list[simkV_list_indices]
        print('simkV:', simkV)

        ## Plot the x-ray spectrum
        ref_src_info.append(simkV)
        ref_src_spec_list = np.array([src_spec_list[simkV_list_indices]])

        axs[0, 0].plot(energies, src_spec_list[simkV_list_indices], label='Char: kvp:%d Anode angle:%d' % (simkV, th))
        axs[0, 0].set_xlabel('Energy  [keV]', fontsize=8)
        axs[0, 0].set_ylabel('Differential fluence  [unit space$^{-1}$ mAs$^{-1}$ keV$^{-1}$]', fontsize=8)
        axs[0, 0].set_title('X-ray Source spectrum generated by spekpy')
        axs[0, 0].legend()

        # Generate filter response
        fltr_formular_list = [fid[0] for fid in fltr_info_dict]
        sorted_fltr_formular_indice_list = find_element_change_indexes(fltr_formular_list)
        rand_flter_formula_ind = np.random.choice(np.arange(len(fltr_formular_list)))
        print(rand_flter_formula_ind)
        ref_fltr_formula_indices = find_bin_index(rand_flter_formula_ind, sorted_fltr_formular_indice_list)

        ref_fltr_formula = fltr_params[ref_fltr_formula_indices]['formula']
        ref_fltr_formula_density = fltr_params[ref_fltr_formula_indices]['density']
        if ref_fltr_formula_indices == 2:
            ref_fltr_thickness = 0
        elif ref_fltr_formula_indices == 0:
            ref_fltr_thickness = np.random.uniform(0.1, 6)
        elif ref_fltr_formula_indices == 1:
            ref_fltr_thickness = np.random.uniform(0.25, 0.6)
        print('ref_fltr_formula:', ref_fltr_formula)
        print('ref_fltr_thickness:', ref_fltr_thickness, 'In flter response list:',
              find_bin_index(ref_fltr_thickness, fltr_params[ref_fltr_formula_indices]['thickness_list']))

        ref_fltr_params = [
            {'formula': ref_fltr_formula, 'density': ref_fltr_formula_density, 'thickness_list': [ref_fltr_thickness]},
        ]
        ref_fltr_dict, ref_fltr_info_dict = gen_filts_specD(energies, composition=ref_fltr_params)

        axs[0, 1].plot(energies[:max_simkV], ref_fltr_dict[0, :max_simkV],
                       label='Material: %s, Thickness: %.2f' % (ref_fltr_info_dict[0][0],
                                                                ref_fltr_info_dict[0][1]))
        axs[0, 1].set_xlabel('Energy  [keV]', fontsize=8)
        axs[0, 1].set_ylabel('Differential fluence  [unit space$^{-1}$ mAs$^{-1}$ keV$^{-1}$]', fontsize=8)
        axs[0, 1].set_title('Filter response')

        fbini = find_bin_index(ref_fltr_thickness,
                               fltr_params[ref_fltr_formula_indices]['thickness_list']) + ref_fltr_formula_indices * 10
        for fli in [fbini, fbini + 1]:
            axs[0, 1].plot(energies[:max_simkV], fltr_dict[fli], '--',
                           label='Component Material: %s, Thickness: %.4f' % (fltr_info_dict[fli][0],
                                                                              fltr_info_dict[fli][1]))
        axs[0, 1].legend()
        # Scintillator model
        scint_formula_list = [param['formula'] for param in scint_params]
        ref_scint_formula_indices = np.random.choice(np.arange(len(scint_formula_list)))
        ref_scint_formula = scint_params[ref_scint_formula_indices]['formula']
        ref_scint_formula_density = scint_params[ref_scint_formula_indices]['density']
        ref_scint_thickness = np.random.uniform(0.02, 0.35)
        print('ref_scint_formula:', ref_scint_formula)
        print('ref_scint_thickness:', ref_scint_thickness, 'In scintillator response list:',
              find_bin_index(ref_scint_thickness, scint_params[ref_scint_formula_indices]['thickness_list']))

        ref_scint_params = [
            {'formula': ref_scint_formula, 'density': ref_scint_formula_density,
             'thickness_list': [ref_scint_thickness]}]
        ref_scints_dict, ref_scints_info_dict = gen_scints_specD(energies, composition=ref_scint_params)

        axs[1, 0].plot(energies[:max_simkV], ref_scints_dict[0, :max_simkV],
                       label='Material: %s, Thickness: %.2f' % (ref_scints_info_dict[0][0],
                                                                ref_scints_info_dict[0][1]))
        axs[1, 0].set_xlabel('Energy  [keV]', fontsize=8)
        axs[1, 0].set_ylabel('Differential fluence  [unit space$^{-1}$ mAs$^{-1}$ keV$^{-1}$]', fontsize=8)
        axs[1, 0].set_title('Scintillator response')
        sbini = find_bin_index(ref_scint_thickness, scint_params[ref_scint_formula_indices][
            'thickness_list']) + ref_scint_formula_indices * 10
        for sci in [sbini, sbini + 1]:
            axs[1, 0].plot(energies[:max_simkV], scints_dict[sci], '--',
                           label='Estimated component Material: %s, Thickness: %.4f' % (scints_info_dict[sci][0],
                                                                                        scints_info_dict[sci][1]))
        axs[1, 0].legend()

        gt_spec_dict = ref_src_spec_list[:, np.newaxis, np.newaxis, :] \
                       * ref_fltr_dict[np.newaxis, :, np.newaxis, :] \
                       * ref_scints_dict[np.newaxis, np.newaxis, :, :]
        gt_spec_dict = gt_spec_dict.reshape((-1, gt_spec_dict.shape[-1])).T
        gt_spec_dict /= np.trapz(gt_spec_dict, energies, axis=0)
        print('Generate spectrum dictionary with shape: ', gt_spec_dict.shape)

        axs[1, 1].plot(energies[:max_simkV], gt_spec_dict[:max_simkV], label='Groundtruth')
        axs[1, 1].set_xlabel('Energy  [keV]', fontsize=8)
        axs[1, 1].set_ylabel('Differential fluence  [unit space$^{-1}$ mAs$^{-1}$ keV$^{-1}$]', fontsize=8)
        axs[1, 1].set_title('Groundtruth System response')
        axs[1, 1].legend()
        print('\nFinished!\n')
        plt.savefig('./output_exp15/exp2_%s_gt_%d.png' % (filename, rand_seed_num))

        # D. Prepare forward matrix F
        spec_F_train_list = []
        proj_list = []

        for mat_id, mat in enumerate(materials):
            lac_vs_E_list = []
            mask_list = mask_scan[mat_id]
            for i in range(len(mask_list)):
                formula = mat
                den = density['%s' % formula]
                lac_vs_E_list.append(get_lin_att_c_vs_E(den, formula, energies))

            pfp = pt_fw_projector(angles, num_channels=nchanl, delta_pixel=rsize[mat_id])
            spec_F = xspec.cal_fw_mat(mask_list, lac_vs_E_list, energies, pfp)
            proj = np.trapz(spec_F * gt_spec_dict.flatten(), energies, axis=-1)
            proj_list.append(proj)
            spec_F_train = spec_F.reshape((-1, spec_F.shape[-1]))
            spec_F_train_list.append(spec_F_train)

        spec_F_train_list = np.array(spec_F_train_list)
        proj_list = np.array(proj_list)
        npt = 0.009
        proj_n_list = [proj + np.sqrt(proj) * np.random.normal(0, npt, size=proj.shape) for proj in proj_list]

        spec_F_train = [spec_F.reshape((-1, spec_F.shape[-1])) for spec_F in spec_F_train_list]
        signal_train = proj_n_list

        plt.clf()
        plt.plot(proj_n_list[0][0, 0, :], label='Noisy: Al')
        plt.plot(proj_list[0, 0, 0, :], label='Ideal: Al')

        plt.plot(proj_n_list[1][0, 0, :], label='Noisy: Ti')
        plt.plot(proj_list[1, 0, 0, :], label='Ideal: Ti')

        plt.plot(proj_n_list[2][0, 0, :], label='Noisy: V')
        plt.plot(proj_list[2, 0, 0, :], label='Ideal: V')

        plt.legend()
        plt.ylim((0,1))
        plt.savefig('./output_exp15/exp2_%s_proj_%d.png' % (filename, rand_seed_num))

        # E. Uncertainty Quantification
        # Estimated system response
        # Parallel Computation
        num_threads = 8

        res, params_combinations = xspec.parallel_anal_sep_model(num_threads,
            [[np.array([src_spec_list[i]])] for i in range(len(simkV_list))],
            fltr_params,
            scint_params,
            np.linspace(0, 10, 3, endpoint=True),
            np.linspace(0.02, 0.5, 3, endpoint=True),
            energies, [signal_train], spec_F_train,
            fltr_th_bound=(0, 10),
            scint_th_bound=(0.02, 0.5),
            learning_rate=2e-1, iterations=2000, tolerance=1e-5, return_history=False)

        np.save('./output_exp15/exp2_res_rdsd_%d.npy' % (rand_seed_num), np.array([res, params_combinations], dtype=object))
