import re
import numpy as np
from numpy.core.numeric import asanyarray
import torch
import matplotlib.pyplot as plt
import warnings

light_speed = 299792458.0 # Speed of light
Planck_constant = 6.62607015E-34 # Planck's constant
Joule_per_eV = 1.602176565E-19 # Joules per electron-volts


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    return torch.tensor(data)


def min_max_normalize_scalar(value, data_min, data_max):
    value = to_tensor(value)
    data_min = to_tensor(data_min)
    data_max = to_tensor(data_max)

    normalized_value = (value - data_min) / (data_max - data_min)

    # If the original input was a standard Python scalar, return a scalar
    if isinstance(value, (float, int)):
        return normalized_value.item()
    return normalized_value


def min_max_denormalize_scalar(normalized_value, data_min, data_max):
    normalized_value = to_tensor(normalized_value)
    data_min = to_tensor(data_min)
    data_max = to_tensor(data_max)

    value = normalized_value * (data_max - data_min) + data_min

    # If the original input was a standard Python scalar, return a scalar
    if isinstance(normalized_value, (float, int)):
        return value.item()
    return value

def is_sorted(lst):
    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))

def get_wavelength(energy):
    # How is energy related to the wavelength of radiation?
    # https://www.e-education.psu.edu/meteo300/node/682
    return (Planck_constant*light_speed/(energy*Joule_per_eV)) #in mm


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



def gen_high_con_mat(peaks, energies, width=2, mat_type='Equilateral Triangle'):
    """Generate high contrast matrix containing normalized triangles functions as columns,

    Parameters
    ----------
    peaks: list
        1D sequence represents the indexes of peaks.
    energies: numpy.ndarray
        List of X-ray energies of a poly-energetic source in units of keV.
    width: traingle

    Returns
    -------
    B: 2D numpy.ndarray
        The high contrast matrix, which is a highly sparse non-negative matrix.

    """
    xv, yv = np.meshgrid(energies[peaks], energies)
    if mat_type == 'Equilateral Triangle':
        B = np.clip(1-np.abs(1/width * (yv-xv)),0,1)
    elif mat_type == 'Right Triangle':
        mask = (yv-xv)>=0
        B = np.clip(1-(1/width * (yv-xv)),0,1)*mask
    elif mat_type == 'Left Triangle':
        mask = (yv-xv)<=0
        B = np.clip(1+(1/width * (yv-xv)),0,1)*mask
    return B

def huber_func(omega, c):
    if np.abs(omega)<c:
        return omega**2/2
    else:
        return c*np.abs(omega)-c**2/2
        
def binwised_spec_cali_cost(y,x,h,F,W,B,beta,c,energies):
    m,n = np.shape(F)
    e=(y - F @W@ (x + B @ h))
    cost = e.T@e/m
    rho_cost = 0
    for i in range(len(x)-1):
        rho_cost+=beta*(energies[i+1]-energies[i])*huber_func((x[i+1]-x[i])/(energies[i+1]-energies[i]),c)
        
    return cost,rho_cost


def concatenate_items(*items):
    concatenated = []
    lengths_info = {}

    for index, item in enumerate(items):
        if isinstance(item, (list, tuple)):
            concatenated.extend(item)
            lengths_info[index] = {'length': len(item), 'is_list': True}
        else:
            concatenated.append(item)
            lengths_info[index] = {'length': 1, 'is_list': False}

    return concatenated, lengths_info

def split_list(input_list, lengths_info):
    output = []
    start = 0
    for index, info in lengths_info.items():
        end = start + info['length']
        if info['is_list']:
            output.append(input_list[start:end])
        else:
            output.append(input_list[start])
        start = end
    return output

def nested_list(tup, indices):
    result = []
    start = 0
    for index in indices:
        result.append(list(tup[start:start+index]))
        start += index
    return result

def contains_nan(tensor):
    return torch.isnan(tensor).any()

def check_gradients_for_nan(model):
    has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None and contains_nan(param.grad):
            print(f"NaN value found in gradients of: {name}")
            has_nan = True
            return has_nan
    return has_nan



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


def get_dict_info(dict_index, src_info, fltr_info_dict, scints_info_dict):
    src_len = len(src_info)
    fltr_len = len(fltr_info_dict)
    scint_len = len(scints_info_dict)

    print(src_info[dict_index // (fltr_len * scint_len)])
    print(fltr_info_dict[dict_index // scint_len % fltr_len])
    print(scints_info_dict[dict_index % scint_len])


def extract_rsn_from_path(path):
    match = re.search(r'_rsn_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def nrmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    y_range = np.sqrt(np.mean((y_true) ** 2))
    return rmse / y_range


def neg_log_space(vmin, vmax, num, scale=1):
    """
    vmin, vmax must be positive.
    """
    return np.abs(
        -np.log(np.linspace(np.exp(-vmin / vmax / scale), np.exp(-vmax / vmax / scale), num=num))) * vmax * scale



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