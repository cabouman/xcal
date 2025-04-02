# Standard library imports
import os

# Third-party library imports for numerical operations
import numpy as np
import h5py
from scipy.ndimage import convolve1d

# Image processing and computer vision libraries
from scipy.interpolate import interp1d
def print_h5py_structure(group, indent=0):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print(" " * indent + f"Group: {key}")
            print_h5py_structure(item, indent + 4)
        elif isinstance(item, h5py.Dataset):
            print(" " * indent + f"Dataset: {key} (Shape: {item.shape}, Dtype: {item.dtype})")
        else:
            print(" " * indent + f"Unknown type: {key}")

def read_nml_als(dataset_name, start_row, end_row, slice_step=1):
    '''Read and normalize als h5 file with data, blank, and dark scans.

    '''
    file = h5py.File(dataset_name, 'r')

    # Preprocess raw measurement
    data_dark = np.mean(file['exchange']['data_dark'][:, start_row:end_row], axis=0, keepdims=True)
    data_white = np.mean(file['exchange']['data_white'][:, start_row:end_row], axis=0, keepdims=True)
    data = np.array(file['exchange']['data'][::slice_step, start_row:end_row]).astype('float64')

    data_norm = data - data_dark
    data_norm /= (data_white - data_dark)

    return data_norm

def read_als_config(dataset_name):
    # Open the HDF5 file
    als_cfg = {}
    with h5py.File(dataset_name, 'r') as f:
        als_cfg['pixel_size' ] =f['measurement']['instrument']['detector']['pixel_size'][0]
        als_cfg['dimension_x' ] =f['measurement']['instrument']['detector']['dimension_x'][0]
        als_cfg['dimension_y' ] =f['measurement']['instrument']['detector']['dimension_y'][0]
        als_cfg['num_angles' ] =f['process']['acquisition']['rotation']['num_angles'][0]
        als_cfg['arange'] = f['process']['acquisition']['rotation']['range'][0]
        als_cfg['theta'] = f['exchange']['theta'][:]
        als_cfg['theta_dark'] = f['exchange']['theta_dark'][:]
        als_cfg['theta_white'] = f['exchange']['theta_white'][:]

    return als_cfg

def load_als_bm832():
    """
    Load the ALS Beamline 8.3.2 spectrum data from 'als_bm832.h5' and return photon counts vs. energy.

    Parameters
    ----------

    Returns
    -------
    energies : np.ndarray
        Energy values in keV with uniform bins.
    spectrum : np.ndarray
        Corresponding photon counts for each energy value.

    Notes
    -----
    This function reads data from the file 'als_bm832.h5' located in the same directory as this script.
    """

    filen = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'als_bm832.h5')
    with h5py.File(filen ,'r+') as fid:
        spectrum = np.array(fid['spectrum'])
        energies = np.array(fid['energies'])
        return energies, spectrum


def als_bm832():
    """
    Returns the differential version of the ALS Beamline 8.3.2 spectrum rebinned into uniform 1 keV bins.

    This function converts the original spectrum, which is defined over non-uniform energy bins,
    into a differential form by redistributing photon counts into uniform 1 keV-wide bins from 0 to 100 keV.
    The output spectrum represents photon counts per 1 keV, preserving total photon count via fractional overlap.

    Returns
    -------
    rebinned_energies : np.ndarray
        Center energies of the new uniform bins, ranging from 0.5 to 99.5 keV.
    rebinned_spectrum : np.ndarray
        Differential photon spectrum: counts per 1 keV bin after redistribution.

    Notes
    -----
    This differential version ensures consistency with systems or models that assume uniform binning.
    It uses overlap-based count redistribution to accurately rebin the original non-uniform data.
    """

    energies, responses = load_als_bm832()

    # Calculate original bin widths
    original_bin_widths = np.diff(energies)

    # Define new equispaced energy bins
    rebinned_energies = np.linspace(0, 100, num=101)  # Equispaced bin edges

    # Function to calculate overlap and redistribute counts
    def distribute_counts(old_edges, old_counts, new_edges):
        # Initialize new responses array
        new_responses = np.zeros(len(new_edges ) -1)
        for i in range(len(old_counts ) -1):
            start_edge = old_edges[i]
            end_edge = old_edges[ i +1]
            count = old_counts[i]

            # Find new bins that overlap with the old bin
            overlap_start = np.searchsorted(new_edges, start_edge, side='right') - 1
            overlap_end = np.searchsorted(new_edges, end_edge, side='left')
            #         print(i, overlap_start, overlap_end)

            for j in range(overlap_start, overlap_end):
                new_start_edge = new_edges[j]
                new_end_edge = new_edges[ j +1]

                # Calculate overlap as a fraction of the old bin width
                overlap_fraction = (min(end_edge, new_end_edge) - max(start_edge, new_start_edge)) / (end_edge - start_edge)
                new_responses[j] += overlap_fraction * count

        return new_responses

    # Redistribute the original counts into the new bins
    rebinned_spectrum = distribute_counts(energies, responses, rebinned_energies)

    return rebinned_energies[1: ] -0.5, rebinned_spectrum


def detect_outliers(data, window_size, threshold_std=3):
    """
    Create a mask for outliers in a 3D data array based on convolution over the third axis.

    Parameters:
    - data: numpy.ndarray, the input data with shape (Nviews, Nrows, Ncols)
    - window_size: int, the size of the window to use for the convolution kernel

    Returns:
    - mask: numpy.ndarray, a boolean array with True indicating outliers
    """
    # Create the convolution kernel based on the window size
    # The kernel will have the central value as 1 and the others as -1/window_size
    kernel = np.full(window_size, -1 / (window_size - 1))
    kernel[window_size // 2] = 1

    # Initialize the mask with the same shape as the data
    mask = np.zeros_like(data, dtype=bool)

    # Perform the convolution and create the mask
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Convolve along the third axis
            convolved = convolve1d(data[i, j, :], kernel, mode='constant', cval=1.0)
            # Determine a threshold for what you consider an outlier
            threshold = np.std(convolved) * threshold_std  # For example, 3 standard deviations
            # Create the mask based on the threshold
            mask[i, j, :] = np.abs(convolved) < threshold

    return mask


# Example usage:
# data is your numpy array with the shape (16, 3, 2560)
# window_size is the size of the window for the convolution kernel
# mask = create_mask_for_outliers(data, window_size=5)

def only_center_mask(data, window_size=None):
    """
    Creates a new mask that isolates a window around the minimum value along the third axis,
    considering only the unmasked points.

    Parameters:
    - data: numpy.ndarray, the input data with shape (Nviews, Nrows, Ncols)
    - mask: numpy.ndarray, the input boolean mask array with the same shape as data
    - window_size: int, the size of the window around the minimum value to keep unmasked

    Returns:
    - new_mask: numpy.ndarray, the new boolean mask array
    """
    new_mask = np.ones_like(data, dtype=bool)  # Initialize new mask to all True (everything masked)
    if window_size is None:
        window_size = data.shape[2]
    half_window = window_size // 2
    kernel = np.ones(window_size)

    # Iterate through the slices
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Use convolve1d to calculate the running window sum
            window_sums = convolve1d(data[i, j, :], kernel, mode='constant', cval=np.inf)
            # Find the index of the minimum value in the slice, ignoring NaNs
            argmin = np.argmin(window_sums)
            # Calculate the window bounds
            start_index = max(argmin - half_window, 0)
            end_index = min(argmin + half_window + 1, data.shape[2])
            # Update the new mask to unmask the window around the minimum
            new_mask[i, j, :start_index] = False
            new_mask[i, j, end_index + 1:] = False

    return new_mask