import torch
from torch.nn import Module, Parameter, Softmax
import numpy as np


def denormalize(normalized_value, bounds):
    """Converts a normalized value [0, 1] back to its original range."""
    return normalized_value * (bounds[1] - bounds[0]) + bounds[0]

from torch.nn import Module

class Reflection_Source(Module):
    def __init__(self, voltage, takeoff_angle, device=None, dtype=None):
        """
        A template source model designed specifically for reflection sources, including all necessary methods.

        Args:
            voltage (tuple): (initial value, lower bound, upper bound) for the source voltage.
                These three values cannot be all None. It will not be optimized when lower == upper.
            takeoff_angle (tuple): (initial value, lower bound, upper bound) for the takeoff angle, in degrees.
                These three values cannot be all None. It will not be optimized when lower == upper.
            device (torch.device, optional): The device tensors will be allocated to.
            dtype (torch.dtype, optional): The desired data type for the tensors.
        """

    def get_parameters(self):
        """
        Returns the current parameters (denormalized) of the source model.

        Returns:
            dict: A dictionary containing the parameters 'voltage' and 'takeoff_angle'.
        """

    def forward(self, energies):
        """
        Takes X-ray energies and returns the source spectrum.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The source response.
        """


class Filter_Model(Module):
    def __init__(self, materials, thickness, device=None, dtype=None):
        """
        A template filter model based on Beer's Law and NIST mass attenuation coefficients, including all necessary methods.

        Args:
            materials (list): A list of possible materials for the filter,
                where each material should be an instance containing formula and density.
            thickness (tuple): (initial value, lower bound, upper bound) for the filter thickness.
                These three values cannot be all None. It will not be optimized when lower == upper.
            device (torch.device, optional): The device tensors will be allocated to.
            dtype (torch.dtype, optional): The desired data type for the tensors.
        """

    def get_parameters(self):
        """
        Returns the current parameters (denormalized) of the filter model.

        Returns:
            dict: A dictionary containing the parameters.
        """

    def forward(self, energies):
        """
        Takes X-ray energies and returns the filter response.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The filter response as a function of input energies, selected material, and its thickness.
        """


class Scintillator_Model(Module):
    def __init__(self, thickness, materials, device=None, dtype=None):
        """
        A template scintillator model based on Beer's Law, NIST mass attenuation coefficients, and mass energy-absorption coefficients, including all necessary methods.

        Args:
            materials (list): A list of possible materials for the scintillator,
                where each material should be an instance containing formula and density.
            thickness (tuple): (initial value, lower bound, upper bound) for the scintillator thickness.
                These three values cannot be all None. It will not be optimized when lower == upper.
            device (torch.device, optional): The device tensors will be allocated to.
            dtype (torch.dtype, optional): The desired data type for the tensors.
        """

    def get_parameters(self):
        """
        Returns the current parameters (denormalized) of the scintillator model.

        Returns:
            dict: A dictionary containing the parameters.
        """

    def forward(self, energies):
        """
        Takes X-ray energies and returns the scintillator response.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The scintillator conversion function as a function of input energies, selected material, and its thickness.
        """



def estimate(energies, normalized_rads, forward_matrices, sources, filters, scintillator, model_combination,
             weight=None, weight_type='unweighted', blank_rads=None, attenuation_space=False,
             learning_rate=0.001, max_iterations=5000, stop_threshold=1e-4, optimizer_type='Adam', loss_type='wmse', logpath=None,
             num_processes=1):
    """
    Estimate the X-ray CT parameters that determine the X-ray energy spectrum including the source voltage,
    anode take-off angle, filter material and thickness, and scintillator and thickness.

    Args:
        energies (numpy.ndarray): Array of interested X-ray photon energies in keV with size N_energies_bins.
        normalized_rads (list of numpy.ndarray): Normalized radiographs at different source voltages and filters.
            Each radiograph has dimensions [N_views, N_rows, N_cols].
        forward_matrices (list of numpy.ndarray): Corresponding forward matrices for normalized_rads. We provide ``xspec.calc_forward_matrix`` to calculate a forward matrix from a 3D mask for a homogenous object. Each forward matrix, corresponding to radiograph, has dimensions [N_views, N_rows, N_cols, N_energiy_bins].

        sources (object): A list of instances of any source model. See xspec.models.
        filters (object): A list of instances of any filter model. See xspec.models.
        scintillator (object): A instance of scintillator model. See xspec.models.
        model_combination : list of Model_combination
            Each instance of Model_combination specify which source, filters, and scintillator are used for one radiograph.
        weight (optional): [Default=None] Weights to apply during the estimation process.
        weight_type (str, optional): [Default='unweighted'] Type of noise model used for data.
            Option “unweighted” corresponds to unweighted reconstruction;
            Option “transmission” is the correct weighting for transmission CT with constant dose or given blank
            radiograph.
        blank_rads (list of numpy.ndarray, optional): A list of blank (object-free) radiograph arrays, where each array corresponds to a specific radiograph and has dimensions [N_views, N_rows, N_cols]. These arrays are used in scenarios where the weight calculation is necessary. Specifically, when 'weight_type' is “transmission”, the weight is determined by the formula: blank radiograph / normalized radiograph. This approach assumes that the variance of the object radiograph is proportional to the object radiograph divided by the square of the blank radiograph.
        attenuation_space (bool): Calculate loss in attenuation space.
        learning_rate (float, optional): [Default=0.001] Learning rate for the optimization process.
        max_iterations (int, optional): [Default=5000] Maximum number of iterations for the optimization.
        stop_threshold (float, optional): [Default=1e-4] Scalar valued stopping threshold in percent.
            If stop_threshold=0.0, then run max iterations.
        optimizer_type (str, optional): [Default='Adam'] Type of optimizer to use. If we do not have
            accurate initial guess use 'Adam', otherwise, 'NNAT_LBFGS' can provide a faster convergence.
        logpath (optional): [Default=None] Path for logging, if required.
        num_processes (int, optional): [Default=1] Number of processes to use for parallel computation.

    Returns:
        Estimated source objects, filters objects, and scintillator object.
    """