from xcal.models import Filter, Scintillator
from xcal.defs import Material
import numpy as np

def get_filter_response(energies, mat, den, thickness):
    """
    Calculates the filter response for a specified material, density, and thickness.

    Args:
        energies (numpy.ndarray): 1D array of energy values (in keV) at which to calculate the filter response.
        mat (str): Chemical formula of the filter material (e.g., 'Al' for aluminum).
        den (float): Density of the filter material in g/cm³.
        thickness (float): Thickness of the filter in mm.

    Returns:
        numpy.ndarray: Array containing the filter response at each specified energy as a numpy array.
    """
    psb_mat = [Material(formula=mat, density=den)]
    fm = Filter(psb_mat, thickness=(thickness, None, None))
    return fm(energies).detach().numpy()


def get_scintillator_response(energies, mat, den, thickness):
    """
    Calculates the scintillator response for a specified material, density, and thickness.

    Args:
        energies (numpy.ndarray): 1D array of energy values (in keV) at which to calculate the scintillator response.
        mat (str): Chemical formula of the scintillator material (e.g., 'Al' for aluminum).
        den (float): Density of the scintillator material in g/cm³.
        thickness (float): Thickness of the scintillator in mm.

    Returns:
        numpy.ndarray: Array containing the filter response at each specified energy as a numpy array.
    """
    psb_mat = [Material(formula=mat, density=den)]
    sm = Scintillator(materials=psb_mat, thickness=(thickness, None, None))
    return sm(energies).detach().numpy()