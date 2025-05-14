import os
import numpy as np
import torch
from xcal.models import Base_Spec_Model
from xcal.defs import Material
from xcal.chem_consts._consts_from_table import get_lin_att_c_vs_E
import matplotlib.pyplot as plt

# Implement the analytical model for filter.
def _obtain_attenuation(energies, formula, density, thickness, torch_mode=False):
    # thickness is mm
    mu = get_lin_att_c_vs_E(density, formula, energies)
    if torch_mode:
        mu = torch.tensor(mu)
        att = torch.exp(-mu * thickness)
    else:
        att = np.exp(-mu * thickness)
    return att

def gen_fltr_res(energies, fltr_mat:Material, fltr_th:float, torch_mode=True):

    return _obtain_attenuation(energies, fltr_mat.formula, fltr_mat.density, fltr_th, torch_mode)

# Gradient descent module.
class Filter(Base_Spec_Model):
    def __init__(self, materials, thickness):
        """
        A template filter model based on Beer's Law and NIST mass attenuation coefficients, including all necessary methods.

        Args:
            materials (list): A list of possible materials for the filter,
                where each material should be an instance containing formula and density.
            thickness (tuple or list): If a tuple, it should be (initial value, lower bound, upper bound) for the filter thickness.
                If a list, it should have the same length as the materials list, specifying thickness for each material.
                These values cannot be all None. It will not be optimized when lower == upper.
        """
        if isinstance(thickness, tuple):
            if all(t is None for t in thickness):
                raise ValueError("Thickness tuple cannot have all None values.")
            params_list = [{'material': mat, 'thickness': thickness} for mat in materials]
        elif isinstance(thickness, list):
            if len(thickness) != len(materials):
                raise ValueError("Length of thickness list must match length of materials list.")
            params_list = [{'material': mat, 'thickness': th} for mat, th in zip(materials, thickness)]
        else:
            raise TypeError("Thickness must be either a tuple or a list.")

        super().__init__(params_list)

    def forward(self, energies):
        """
        Takes X-ray energies and returns the filter response.

        Args:
            energies (torch.Tensor): A tensor containing the X-ray energies of a poly-energetic source in units of keV.

        Returns:
            torch.Tensor: The filter response as a function of input energies, selected material, and its thickness.
        """
        # Retrieves
        mat = self.get_params()[f"{self.prefix}_material"]
        th = self.get_params()[f"{self.prefix}_thickness"]
        energies = torch.tensor(energies, dtype=torch.float32) if not isinstance(energies, torch.Tensor) else energies
        return gen_fltr_res(energies, mat, th)

if __name__ == '__main__':

    result_folder = 'T04_output'
    os.makedirs(result_folder,exist_ok=True)

    th = 2.5  # target thickness in um

    psb_fltr_mat = [Material(formula='Al', density=2.702), Material(formula='Cu', density=8.92)]
    filter_1 = Filter(psb_fltr_mat, thickness=(th, 0, 10))

    ee = np.linspace(1, 150, 150)
    ff = filter_1(ee)
    est_param = filter_1.get_params()
    print(f'{est_param}')

    plt.plot(ee, ff.data)
    plt.savefig(f'{result_folder}/T04_fltr.png')
