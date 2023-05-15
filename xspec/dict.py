import numpy as np
from xspec.chem_consts._consts_from_table import get_lin_att_c_vs_E, get_lin_absp_c_vs_E


def _obtain_attenuation(energies, formula, density, thickness):
    # thickness is mm
    mu = get_lin_att_c_vs_E(density, formula, energies)
    att = np.exp(-mu * thickness)
    return att


def gen_filts_specD(energies, composition=[]):
    src_fltr_dict = []
    src_fltr_info_dict = []

    # Generate source filter spectrum dictionary
    for sfp in composition:
        formula = sfp['formula']
        density = sfp['density']
        thickness_list = sfp['thickness_list']
        src_fltr_dict += [_obtain_attenuation(energies, formula, density, thickness) for thickness in thickness_list]
        src_fltr_info_dict += [(formula, thickness) for thickness in thickness_list]
    src_fltr_dict = np.array(src_fltr_dict)
    return src_fltr_dict, src_fltr_info_dict


def _obtain_absorption(energies, formula, density, thickness):
    mu_en = get_lin_absp_c_vs_E(density, formula, energies)
    mu = get_lin_att_c_vs_E(density, formula, energies)
    absr = energies*mu_en/mu*(1-np.exp(-mu*thickness))
    return absr


def gen_scints_specD(energies, composition=[]):
    src_scint_dict = []
    src_scint_info_dict = []

    # Generate scintillator response dictionary
    for sfp in composition:
        formula = sfp['formula']
        density = sfp['density']
        thickness_list = sfp['thickness_list']
        src_scint_dict += [_obtain_absorption(energies, formula, density, thickness) for thickness in thickness_list]
        src_scint_info_dict += [(formula, thickness) for thickness in thickness_list]
    src_scint_dict = np.array(src_scint_dict)
    return src_scint_dict, src_scint_info_dict
