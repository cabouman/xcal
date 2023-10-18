import numpy as np
import torch
from xspec.chem_consts._consts_from_table import get_lin_att_c_vs_E, get_lin_absp_c_vs_E

from xspec._defs import *

def _obtain_attenuation(energies, formula, density, thickness, torch_mode=False):
    # thickness is mm
    if formula == 'air':
        att = np.ones(energies.shape)
    else:
        mu = get_lin_att_c_vs_E(density, formula, energies)
        if torch_mode:
            mu = torch.tensor(mu)
            att = torch.exp(-mu * thickness)
        else:
            att = np.exp(-mu * thickness)
    return att

def gen_fltr_res(energies, fltr_mat:Material, fltr_th, torch_mode=True):
    if torch_mode:
        fltr_res = torch.ones(energies.shape)
    else:
        fltr_res = np.ones(energies.shape)

    for fm, fth in zip(fltr_mat,fltr_th):
        fltr_res *= _obtain_attenuation(energies, fm.formula, fm.density, fth, torch_mode)
    return fltr_res

def gen_filts_specD(energies, composition=[], torch_mode=False):
    src_fltr_dict = []
    src_fltr_info_dict = []

    # Generate source filter spectrum dictionary
    for sfp in composition:
        formula = sfp['formula']
        density = sfp['density']
        thickness_list = sfp['thickness_list']
        src_fltr_dict += [_obtain_attenuation(energies, formula, density, thickness, torch_mode) for thickness in thickness_list]
        src_fltr_info_dict += [(formula, thickness) for thickness in thickness_list]

    if not torch_mode:
        src_fltr_dict = np.array(src_fltr_dict)
    else:
        src_fltr_dict = torch.stack(src_fltr_dict,0)
    return src_fltr_dict, src_fltr_info_dict


def _obtain_absorption(energies, formula, density, thickness, torch_mode=False):
    mu_en = get_lin_absp_c_vs_E(density, formula, energies)
    mu = get_lin_att_c_vs_E(density, formula, energies)
    if torch_mode:
        energies = torch.Tensor(energies) if energies is not torch.Tensor else energies
        mu = torch.tensor(mu)
        mu_en =torch.tensor(mu_en)
        absr = energies*mu_en/mu*(1-torch.exp(-mu*thickness))
    else:
        absr = energies*mu_en/mu*(1-np.exp(-mu*thickness))
    return absr

def gen_scint_cvt_func(energies, scint_mat:Material, scint_th, torch_mode=True):
    return _obtain_absorption(energies, scint_mat.formula, scint_mat.density, scint_th, torch_mode)

def gen_scints_specD(energies, composition=[], torch_mode=False):
    src_scint_dict = []
    src_scint_info_dict = []

    # Generate scintillator response dictionary
    for sfp in composition:
        formula = sfp['formula']
        density = sfp['density']
        thickness_list = sfp['thickness_list']
        src_scint_dict += [_obtain_absorption(energies, formula, density, thickness, torch_mode) for thickness in thickness_list]
        src_scint_info_dict += [(formula, thickness) for thickness in thickness_list]
    if not torch_mode:
        src_scint_dict = np.array(src_scint_dict)
    else:
        src_scint_dict = torch.stack(src_scint_dict, 0)
    return src_scint_dict, src_scint_info_dict
