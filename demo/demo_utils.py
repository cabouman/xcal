# Basic Packages
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import svmbir
import h5py

from xspec.chem_consts import get_lin_att_c_vs_E
from xspec.dictSE import cal_fw_mat
from xspec._utils import Gen_Circle
import spekpy as sp  # Import SpekPy
from xspec.defs import *
from xspec import paramSE
from xspec.chem_consts._periodictabledata import density

import torch

def read_mv_hdf5(file_name):
    data = []
    with h5py.File(file_name, 'r') as f:
        for key in f.keys():
            grp_i = f[key]
            dict_i = {}
            for sub_key in grp_i.keys():
                if isinstance(grp_i[sub_key], h5py.Group):
                    dict_i[sub_key] = {k: v for k, v in grp_i[sub_key].attrs.items()}
                else:
                    dict_i[sub_key] = np.array(grp_i[sub_key])
            data.append(dict_i)
    return data


def gen_datasets_3_voltages():
