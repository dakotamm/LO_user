"""
Finding hypoxic depth and volume using observational data.

Test on mac in ipython:
run obs_hyp_vol_DM -gtx cas6_v0_live -source dfo1 -otype ctd -year 2017 -test False

"""

import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from lo_tools import Lfun, zfun, zrfun
from lo_tools import extract_argfun as exfun
import cast_functions as cfun
from lo_tools import plotting_functions as pfun
import tef_fun as tfun
import pickle

import VFC_functions as vfun

from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import KDTree

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import itertools

from collections import defaultdict


Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

#month_num = ['01','02','03','04','05','06','07','08','09','10','11','12']

#month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

segments = ['G1']

sub_vol_dict_obs = {}

var_array_dict = {}

hyp_thick_dict_obs = {}

# get segment info
vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
v_df = pd.read_pickle(vol_dir / 'volumes.p')
j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
seg_list = list(v_df.index)

info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')

# dt = pd.Timestamp('2022-01-01 01:30:00')
# fn_his = cfun.get_his_fn_from_dt(Ldir, dt)

# #grid info
# G, S, T = zrfun.get_basic_info(fn_his)
# Lon = G['lon_rho'][0, :]
# Lat = G['lat_rho'][:, 0]
# z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
# dz = np.diff(z_w_grid,axis=0)
# dv = dz*G['DX']*G['DY']

# %%








