"""
Created on Mon Apr 10 17:19:05 2023

@author: dakotamascarenas
"""

import VFC_functions as vfun

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

from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import KDTree

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import itertools

from collections import defaultdict

# %%

# get segment info
vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
v_df = pd.read_pickle(vol_dir / 'volumes.p')
j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
seg_list = list(v_df.index)

info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')

dt = pd.Timestamp('2022-01-01 01:30:00')
fn_his = cfun.get_his_fn_from_dt(Ldir, dt)

#grid info
G, S, T = zrfun.get_basic_info(fn_his)
Lon = G['lon_rho'][0, :]
Lat = G['lat_rho'][:, 0]
z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
dz = np.diff(z_w_grid,axis=0)
dv = dz*G['DX']*G['DY']

in_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' /
    (Ldir['source'] + '_' + Ldir['otype'] + '_' + year_str))

plt.close('all')

pfun.start_plot(fs=14, figsize=(10,10))

fig0, axes0 = plt.subplots(nrows=1, ncols=1, squeeze=False)

pfun.add_coast(axes0[0,0])

# %%

jjj_dict, iii_dict = vfun.getAllSegmentIndices(info_fn, seg_list, j_dict, i_dict)

mask_idx = G['mask_rho']


# %%

var = 'oxygen'

threshold_val = 20

var_array_dict = {}

sub_vol_dict_LO = {}

sub_thick_dict_LO = {}

var_array_dict[dt] = {}

sub_vol_dict_LO[dt] = {}

sub_thick_dict_LO[dt] = {}

var_array_dict[dt]['all'], sub_vol_dict_LO[dt]['all'], sub_thick_dict_LO[dt]['all'] = vfun.getLOSubVolThick(fn_his, jjj_dict['all'], iii_dict['all'], var, threshold_val)

# %%

pfun.start_plot(fs=14, figsize=(10,10))

fig1, axes1 = plt.subplots(nrows=1, ncols=1, squeeze=False)

sub_thick = np.asarray(sub_thick_dict_LO[dt]['all'])

sub_thick_bin = sub_thick.copy()

sub_thick_bin[sub_thick > 0] = 1

#pfun.add_coast(axes0[0,0])

#axes1[0,0].pcolormesh(Lon[np.unique(iii_dict['all'])],Lat[np.unique(jjj_dict['all'])],sub_thick_dict_LO[dt]['all'],cmap='summer')

mask_idx_sliced = mask_idx[min(jjj_dict['all']):max(jjj_dict['all'])+1,min(iii_dict['all']):max(iii_dict['all'])+1]

dif = mask_idx_sliced - sub_thick_bin

thing = axes1[0,0].pcolormesh(Lon[np.unique(iii_dict['all'])],Lat[np.unique(jjj_dict['all'])], dif, vmin = -1, vmax = 1)



clb = plt.colorbar(thing, ax = axes1[0,0])

clb.ax.set_title('mask_rho - segments')

axes1[0,0].set_xlim(Lon[min(iii_dict['all'])],Lon[max(iii_dict['all']) + 1])
axes1[0,0].set_ylim(Lat[min(jjj_dict['all'])],Lat[max(jjj_dict['all']) + 1])

#pfun.add_coast(axes1[0,0])

fig1.tight_layout()





