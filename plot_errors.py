"""
Plot casts and color by normalized RMSE from VFC method vs. LO output

Created on 2023/03/23

Test on mac:
run plot_errors -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test False

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

import math


Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

month_num = ['01','02','03','04','05','06','07','08','09','10','11','12']

month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

segments = ['G1','G2','G3','G4','G5','G6']

sub_vol_dict_obs = {}

sub_vol_dict_LO = {}

var_array_dict = {}

sub_thick_dict_LO = {}

sub_thick_dict_obs = {}

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

surf_casts_array_dict = {}

ii_cast_dict = {}

jj_cast_dict = {}

var = 'oxygen'

threshold_val = 4 #mg/L

cast_start = datetime(2019,6,1)

cast_end = datetime(2019,6,30)

norm_RMSE_dict = {}


# %%

for (mon_num, mon_str) in zip(month_num,month_str):

    dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
    fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
    
    var_array_dict[dt] = {}
    
    sub_thick_dict_LO[dt] = {}
    
    sub_thick_dict_obs[dt] = {}
    
    surf_casts_array_dict[dt] = {}
    
    jj_cast_dict[dt] = {}
    
    ii_cast_dict[dt] = {}
    
    sub_vol_dict_LO[dt] = {}
    
    sub_vol_dict_obs[dt] = {}
    
    for seg_name in segments:
        
        var_array_dict[dt][seg_name], sub_vol_dict_LO[dt][seg_name], sub_thick_dict_LO[dt][seg_name] = vfun.getLOSubVolThick(fn_his, j_dict[seg_name], i_dict[seg_name], var, threshold_val)
        
        surf_casts_array_dict[dt][seg_name], jj_cast_dict[dt][seg_name], ii_cast_dict[dt][seg_name] = vfun.assignSurfaceToCasts(Ldir, info_fn, cast_start, cast_end, Lon, Lat, j_dict[seg_name], i_dict[seg_name], G['mask_rho'])        
        
        fn_list = list(sorted(in_dir.glob('*' + seg_name + '_6-6_2019_' + str(dt.month) + '_2022.nc')))
        
        sub_vol_dict_obs[dt][seg_name], sub_thick_dict_obs[dt][seg_name] = vfun.getCastsSubVolThick(in_dir, fn_list, var, threshold_val, fn_his, j_dict[seg_name], i_dict[seg_name], ii_cast_dict[dt][seg_name], surf_casts_array_dict[dt][seg_name], var_array_dict[dt][seg_name])

        print(mon_str + ' ' + seg_name)
        
# %%

for seg_name in segments:
            
    y_LO = []

    y_obs = []  
        
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
        
        y_LO.append(sub_vol_dict_LO[dt][seg_name]*1e-9)
        
        y_obs.append(sub_vol_dict_obs[dt][seg_name]*1e-9)
                    
    y_LO = np.array(y_LO)
    
    y_obs = np.array(y_obs)
    
    MSE = np.square(np.subtract(y_LO,y_obs)).mean()
    
    RMSE = math.sqrt(MSE)
    
    norm_RMSE = RMSE/(y_LO.max()-y_LO.min())
    
    norm_RMSE_dict[seg_name] = norm_RMSE
    
# %%

all_j_idx = []

all_i_idx = []

norm_RMSE_array = np.zeros(np.shape(z_rho_grid[0]))



for seg_name in segments:
    
    all_j_idx.extend(j_dict[seg_name])
    
    all_i_idx.extend(i_dict[seg_name])
    
    norm_RMSE_array[j_dict[seg_name],i_dict[seg_name]] = norm_RMSE_dict[seg_name]
    
norm_RMSE_array[norm_RMSE_array == 0] = np.nan

norm_RMSE_array_sliced = norm_RMSE_array[min(all_j_idx):max(all_j_idx)+1,min(all_i_idx):max(all_i_idx)+1]


# %%

plt.close('all')
pfun.start_plot(fs=14, figsize=(10,10))
fig0, axes0 = plt.subplots(nrows=1, ncols=1, squeeze=False)
cf = axes0[0,0].pcolormesh(Lon[np.unique(all_i_idx)],Lat[np.unique(all_j_idx)],norm_RMSE_array_sliced, cmap = 'YlOrRd', vmin = 0.06, vmax = 0.23)
for seg_name in segments:
    for m in range(len(ii_cast_dict[dt][seg_name])):
        axes0[0,0].plot(Lon[ii_cast_dict[dt][seg_name][m]],Lat[jj_cast_dict[dt][seg_name][m]],'o',markeredgecolor='black', markerfacecolor="lightgrey",markersize=5)
        
axes0[0,0].tick_params(labelrotation=45)
pfun.add_coast(axes0[0,0])
pfun.dar(axes0[0,0])
fig0.colorbar(cf, ax=axes0[0,0])
axes0[0,0].set_xlim([min(Lon[np.unique(all_i_idx)]),max(Lon[np.unique(all_i_idx)])])
axes0[0,0].set_ylim([min(Lat[np.unique(all_j_idx)]),max(Lat[np.unique(all_j_idx)])])

plt.title('DFO Strait of Georgia Normalized RMSE (2022)')
fig0.tight_layout()
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/dfo_g_norm_RMSE_2022.png') # need to not hardcode


