"""
Finding hypoxic volume using VFC to compare to LO output.

Created on 2023/03/23

Test on mac:
run find_VFC_LO_vol_DM -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test False

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


#%%

surf_casts_array_dict = {}

ii_cast_dict = {}

jj_cast_dict = {}

var = 'oxygen'

threshold_val = 4 #mg/L

cast_start = datetime(2019,6,1)

cast_end = datetime(2019,8,30)

#%%

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
        
        fn_list = list(sorted(in_dir.glob('*' + seg_name + '_6-8_2019_' + str(dt.month) + '_2022.nc')))
        
        sub_vol_dict_obs[dt][seg_name], sub_thick_dict_obs[dt][seg_name] = vfun.getCastsSubVolThick(in_dir, fn_list, var, threshold_val, fn_his, j_dict[seg_name], i_dict[seg_name], ii_cast_dict[dt][seg_name], surf_casts_array_dict[dt][seg_name], var_array_dict[dt][seg_name])

        print(mon_str + ' ' + seg_name)
# %%

sect_df = tfun.get_sect_df('cas6')

min_lat = [48, 48.4]
max_lat = [49, 48.7]
min_lon = [-124, -123.4]
max_lon = [-122.25,-122.4]

import math

plt.close('all')

pfun.start_plot(fs=14, figsize=(14,14))

fig0, axes0 = plt.subplots(nrows=2, ncols=2, squeeze=False)

n_r = 0

n_c = 0

for seg_name in segments:
        
    cmap = cm.get_cmap('twilight', 12)
    
    y_LO = []

    y_obs = []  

    d = 0      
    
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
        
        axes0[n_r,n_c].plot(sub_vol_dict_LO[dt][seg_name]*1e-9, sub_vol_dict_obs[dt][seg_name]*1e-9, 'o', c=cmap(d), markersize = 10, label = mon_str)
        
        d+=1
        
        y_LO.append(sub_vol_dict_LO[dt][seg_name]*1e-9)
        
        y_obs.append(sub_vol_dict_obs[dt][seg_name]*1e-9)
                    
    y_LO = np.array(y_LO)
    
    y_obs = np.array(y_obs)
    
    x_1 = np.linspace(0, max(y_LO))
    
    y_1 = x_1
    
    axes0[n_r,n_c].plot(x_1,y_1, color = 'grey', alpha = 0.5)
    
    MSE = np.square(np.subtract(y_LO,y_obs)).mean()
    
    RMSE = math.sqrt(MSE)
    
    norm_RMSE = RMSE/(y_LO.max()-y_LO.min())
        
    axes0[n_r,n_c].set_xlabel('LO Sub 4 mg/L Vol [km^3]')
    axes0[n_r,n_c].set_ylabel('VFC Sub 4 mg/L [km^3]')
    axes0[n_r,n_c].set_title(seg_name + ' Vol Comparison, Norm RMSE = '+str(round(norm_RMSE,3)))
    n_c += 1
    
    if n_c > 1:
        n_r += 1
        n_c = 0
    
    print(str(n_r) + ' ' + str(n_c))
    
handles, labels = axes0[1,1].get_legend_handles_labels()
fig0.legend(handles, labels, bbox_to_anchor=(0, -0.2, 1, 0.2), loc="upper left",
                mode="expand", borderaxespad=0, ncol=12) #loc='upper center')
    
fig0.tight_layout()
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/comp_vol_G.png',bbox_inches='tight')


# %%






