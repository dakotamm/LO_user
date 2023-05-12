"""
Finding hypoxic depth and volume using observational data.

Test on mac in ipython:
run extract_casts_DM_new -gtx cas6_v0_live -source dfo1 -otype ctd -year 2017 -test False

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

import VFC_functions_2 as vfun

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

cast_start = [datetime(int(year_str),1,1), datetime(int(year_str),2,1), datetime(int(year_str),3,1), datetime(int(year_str),4,1), datetime(int(year_str),5,1),
              datetime(int(year_str),6,1), datetime(int(year_str),7,1), datetime(int(year_str),8,1), datetime(int(year_str),9,1), datetime(int(year_str),10,1),
              datetime(int(year_str),11,1), datetime(int(year_str),12,1)]

cast_end = [datetime(int(year_str),1,31), datetime(int(year_str),2,28), datetime(int(year_str),3,31), datetime(int(year_str),4,30), datetime(int(year_str),5,31),
              datetime(int(year_str),6,30), datetime(int(year_str),7,31), datetime(int(year_str),8,31), datetime(int(year_str),9,30), datetime(int(year_str),10,31),
              datetime(int(year_str),11,30), datetime(int(year_str),12,31)]

# THIS IS REALLY SILLY ^^^^ BUT FOR SIMPLICITY FOR NOW, oddly challenging...

segments = ['G1','G2','G3','G4','G5','G6']

sub_vol_dict_obs = {}

sub_vol_dict_LO = {}

var_array_dict = {}

sub_thick_dict_LO = {}

sub_thick_dict_obs = {}

surf_casts_array_dict = {}

# %%


info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')

fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ( year_str + '.p')

vol_dir, v_df, j_dict, i_dict, seg_list = vfun.getSegmentInfo(Ldir)


dt = pd.Timestamp('2022-01-01 01:30:00')
fn_his = cfun.get_his_fn_from_dt(Ldir, dt)

G, S, T, land_mask, Lon, Lat, z_rho_grid, dz, dv = vfun.getGridInfo(fn_his)


# %%

jjj_dict, iii_dict = vfun.defineSegmentIndices(segments, segments, j_dict, i_dict)


info_df = vfun.getCleanInfoDF(info_fn, land_mask, Lon, Lat, segments, jjj_dict, iii_dict)

# %%

for seg_name in segments:
    
    info_df_use = info_df[info_df['segment'] == seg_name]
        
    info_df_use = info_df_use[~np.isnan(info_df_use['jj_cast'])]
    
    info_df_use = info_df_use[~np.isnan(info_df_use['ii_cast'])]
    
    for (mon_num, mon_str, cast_st, cast_en) in zip(month_num, month_str, cast_start, cast_end):
                
        info_df_use_time = info_df_use[info_df_use['time'] <= cast_en]
        
        info_df_use_time  = info_df_use_time[info_df_use_time['time'] >= cast_st]
        
        if info_df_use_time.empty:
            continue
        else:
        
            dt = pd.Timestamp('2017-' + mon_num +'-01 01:30:00')
            fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
        
            vfun.extractLOCasts(Ldir, info_df_use_time, fn_his)
        
        
        
        


