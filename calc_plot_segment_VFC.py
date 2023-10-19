"""

run calc_plot_segment_VFC -gtx cas6_v0_live -year 2017 -test False

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
from lo_tools import plotting_functions as pfun
import pickle

from time import time as Time

import matplotlib.pyplot as plt


import numpy as np

import VFC_functions_temp2 as vfun

# %%

Ldir = exfun.intro() # this handles the argument passing


# %%

# TEMP

with open('/Users/dakotamascarenas/Desktop/2017_data_dict_full.pkl', 'rb') as f: # (str(save_dir) + '/' + '2017_data_dict.pkl'), 'wb') as f: 
    data_dict_full = pickle.load(f)
    
with open('/Users/dakotamascarenas/Desktop/2017_data_dict.pkl', 'rb') as f: # (str(save_dir) + '/' + '2017_data_dict.pkl'), 'wb') as f: 
    data_dict = pickle.load(f)
    
# %%

dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)

if Ldir['testing']:
    
    month_num = ['09']
    
    month_str = ['Sep']

    # month_num =  ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    # month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 2 #mg/L DO

threshold_depth_list = [30, 100]

layer_list = [0, -1 ]

#threshold_pct = 0.2 #m GOTTA BE PERCENT

var_list = ['DO_mg_L', 'S_g_kg', 'T_deg_C', 'NO3_uM', 'NH4_uM', 'TA_uM', 'DIC_uM', 'DO_uM']

segments = 'regions' #custom (specify string list and string build list), basins, whole domain, sound and strait

info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))

# %%


jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

# %%

info_df, df = vfun.getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var_list)

# %%

info_df['month'] = info_df['time'].dt.month

df['month'] = df['time'].dt.month

# %%

if Ldir['testing']:

    var_list = ['DO_mg_L']
    
# %%

hyp_thick_dict = {}


hyp_vol_df = pd.DataFrame()

hyp_vol_df['segment'] = []

hyp_vol_df['month'] = []

hyp_vol_df['vol_km3'] = []


avg_below_df = pd.DataFrame()

avg_below_df['segment'] = []

avg_below_df['month'] = []

avg_below_df['threshold_depth'] = []

avg_below_df['var'] = []

avg_below_df['avg'] = []


avg_layer_df = pd.DataFrame()

avg_layer_df['segment'] = []

avg_layer_df['month'] = []

avg_layer_df['layer'] = []

avg_layer_df['var'] = []

avg_layer_df['avg'] = []


for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    hyp_thick_dict[seg_name] = {}

    for (mon_num, mon_str) in zip(month_num, month_str):
        
        hyp_thick_dict[seg_name][int(mon_num)] = {}
        
        for var in var_list:
            
            var_array = data_dict[seg_name][int(mon_num)][var].copy()
            
            if var == 'DO_mg_L':
            
                hyp_thick_dict[seg_name][int(mon_num)][var] = vfun.getSubThick(var_array, threshold_val, jjj, iii, dz)
                
                hyp_vol_df_temp = pd.DataFrame()
                
                hyp_vol_df_temp['segment'] = [seg_name]
                
                hyp_vol_df_temp['month'] = [mon_num]
                
                hyp_vol_df_temp['vol_km3'] = vfun.getSubVol(var_array, threshold_val, jjj, iii, dv)*1e-9 # convert to km^3 from m^3
                
                hyp_vol_df = pd.concat([hyp_vol_df, hyp_vol_df_temp], ignore_index=True)
                
            for threshold_depth in threshold_depth_list:
            
                avg_below_df_temp = pd.DataFrame()
                
                avg_below_df_temp['segment'] = [seg_name]
    
                avg_below_df_temp['month'] = [mon_num]
    
                avg_below_df_temp['threshold_depth'] = [threshold_depth]
    
                avg_below_df_temp['var'] = [var]
    
                avg_below_df_temp['avg'] = vfun.getAvgBelow(var_array, threshold_depth, jjj, iii, z_rho_grid)
                
                avg_below_df = pd.concat([avg_below_df, avg_below_df_temp], ignore_index=True)
                
            for layer in layer_list:
                
                avg_layer_df_temp = pd.DataFrame()
                
                avg_layer_df_temp['segment'] = [seg_name]
    
                avg_layer_df_temp['month'] = [mon_num]
    
                avg_layer_df_temp['layer'] = [layer]
    
                avg_layer_df_temp['var'] = [var]
    
                avg_layer_df_temp['avg'] = vfun.getAvgLayer(var_array, layer)
                
                avg_layer_df = pd.concat([avg_layer_df, avg_layer_df_temp], ignore_index=True)
            
            
# %%


      
            
            
            
            
            
            
            
            
            
            
            
