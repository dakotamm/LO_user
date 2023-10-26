"""

run  -gtx cas6_v0_live -year 2017 -test False

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
import pickle

from time import time as Time

import matplotlib.pyplot as plt


import numpy as np

import VFC_functions_temp1 as vfun

# %%

Ldir = exfun.intro() # this handles the argument passing


# %%

with open('/Users/dakotamascarenas/Desktop/2017_data_dict_full.pkl', 'rb') as f: # (str(save_dir) + '/' + '2017_data_dict.pkl'), 'wb') as f: 
    data_dict_full = pickle.load(f)
    
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

var_list = ['DO_mg_L', 'S_g_kg', 'T_deg_C']

segments = 'all' #custom (specify string list and string build list), basins, whole domain, sound and strait

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

fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)

land_mask_plot = land_mask.copy()

land_mask_plot[land_mask ==0] = np.nan


for (mon_num, mon_str) in zip(month_num, month_str):
    
    info_df_use = info_df[info_df['month'] == int(mon_num)]
            
    df_use = df[df['month'] == int(mon_num)]
    
    for var in var_list:
        
        for cell in range(np.size(z_rho_grid, axis=0)):
        
            fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
            
            c00 = ax[0,0].pcolormesh(plon, plat, land_mask_plot, cmap='Greys')
        
            c0 = ax[0,0].pcolormesh(plon, plat, data_dict_full[int(mon_num)][var][cell,:,:]) #cmap = 'viridis', vmin = 0, vmax = 10)

            for cid in info_df_use.index:
                        
                ax[0,0].plot(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10)
                
            ax[0,0].set_xlim([min(plon[0]),max(plon[0])])
            ax[0,0].set_ylim([min(plat[0]), max(plat[-1])])
            ax[0,0].tick_params(labelrotation=45)
            ax[0,0].set_title(var + 'Layer ' + str(cell))
            
            fig.colorbar(c0,ax=ax[0,0])
            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/TEF_VFC_'+var + '_layer_' + str(cell) + '.png', bbox_inches='tight')
