"""

run plot_segment_VFC -gtx cas6_v0_live -year 2017 -test False

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
from lo_tools import plotting_functions as pfun
import pickle

from time import time as Time

import matplotlib.pyplot as plt


import numpy as np

import VFC_functions_temp3 as vfun

# %%

Ldir = exfun.intro() # this handles the argument passing


# %%

with open('/Users/dakotamascarenas/Desktop/' + str(Ldir['year']) + '_data_dict_full.pkl', 'rb') as f: # (str(save_dir) + '/' + '2017_data_dict.pkl'), 'wb') as f: 
    data_dict_full = pickle.load(f)
    
# %%

dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)

if Ldir['testing']:
    
    month_num = ['12']
    
    month_str = ['Dec']

    # month_num =  ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    # month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 2 #mg/L DO

var_list = ['DO_mg_L', 'S_g_kg', 'T_deg_C', 'NO3_uM', 'NH4_uM', 'TA_uM', 'DIC_uM']

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

fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)

land_mask_plot = land_mask.copy()

land_mask_plot[land_mask ==0] = np.nan


for (mon_num, mon_str) in zip(month_num, month_str):
    
    info_df_use = info_df[info_df['month'] == int(mon_num)]
            
    df_use = df[df['month'] == int(mon_num)]
    
    for var in var_list:
        
        if var in df_use:
        
            for cell in range(np.size(z_rho_grid, axis=0)):
            
                pfun.start_plot(fs=14, figsize=(10,15))
    
                fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
                
                c00 = ax[0,0].pcolormesh(plon, plat, land_mask_plot, cmap='Greys')
            
                c0 = ax[0,0].pcolormesh(plon, plat, data_dict_full[int(mon_num)][var][cell,:,:]) #cmap = 'viridis', vmin = 0, vmax = 10)
    
                for cid in info_df_use.index:
                            
                    ax[0,0].plot(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
                if var == 'chlorophyll':
                    
                    units = '[mg m-3]'
                    
                    clim_min = 0.022
                    
                    clim_max = 0.028
                    
                elif var == 'S_g_kg':
                    
                    units = '[g/kg]'
                    
                    clim_min = 20
                    
                    clim_max = 35
                    
                elif var == 'T_deg_C':
                    
                    units = '[deg C]'
                    
                    clim_min = 4
                    
                    clim_max = 14
                    
                elif var == 'NO3_uM':
                    
                    units = '[uM]'
                    
                    clim_min = 0
                    
                    clim_max = 45
                    
                elif var == 'NH4_uM':
                    
                    units = '[uM]'
                    
                    clim_min = -0.1
                    
                    clim_max = 0.1
    
                elif var == 'DIC_uM':
    
                    units = '[uM]'
    
                    clim_min = 1700
    
                    clim_max = 2500
                    
                elif var == 'TA_uM':
                    
                    units = '[uM]'
                    
                    clim_min = 1900
                    
                    clim_max = 2500
                    
                elif var == 'DO_uM':
                    
                    units = '[uM]'
                    
                    clim_min = 0
                    
                    clim_max = 350
                    
                elif var == 'DO_mg_L':
                    
                    units = '[uM]'
                    
                    clim_min = 0
                    
                    clim_max = 14
                    
                ax[0,0].set_xlim([-125.5,-122])
                ax[0,0].set_ylim([46.5, 51])
                ax[0,0].tick_params(labelrotation=45)
                ax[0,0].set_title(mon_str + ' ' + str(Ldir['year']) + ' ' + var + ' Layer ' + str(cell))
                
                c0.set_clim(clim_min, clim_max)
                
                fig.colorbar(c0,ax=ax[0,0])
                                        
                pfun.add_coast(ax[0,0])
                pfun.dar(ax[0,0])
                
                # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/TEF_VFC_' + str(Ldir['year']) + '_' + mon_str + '_' + var + '_layer_' + str(cell) + '.png', bbox_inches='tight')
                
                plt.savefig('/Users/dakotamascarenas/Desktop/pltz/TEF_VFC_' + str(Ldir['year']) + '_' + mon_str + '_' + var + '_layer_' + '{:04}'.format(cell) + '.png', bbox_inches='tight')

                
            # pfun.start_plot(fs=14, figsize=(10,15))

            # fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
            
            # c00 = ax[0,0].pcolormesh(plon, plat, land_mask_plot, cmap='Greys')
        
            # c0 = ax[0,0].pcolormesh(plon, plat, data_dict_full[int(mon_num)][var][cell,:,:]) #cmap = 'viridis', vmin = 0, vmax = 10)

            # for cid in info_df_use.index:
                        
            #     ax[0,0].plot(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10)
                
            # ax[0,0].set_xlim([-123.5,-122])
            # ax[0,0].set_ylim([47, 48.5])
            # ax[0,0].tick_params(labelrotation=45)
            # ax[0,0].set_title(mon_str + ' ' + str(Ldir['year']) + ' ' + var + ' Layer ' + str(cell))
            
            # fig.colorbar(c0,ax=ax[0,0])
            
            # pfun.add_coast(ax[0,0])
            # pfun.dar(ax[0,0])
            
            # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/TEF_VFC_PS_' + str(Ldir['year']) + '_' + mon_str + '_' + var + '_layer_' + str(cell) + '.png', bbox_inches='tight')

# %%

