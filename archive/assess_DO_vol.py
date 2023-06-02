"""
Finding hypoxic depth and volume using observational data.

Test on mac in ipython:
run assess_DO_vol -gtx cas6_v0_live -year 2017 -test False

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

import os


Ldir = exfun.intro() # this handles the argument passing

dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
fn_his = cfun.get_his_fn_from_dt(Ldir, dt)

month_num = ['01','02','03','04','05','06','07','08','09','10','11','12']

month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 5 #mg/L DO

var = 'DO_mg_L'

segments = 'basins' #custom (specify string list and string build list), basins, whole domain, sound and strait

# seg_build_list = optional
    
G, S, T, land_mask, Lon, Lat, z_rho_grid, dz, dv = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)

info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))


# %%


jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

# %%

info_df = vfun.getCleanInfoDF(info_fn, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict)

df = vfun.getCleanDF(fn, info_df)

# %%

info_df['month'] = info_df['time'].dt.month

df['month'] = df['time'].dt.month

# %%

sub_thick_LO_his = {}

sub_thick_LO_casts = {}

sub_thick_obs = {}

sub_vol_LO_his = {}

sub_vol_LO_casts = {}

sub_vol_obs = {}

sub_thick_obs = {}

ii_casts = {}

jj_casts = {}

cid_dict = {}

surf_casts_array = {}

sub_casts_array_obs = {}

sub_casts_array_LO_casts = {}

# %%


for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    sub_thick_LO_his[seg_name] = {}

    sub_thick_LO_casts[seg_name] = {}

    sub_thick_obs[seg_name] = {}

    sub_vol_LO_his[seg_name] = {}

    sub_vol_LO_casts[seg_name] = {}
    
    sub_vol_obs[seg_name] = {}

    sub_thick_obs[seg_name] = {}

    ii_casts[seg_name] = {}

    jj_casts[seg_name] = {}
    
    cid_dict[seg_name] = {}
    
    surf_casts_array[seg_name] = {}
    
    sub_casts_array_obs[seg_name] = {}
    
    sub_casts_array_LO_casts[seg_name] = {}

    
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
        fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
        
        G, S, T, land_mask, Lon, Lat, z_rho_grid, dz, dv = vfun.getGridInfo(fn_his)
        
        sub_vol_LO_his[seg_name][int(mon_num)], sub_thick_LO_his[seg_name][int(mon_num)] = vfun.getLOHisSubVolThick(dv, dz, fn_his, jjj, iii, var, threshold_val)
        
        info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
                
        df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
        
        sub_vol_obs[seg_name][int(mon_num)], sub_thick_obs[seg_name][int(mon_num)], surf_casts_array[seg_name][int(mon_num)], sub_casts_array_obs[seg_name][int(mon_num)] = vfun.getOBSCastsSubVolThick(info_df_use, df_use, var, threshold_val, z_rho_grid, dv, dz, land_mask, jjj, iii)
                        
        jj_casts[seg_name][int(mon_num)] = info_df_use['jj_cast'].to_numpy()
        
        ii_casts[seg_name][int(mon_num)] = info_df_use['ii_cast'].to_numpy()
        
        cid_dict[seg_name][int(mon_num)] =info_df_use.index.to_numpy()
        
        vfun.extractLOCasts(Ldir, info_df_use, fn_his)
        
        sub_vol_LO_casts[seg_name][int(mon_num)], sub_thick_LO_casts[seg_name][int(mon_num)], sub_casts_array_LO_casts[seg_name][int(mon_num)] = vfun.getLOCastsSubVolThick(Ldir, info_df_use, var, threshold_val, z_rho_grid, dv, dz, land_mask, jjj, iii, surf_casts_array[seg_name][int(mon_num)])
        
        print(seg_name + mon_str)
        


# %% 

seg_list = ['Whidbey Basin']

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    iii = iii_dict[seg_name]
    
    
    min_lat = Lat[min(jjj) - 10]
    max_lat = Lat[max(jjj) + 10]
    
    min_lon = Lon[min(iii) - 10]
    max_lon = Lon[max(iii) + 10]    
    
    
    for (mon_num, mon_str) in zip(month_num,month_str):
         
        pfun.start_plot(fs=14, figsize=(16,18))
        fig0, axes0 = plt.subplots(nrows=2, ncols=1, squeeze=False)
        
        c0 = axes0[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_LO_his[seg_name][int(mon_num)], cmap='Blues', alpha = 0.8, vmin = 0, vmax = 300)
        
        axes0[0,0].set_xlim([min_lon,max_lon])
        axes0[0,0].set_ylim([min_lat,max_lat])
        axes0[0,0].tick_params(labelrotation=45)
        axes0[0,0].set_title('LO ' + mon_str + ' ' + str(Ldir['year']) + ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[0,0])
        
        
        c1 = axes0[1,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_obs[seg_name][int(mon_num)], cmap='Greens', alpha = 0.8, vmin = 0, vmax = 300)
        
        for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
            axes0[1,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        axes0[1,0].set_xlim([min_lon,max_lon])
        axes0[1,0].set_ylim([min_lat,max_lat])
        axes0[1,0].tick_params(labelrotation=45)
        axes0[1,0].set_title('Obs VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[1,0])
        
        
        # c1 = axes0[1,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_LO_casts[seg_name][int(mon_num)], cmap='Purples', alpha = 0.8, vmin = 0, vmax = 300)
        
        # for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
        #     axes0[1,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        # axes0[1,0].set_xlim([min_lon,max_lon])
        # axes0[1,0].set_ylim([min_lat,max_lat])
        # axes0[1,0].tick_params(labelrotation=45)
        # axes0[1,0].set_title('LO VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        # pfun.add_coast(axes0[1,0])
        
        
        # c2 = axes0[2,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_obs[seg_name][int(mon_num)], cmap='Greens', alpha = 0.8, vmin = 0, vmax = 300)
        
        # for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
        #     axes0[2,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        # axes0[2,0].set_xlim([min_lon,max_lon])
        # axes0[2,0].set_ylim([min_lat,max_lat])
        # axes0[2,0].tick_params(labelrotation=45)
        # axes0[2,0].set_title('Obs VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        # pfun.add_coast(axes0[2,0])
        
        fig0.colorbar(c0,ax=axes0[0,0], label = 'Subthreshold Thickness [m]')
        
        fig0.colorbar(c1,ax=axes0[1,0], label = 'Subthreshold Thickness [m]')
        
        #fig0.colorbar(c2,ax=axes0[2,0], label = 'Subthreshold Thickness [m]')
        
        fig0.tight_layout()
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+seg_name + '_sub_thick_'+str(threshold_val)+'_mg_L_DO_casts_' + str(Ldir['year']) + '_00' + mon_num+'.png')

                                                      
        
# %%

pfun.start_plot(fs=14, figsize=(16,9))
fig1, axes1 = plt.subplots(nrows=1, ncols=1, squeeze=False)
plt.grid()

for seg_name in seg_list:

    # vol_LO = sub_vol_LO_his[seg_name]
    
    # vol_LO = sorted(vol_LO.items())
            
    # x_LO, y_LO = zip(*vol_LO)
    
    # #x_LO = int(x_LO)
    
    # y_LO = np.multiply(y_LO, 1e-9)
            
    # plt.plot(x_LO,y_LO,label = 'LO', linestyle = '--')

   #  vol_casts = sub_vol_LO_casts[seg_name]
    
   #  vol_casts = sorted(vol_casts.items())
            
   #  x_casts, y_casts = zip(*vol_casts)
    
   # # x_obs = int(x_obs)
    
   #  y_casts = np.multiply(y_casts, 1e-9)
            
   #  plt.plot(x_casts,y_casts,label = seg_name, linestyle = '-.')

    vol_obs = sub_vol_obs[seg_name]
    
    vol_obs = sorted(vol_obs.items())
            
    x_obs, y_obs = zip(*vol_obs)
    
   # x_obs = int(x_obs)
    
    y_obs = np.multiply(y_obs, 1e-9)
            
    plt.plot(x_obs,y_obs,label = seg_name)
    
    
axes1[0,0].set_xlabel('Months (2017)')
    
axes1[0,0].set_ylabel('Sub-'+str(threshold_val)+' mg/L DO Volume [km^3]')

plt.legend()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+ str(Ldir['year']) +'_sub_vol_'+str(threshold_val)+'_mg_L_DO.png')

# %%

import math

norm_RMSE_dict = {}

plt.close('all')

pfun.start_plot(fs=14, figsize=(10,10))

fig2, axes2 = plt.subplots(nrows=1, ncols=1, squeeze=False)

for seg_name in seg_list:
    
    cmap = cm.get_cmap('twilight', 12)
            
    y_LO = []

    y_obs = []  
    
    d = 0  
        
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        #dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
        
        axes2[0,0].plot(sub_vol_obs[seg_name][int(mon_num)]*1e-9, sub_vol_LO_his[seg_name][int(mon_num)]*1e-9, 'o', c=cmap(d), markersize = 10, label = mon_str)
        
        d+=1
        
        y_LO.append(sub_vol_LO_his[seg_name][int(mon_num)]*1e-9)
        
        y_obs.append(sub_vol_obs[seg_name][int(mon_num)]*1e-9)
                    
    y_LO = np.array(y_LO)
    
    y_obs = np.array(y_obs)
    
    x_1 = np.linspace(0, max(y_LO))
    
    y_1 = x_1
    
    axes2[0,0].plot(x_1,y_1, color = 'grey', alpha = 0.5)
    
    MSE = np.square(abs(np.subtract(y_obs,y_LO))).mean() # CHECK ON ABS
    
    RMSE = math.sqrt(MSE)
    
    norm_RMSE = RMSE/(y_obs.max()-y_obs.min())
    
    norm_RMSE_dict[seg_name] = norm_RMSE
    
    axes2[0,0].set_xlabel('Obs Sub 5 mg/L Vol [km^3]')
    axes2[0,0].set_ylabel('LO Sub 5 mg/L [km^3]')
    axes2[0,0].set_title(seg_name + ' Vol Comparison, Norm RMSE = '+str(round(norm_RMSE,3)))
    #n_c += 1
    
handles, labels = axes2[0,0].get_legend_handles_labels()
fig2.legend(handles, labels, bbox_to_anchor=(0, -0.2, 1, 0.2), loc="upper left",
                mode="expand", borderaxespad=0, ncol=12) #loc='upper center')
    
fig2.tight_layout()
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/comp_vol_'+seg_name+'.png',bbox_inches='tight')
    
    
                
        
        
# %%

import pickle

file_dir = '/Users/dakotamascarenas/Desktop/'


with open((file_dir + 'sub_casts_array_LO_casts.pkl'), 'wb') as f: 
    pickle.dump(sub_casts_array_LO_casts, f)

with open((file_dir + 'sub_casts_array_obs.pkl'), 'wb') as f: 
    pickle.dump(sub_casts_array_obs, f)      

with open((file_dir + 'sub_thick_LO_casts.pkl'), 'wb') as f: 
    pickle.dump(sub_thick_LO_casts, f)

with open((file_dir + 'sub_thick_LO_his.pkl'), 'wb') as f: 
    pickle.dump(sub_thick_LO_his, f)  
    
with open((file_dir + 'sub_thick_obs.pkl'), 'wb') as f: 
    pickle.dump(sub_thick_obs, f)

with open((file_dir + 'sub_vol_LO_casts.pkl'), 'wb') as f: 
    pickle.dump(sub_vol_LO_casts, f)      

with open((file_dir + 'sub_vol_LO_his.pkl'), 'wb') as f: 
    pickle.dump(sub_vol_LO_his, f)

with open((file_dir + 'sub_vol_obs.pkl'), 'wb') as f: 
    pickle.dump(sub_vol_obs, f)  
    
with open((file_dir + 'surf_casts_array.pkl'), 'wb') as f: 
    pickle.dump(surf_casts_array, f)  
    
with open((file_dir + 'jj_casts.pkl'), 'wb') as f: 
    pickle.dump(jj_casts, f)

with open((file_dir + 'ii_casts.pkl'), 'wb') as f: 
    pickle.dump(ii_casts, f)  
    
with open((file_dir + 'cid_dict.pkl'), 'wb') as f: 
    pickle.dump(cid_dict, f)      
                 


# %%

