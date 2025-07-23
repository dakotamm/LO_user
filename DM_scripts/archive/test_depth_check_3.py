"""
Blah

Test on mac in ipython:
run test_depth_check_3 -gtx cas6_v0_live -year 2017 -test False

"""

import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from lo_tools import Lfun, zfun, zrfun
from lo_tools import extract_argfun as exfun
import cast_functions as cfun #remove local dependency
from lo_tools import plotting_functions as pfun
import tef_fun as tfun #remove local dependency
import pickle

#import VFC_functions_3 as vfun3

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

import os

import seaborn as sns

import copy


Ldir = exfun.intro() # this handles the argument passing

# %%

dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
fn_his = cfun.get_his_fn_from_dt(Ldir, dt)

if Ldir['testing']:
    
    month_num = ['09']
    
    month_str = ['Sep']

    # month_num =  ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    # month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 2 #mg/L DO

var = 'DO_mg_L'

segments = 'basins' #custom (specify string list and string build list), basins, whole domain, sound and strait

# seg_build_list = optional
    
G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)

info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))

# %%

jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

if Ldir['testing']:

    seg_list = ['Strait of Juan de Fuca']
    
# %%

info_df, df = vfun.getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var)

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

    ii_casts[seg_name] = {}

    jj_casts[seg_name] = {}
    
    cid_dict[seg_name] = {}
    
    surf_casts_array[seg_name] = {}
    
    sub_casts_array_obs[seg_name] = {}
    
    sub_casts_array_LO_casts[seg_name] = {}
    
    
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
        
        df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
        
        
        dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
        
        fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
        
        G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)
        
        
        sub_vol_LO_his[seg_name][int(mon_num)], sub_thick_LO_his[seg_name][int(mon_num)] = vfun.getLOHisSubVolThick(dv, dz, fn_his, jjj, iii, var, threshold_val)
        
        
        surf_casts_array[seg_name][int(mon_num)] = vfun.assignSurfaceToCasts(info_df_use, jjj, iii)
        
        
        vfun.extractLOCasts(Ldir, info_df_use, fn_his)
        
        
        #z_rho_grid_obs_TEMP = z_rho_grid.copy()
        
        #z_rho_grid_LO_casts_TEMP = z_rho_grid.copy()
        
        #surf_casts_array_LO_casts_TEMP = copy.deepcopy(surf_casts_array[seg_name][int(mon_num)])
        
        #surf_casts_array_obs_TEMP = copy.deepcopy(surf_casts_array[seg_name][int(mon_num)])
        
        #info_df_use_LO_casts_TEMP = info_df_use.copy(deep=True)
        
        #info_df_use_obs_TEMP = info_df_use.copy(deep=True)
        
        
        #sub_vol_obs[seg_name][int(mon_num)], sub_thick_obs[seg_name][int(mon_num)], sub_casts_array_obs[seg_name][int(mon_num)] = vfun.getOBSCastsSubVolThick(info_df_use_obs_TEMP, df_use, var, threshold_val, z_rho_grid_obs_TEMP, land_mask, dv, dz, jjj, iii, surf_casts_array_obs_TEMP)
        
        #sub_vol_LO_casts[seg_name][int(mon_num)], sub_thick_LO_casts[seg_name][int(mon_num)], sub_casts_array_LO_casts[seg_name][int(mon_num)] = vfun.getLOCastsSubVolThick(Ldir, info_df_use_LO_casts_TEMP, var, threshold_val, z_rho_grid_LO_casts_TEMP, land_mask, dv, dz, jjj, iii, surf_casts_array_LO_casts_TEMP)

        sub_vol_LO_casts[seg_name][int(mon_num)], sub_thick_LO_casts[seg_name][int(mon_num)], sub_casts_array_LO_casts[seg_name][int(mon_num)] = vfun.getLOCastsSubVolThick(Ldir, info_df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])
        
        
        
       # info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
        
        # df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
        
        
        # dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
        
        # fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
        
        # G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)
        
        
       # sub_vol_LO_his[seg_name][int(mon_num)], sub_thick_LO_his[seg_name][int(mon_num)] = vfun.getLOHisSubVolThick(dv, dz, fn_his, jjj, iii, var, threshold_val)
        
        
        #surf_casts_array[seg_name][int(mon_num)] = vfun.assignSurfaceToCasts(info_df_use, jjj, iii)
        
        
       # vfun.extractLOCasts(Ldir, info_df_use, fn_his)        
        
        
        sub_vol_obs[seg_name][int(mon_num)], sub_thick_obs[seg_name][int(mon_num)], sub_casts_array_obs[seg_name][int(mon_num)] = vfun.getOBSCastsSubVolThick(info_df_use, df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])
        
        print(seg_name + ' ' + mon_str)

# %%

fig_dir = '/Users/dakotamascarenas/Desktop/pltz/debug/'

if Ldir['testing']:
    
    for seg_name in seg_list:
    
        for (mon_num, mon_str) in zip(month_num, month_str):
            
            info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
                    
            df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
            
            cmap = cm.get_cmap('viridis', lut= len(info_df_use.index))
            
            m = 0
            
            for cid in info_df_use.index:
                
                
                depth = -h[int(info_df_use.loc[cid, 'jj_cast']), int(info_df_use.loc[cid, 'ii_cast'])]
                
                LO_casts_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' / (str(Ldir['year'])) / (str(info_df_use['segment'].iloc[0]) + '_' + str(info_df_use['time'].dt.date.min()) + '_' + str(info_df_use['time'].dt.date.max()) ) )
                
                df_temp = df_use[df_use['cid'] == cid]
                
                fn = (LO_casts_dir) / (str(cid) + '.nc')
                
                if fn.exists(): 
                
                    z_rho, var_out = vfun.getLOCastsAttrs(fn) #need to generalize
        
                fig, ax = plt.subplots()
                
                ax.plot(df_temp['DO_mg_L'], df_temp['z'], 'o', markersize = 15, color=cmap(m))
                
                ax.plot(var_out, z_rho, '+', color=cmap(m), markersize = 15)
                
                plt.axhline(y = depth, color = 'orange', linestyle = '-')
                
                plt.axvline(x = threshold_val, color = 'grey', linestyle = '-')
                
                ax.set(xlabel='DO [mg/L]', ylabel='Depth [m]', title= str(cid) + ' ' + mon_str)
                ax.grid()
        
                fig.savefig(fig_dir + str(cid) + '.png')
                
                m+=1


# %%

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    iii = iii_dict[seg_name]
    
    
    min_lat = Lat[min(jjj) - 10]
    max_lat = Lat[max(jjj) + 10]
    
    min_lon = Lon[min(iii) - 10]
    max_lon = Lon[max(iii) + 10]    
    
    
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
        fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
        
        G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)
        
        sub_thick_LO_his_plot = np.empty(np.shape(land_mask))
        
        sub_thick_LO_his_plot.fill(np.nan)
        
        sub_thick_LO_his_plot[jjj,iii] = sub_thick_LO_his[seg_name][int(mon_num)]
        
        sub_thick_LO_casts_plot = np.empty(np.shape(land_mask))
        
        sub_thick_LO_casts_plot.fill(np.nan)
        
        sub_thick_LO_casts_plot[jjj,iii] = sub_thick_LO_casts[seg_name][int(mon_num)]
        
        sub_thick_obs_plot = np.empty(np.shape(land_mask))
        
        sub_thick_obs_plot.fill(np.nan)
        
        sub_thick_obs_plot[jjj,iii] = sub_thick_obs[seg_name][int(mon_num)]
        
        cmap0 = cm.get_cmap('viridis', lut= len(info_df_use.index))
        
        
        info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
                
        df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
        
        
        pfun.start_plot(fs=14, figsize=(16,27))
        fig0, axes0 = plt.subplots(nrows=3, ncols=1, squeeze=False)
        
        c0 = axes0[0,0].pcolormesh(plon,plat, sub_thick_LO_his_plot, cmap='Blues', alpha = 0.8, vmin = 0, vmax = 300)
        
        axes0[0,0].set_xlim([min_lon,max_lon])
        axes0[0,0].set_ylim([min_lat,max_lat])
        axes0[0,0].tick_params(labelrotation=45)
        axes0[0,0].set_title('LO His ' + mon_str + ' ' + str(Ldir['year']) + ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[0,0])
        pfun.dar(axes0[0,0])
        
        
        # c1 = axes0[1,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_obs[seg_name][int(mon_num)], cmap='Greens', alpha = 0.8, vmin = 0, vmax = 300)
        
        # for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
        #     axes0[1,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        # axes0[1,0].set_xlim([min_lon,max_lon])
        # axes0[1,0].set_ylim([min_lat,max_lat])
        # axes0[1,0].tick_params(labelrotation=45)
        # axes0[1,0].set_title('Obs VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        # pfun.add_coast(axes0[1,0])
        
        
        c1 = axes0[1,0].pcolormesh(plon,plat, sub_thick_LO_casts_plot, cmap='Blues', alpha = 0.8, vmin = 0, vmax = 300)
        
        n= 0
        for cid in info_df_use.index:
                    
            axes0[1,0].plot(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'],'o', c = cmap0(n), markeredgecolor='black', markersize=10)
            axes0[1,0].text(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'], str(cid)) 
            
            n+=1
        
        axes0[1,0].set_xlim([min_lon,max_lon])
        axes0[1,0].set_ylim([min_lat,max_lat])
        axes0[1,0].tick_params(labelrotation=45)
        axes0[1,0].set_title('LO VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[1,0])
        pfun.dar(axes0[1,0])

        
        
        c2 = axes0[2,0].pcolormesh(plon,plat, sub_thick_obs_plot, cmap='Blues', alpha = 0.8, vmin = 0, vmax = 300)
        
        n= 0
        for cid in info_df_use.index:
                    
            axes0[2,0].plot(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'],'o', c = cmap0(n), markeredgecolor='black', markersize=10)
            axes0[2,0].text(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'], str(cid)) 
            
            n+=1

        
        axes0[2,0].set_xlim([min_lon,max_lon])
        axes0[2,0].set_ylim([min_lat,max_lat])
        axes0[2,0].tick_params(labelrotation=45)
        axes0[2,0].set_title('Obs VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[2,0])
        pfun.dar(axes0[2,0])

        
        fig0.colorbar(c0,ax=axes0[0,0], label = 'Subthreshold Thickness [m]')
        
        fig0.colorbar(c1,ax=axes0[1,0], label = 'Subthreshold Thickness [m]')
        
        fig0.colorbar(c2,ax=axes0[2,0], label = 'Subthreshold Thickness [m]')
        
        fig0.tight_layout()
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+seg_name + '_sub_thick_'+str(threshold_val)+'_mg_L_DO_casts_' + str(Ldir['year']) + '_00' + mon_num+'.png', bbox_inches='tight')


# %%

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