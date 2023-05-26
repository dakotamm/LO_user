"""
Testing VFC functions version 3.

Test on mac in ipython:
run vfc_3_test -gtx cas6_v0_live -year 2017 -test True

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

import VFC_functions_2 as vfun2

import VFC_functions_3 as vfun3

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

if Ldir['testing']:

    month_num = ['09'] # ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Sep'] #,['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 5 #mg/L DO

var = 'DO_mg_L'

segments = 'basins' #custom (specify string list and string build list), basins, whole domain, sound and strait

# seg_build_list = optional
    
G, S, T, land_mask, Lon, Lat, z_rho_grid, dz, dv = vfun3.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun3.getSegmentInfo(Ldir)

info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))


# %%


jjj_dict, iii_dict, seg_list = vfun3.defineSegmentIndices(segments, j_dict, i_dict)

if Ldir['testing']:

    seg_list = ['Whidbey Basin']

# %%

info_df = vfun3.getCleanInfoDF(info_fn, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict)

df = vfun3.getCleanDF(fn, info_df)

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


sub_thick_LO_his2 = {}

sub_thick_LO_casts2 = {}

sub_thick_obs2 = {}

sub_vol_LO_his2 = {}

sub_vol_LO_casts2 = {}

sub_vol_obs2 = {}

sub_thick_obs2 = {}

surf_casts_array2 = {}

sub_casts_array_obs2 = {}

sub_casts_array_LO_casts2 = {}


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
    
    
    sub_thick_LO_his2[seg_name] = {}

    sub_thick_LO_casts2[seg_name] = {}

    sub_thick_obs2[seg_name] = {}

    sub_vol_LO_his2[seg_name] = {}

    sub_vol_LO_casts2[seg_name] = {}
    
    sub_vol_obs2[seg_name] = {}
    
    surf_casts_array2[seg_name] = {}
    
    sub_casts_array_obs2[seg_name] = {}
    
    sub_casts_array_LO_casts2[seg_name] = {}

    
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
        fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
        
        G, S, T, land_mask, Lon, Lat, z_rho_grid, dz, dv = vfun3.getGridInfo(fn_his)
        
        sub_vol_LO_his[seg_name][int(mon_num)], sub_thick_LO_his[seg_name][int(mon_num)] = vfun3.getLOHisSubVolThick(dv, dz, fn_his, jjj, iii, var, threshold_val)
        
        sub_vol_LO_his2[seg_name][int(mon_num)], sub_thick_LO_his2[seg_name][int(mon_num)] = vfun2.getLOHisSubVolThick(dv, dz, fn_his, jjj, iii, var, threshold_val)

        
        info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
                
        df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
        
        
        sub_vol_obs2[seg_name][int(mon_num)], sub_thick_obs2[seg_name][int(mon_num)], surf_casts_array2[seg_name][int(mon_num)], sub_casts_array_obs2[seg_name][int(mon_num)] = vfun2.getOBSCastsSubVolThick(info_df_use, df_use, var, threshold_val, z_rho_grid, dv, dz, land_mask, jjj, iii)
                        
        vfun2.extractLOCasts(Ldir, info_df_use, fn_his)
        
        sub_vol_LO_casts2[seg_name][int(mon_num)], sub_thick_LO_casts2[seg_name][int(mon_num)], sub_casts_array_LO_casts2[seg_name][int(mon_num)] = vfun2.getLOCastsSubVolThick(Ldir, info_df_use, var, threshold_val, z_rho_grid, dv, dz, land_mask, jjj, iii, surf_casts_array2[seg_name][int(mon_num)])

        
        surf_casts_array[seg_name][int(mon_num)] = vfun3.assignSurfaceToCasts(info_df_use, jjj, iii)
        
        vfun3.extractLOCasts(Ldir, info_df_use, fn_his)
        
        sub_vol_obs[seg_name][int(mon_num)], sub_thick_obs[seg_name][int(mon_num)], sub_casts_array_obs[seg_name][int(mon_num)] = vfun3.getOBSCastsSubVolThick(info_df_use, df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])

        sub_vol_LO_casts[seg_name][int(mon_num)], sub_thick_LO_casts[seg_name][int(mon_num)], sub_casts_array_LO_casts[seg_name][int(mon_num)] = vfun3.getLOCastsSubVolThick(Ldir, info_df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])

        
        jj_casts[seg_name][int(mon_num)] = info_df_use['jj_cast'].to_numpy()
        
        ii_casts[seg_name][int(mon_num)] = info_df_use['ii_cast'].to_numpy()
        
        cid_dict[seg_name][int(mon_num)] =info_df_use.index.to_numpy()
        
        print(seg_name + mon_str)
        
# %%

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    iii = iii_dict[seg_name]
    
    
    min_lat = Lat[min(jjj) - 10]
    max_lat = Lat[max(jjj) + 10]
    
    min_lon = Lon[min(iii) - 10]
    max_lon = Lon[max(iii) + 10]    
    
    
    for (mon_num, mon_str) in zip(month_num,month_str):
         
        pfun.start_plot(fs=14, figsize=(16,27))
        fig0, axes0 = plt.subplots(nrows=3, ncols=1, squeeze=False)
        
        c0 = axes0[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_LO_his2[seg_name][int(mon_num)], cmap='Blues', alpha = 0.8, vmin = 0, vmax = 300)
        
        axes0[0,0].set_xlim([min_lon,max_lon])
        axes0[0,0].set_ylim([min_lat,max_lat])
        axes0[0,0].tick_params(labelrotation=45)
        axes0[0,0].set_title('LO ' + mon_str + ' ' + str(Ldir['year']) + ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[0,0])
        
        
        # c1 = axes0[1,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_obs[seg_name][int(mon_num)], cmap='Greens', alpha = 0.8, vmin = 0, vmax = 300)
        
        # for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
        #     axes0[1,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        # axes0[1,0].set_xlim([min_lon,max_lon])
        # axes0[1,0].set_ylim([min_lat,max_lat])
        # axes0[1,0].tick_params(labelrotation=45)
        # axes0[1,0].set_title('Obs VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        # pfun.add_coast(axes0[1,0])
        
        
        c1 = axes0[1,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_LO_casts2[seg_name][int(mon_num)], cmap='Purples', alpha = 0.8, vmin = 0, vmax = 300)
        
        for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
            axes0[1,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        axes0[1,0].set_xlim([min_lon,max_lon])
        axes0[1,0].set_ylim([min_lat,max_lat])
        axes0[1,0].tick_params(labelrotation=45)
        axes0[1,0].set_title('LO VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[1,0])
        
        
        c2 = axes0[2,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_obs2[seg_name][int(mon_num)], cmap='Greens', alpha = 0.8, vmin = 0, vmax = 300)
        
        for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
            axes0[2,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        axes0[2,0].set_xlim([min_lon,max_lon])
        axes0[2,0].set_ylim([min_lat,max_lat])
        axes0[2,0].tick_params(labelrotation=45)
        axes0[2,0].set_title('Obs VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[2,0])
        
        fig0.colorbar(c0,ax=axes0[0,0], label = 'Subthreshold Thickness [m]')
        
        fig0.colorbar(c1,ax=axes0[1,0], label = 'Subthreshold Thickness [m]')
        
        fig0.colorbar(c2,ax=axes0[2,0], label = 'Subthreshold Thickness [m]')
        
        fig0.tight_layout()
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/V2_'+seg_name + '_sub_thick_'+str(threshold_val)+'_mg_L_DO_casts_' + str(Ldir['year']) + '_00' + mon_num+'.png')
        
        
        
        
        sub_thick_LO_his_plot = np.empty(np.shape(land_mask))
        
        sub_thick_LO_his_plot.fill(np.nan)
        
        sub_thick_LO_his_plot[jjj,iii] = sub_thick_LO_his[seg_name][int(mon_num)]
        
        sub_thick_LO_casts_plot = np.empty(np.shape(land_mask))
        
        sub_thick_LO_casts_plot.fill(np.nan)
        
        sub_thick_LO_casts_plot[jjj,iii] = sub_thick_LO_casts[seg_name][int(mon_num)]
        
        sub_thick_obs_plot = np.empty(np.shape(land_mask))
        
        sub_thick_obs_plot.fill(np.nan)
        
        sub_thick_obs_plot[jjj,iii] = sub_thick_obs[seg_name][int(mon_num)]
        
        
        plon,plat = pfun.get_plon_plat(G['lon_rho'],G['lat_rho'])
        
        
        pfun.start_plot(fs=14, figsize=(16,27))
        fig1, axes0 = plt.subplots(nrows=3, ncols=1, squeeze=False)
        
        c0 = axes0[0,0].pcolormesh(plon,plat, sub_thick_LO_his_plot, cmap='Blues', alpha = 0.8, vmin = 0, vmax = 300)
        
        axes0[0,0].set_xlim([min_lon,max_lon])
        axes0[0,0].set_ylim([min_lat,max_lat])
        axes0[0,0].tick_params(labelrotation=45)
        axes0[0,0].set_title('LO ' + mon_str + ' ' + str(Ldir['year']) + ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[0,0])
        
        
        # c1 = axes0[1,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_obs[seg_name][int(mon_num)], cmap='Greens', alpha = 0.8, vmin = 0, vmax = 300)
        
        # for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
        #     axes0[1,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        # axes0[1,0].set_xlim([min_lon,max_lon])
        # axes0[1,0].set_ylim([min_lat,max_lat])
        # axes0[1,0].tick_params(labelrotation=45)
        # axes0[1,0].set_title('Obs VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        # pfun.add_coast(axes0[1,0])
        
        
        c1 = axes0[1,0].pcolormesh(plon,plat, sub_thick_LO_casts_plot, cmap='Purples', alpha = 0.8, vmin = 0, vmax = 300)
        
        for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
            axes0[1,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        axes0[1,0].set_xlim([min_lon,max_lon])
        axes0[1,0].set_ylim([min_lat,max_lat])
        axes0[1,0].tick_params(labelrotation=45)
        axes0[1,0].set_title('LO VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[1,0])
        
        
        c2 = axes0[2,0].pcolormesh(plon,plat, sub_thick_obs_plot, cmap='Greens', alpha = 0.8, vmin = 0, vmax = 300)
        
        for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
            axes0[2,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        axes0[2,0].set_xlim([min_lon,max_lon])
        axes0[2,0].set_ylim([min_lat,max_lat])
        axes0[2,0].tick_params(labelrotation=45)
        axes0[2,0].set_title('Obs VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[2,0])
        
        fig1.colorbar(c0,ax=axes0[0,0], label = 'Subthreshold Thickness [m]')
        
        fig1.colorbar(c1,ax=axes0[1,0], label = 'Subthreshold Thickness [m]')
        
        fig1.colorbar(c2,ax=axes0[2,0], label = 'Subthreshold Thickness [m]')
        
        fig1.tight_layout()
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/V3_'+seg_name + '_sub_thick_'+str(threshold_val)+'_mg_L_DO_casts_' + str(Ldir['year']) + '_00' + mon_num+'.png')
        
        