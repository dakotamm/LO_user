"""
Blah

Test on mac in ipython:
run test_depth_check -gtx cas6_v0_live -year 2017 -test True

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

threshold_val = 5 #mg/L DO

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

    seg_list = ['Strait of Georgia']
    
    
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

    ii_casts[seg_name] = {}

    jj_casts[seg_name] = {}
    
    cid_dict[seg_name] = {}
    
    surf_casts_array[seg_name] = {}
    
    sub_casts_array_obs[seg_name] = {}
    
    sub_casts_array_LO_casts[seg_name] = {}
    

    
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
        fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
        
        G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)
        
        sub_vol_LO_his[seg_name][int(mon_num)], sub_thick_LO_his[seg_name][int(mon_num)] = vfun.getLOHisSubVolThick(dv, dz, fn_his, jjj, iii, var, threshold_val)
        
        
        info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
                
        df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
        
        
        surf_casts_array[seg_name][int(mon_num)] = vfun.assignSurfaceToCasts(info_df_use, jjj, iii)
        
        vfun.extractLOCasts(Ldir, info_df_use, fn_his)
        
        sub_vol_obs[seg_name][int(mon_num)], sub_thick_obs[seg_name][int(mon_num)], sub_casts_array_obs[seg_name][int(mon_num)] = vfun.getOBSCastsSubVolThick(info_df_use, df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])

        sub_vol_LO_casts[seg_name][int(mon_num)], sub_thick_LO_casts[seg_name][int(mon_num)], sub_casts_array_LO_casts[seg_name][int(mon_num)] = vfun.getLOCastsSubVolThick(Ldir, info_df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])

        
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
        
        
        c1 = axes0[1,0].pcolormesh(plon,plat, sub_thick_LO_casts_plot, cmap='Purples', alpha = 0.8, vmin = 0, vmax = 300)
        
        for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
            axes0[1,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        axes0[1,0].set_xlim([min_lon,max_lon])
        axes0[1,0].set_ylim([min_lat,max_lat])
        axes0[1,0].tick_params(labelrotation=45)
        axes0[1,0].set_title('LO VFC ' + mon_str + ' ' + str(Ldir['year'])+ ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[1,0])
        pfun.dar(axes0[1,0])

        
        
        c2 = axes0[2,0].pcolormesh(plon,plat, sub_thick_obs_plot, cmap='Greens', alpha = 0.8, vmin = 0, vmax = 300)
        
        for n in range(len(ii_casts[seg_name][int(mon_num)])):
        
            axes0[2,0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][n])],Lat[int(jj_casts[seg_name][int(mon_num)][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
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

# %%

vol_df = pd.DataFrame()


vol_df['segment'] = []

vol_df['month'] = []

vol_df['data_type'] = []

vol_df['vol_km3'] = []



for seg_name in seg_list:
    
    
    for (mon_num, mon_str) in zip(month_num,month_str):
        
        df_temp = pd.DataFrame()
                
        df_temp['segment'] = [seg_name]
        
        df_temp['month'] = [int(mon_num)]
        
        df_temp['data_type'] = ['LO His']
        
        df_temp['vol_km3'] = [sub_vol_LO_his[seg_name][int(mon_num)]*1e-9] #convert to km^3
        
        vol_df = pd.concat([vol_df, df_temp], ignore_index=True)
        
        
        df_temp['data_type'] = ['LO Casts']
        
        df_temp['vol_km3'] = [sub_vol_LO_casts[seg_name][int(mon_num)]*1e-9]
        
        vol_df = pd.concat([vol_df, df_temp], ignore_index=True)
        
        
        df_temp['data_type'] = ['OBS']
        
        df_temp['vol_km3'] = [sub_vol_obs[seg_name][int(mon_num)]*1e-9]
        
        vol_df = pd.concat([vol_df, df_temp], ignore_index=True)
        
vol_df.to_pickle((file_dir + 'vol_df.p'))

# %%


LO_his_mins = vol_df[vol_df['data_type'] == 'LO His'].groupby(['segment', 'data_type'])['vol_km3'].min()

LO_his_maxs = vol_df[vol_df['data_type'] == 'LO His'].groupby(['segment', 'data_type'])['vol_km3'].max()

LO_his_ranges = LO_his_maxs - LO_his_mins

LO_his_ranges = LO_his_ranges.to_frame().reset_index()


obs_mins = vol_df[vol_df['data_type'] == 'OBS'].groupby(['segment', 'data_type'])['vol_km3'].min()

obs_maxs = vol_df[vol_df['data_type'] == 'OBS'].groupby(['segment', 'data_type'])['vol_km3'].max()

obs_ranges = obs_maxs - obs_mins

obs_ranges = obs_ranges.to_frame().reset_index()


vol_df_wide = vol_df.pivot(index=['month', 'segment'], columns = 'data_type', values='vol_km3').reset_index()

vol_df_wide = pd.merge(vol_df_wide, LO_his_ranges, how='left', on='segment')

vol_df_wide = vol_df_wide.rename(columns = {'vol_km3':'LO_his_ranges'})

vol_df_wide = pd.merge(vol_df_wide, obs_ranges, how='left', on='segment')

vol_df_wide = vol_df_wide.rename(columns = {'vol_km3':'obs_ranges'})

vol_df_wide = vol_df_wide[['month','segment','LO Casts','LO His', 'OBS', 'LO_his_ranges', 'obs_ranges']]


# %%



vol_df_wide['SE_LO_his_LO_casts'] = np.square(vol_df_wide['LO His'] - vol_df_wide['LO Casts'])

vol_df_wide['SE_OBS_LO_his'] = np.square(vol_df_wide['OBS'] - vol_df_wide['LO His'])


temp1 = vol_df_wide.groupby(['segment'])['SE_LO_his_LO_casts'].mean().to_frame().reset_index()

temp2 = vol_df_wide.groupby(['segment'])['SE_OBS_LO_his'].mean().to_frame().reset_index()

temp1 = temp1.rename(columns = {'SE_LO_his_LO_casts':'MSE_LO_his_LO_casts'})

temp2 = temp2.rename(columns = {'SE_OBS_LO_his':'MSE_OBS_LO_his'})


temp1['RMSE_LO_his_LO_casts'] = np.sqrt(temp1['MSE_LO_his_LO_casts'])

temp2['RMSE_OBS_LO_his'] = np.sqrt(temp2['MSE_OBS_LO_his'])

# %%


vol_df_wide = pd.merge(vol_df_wide, temp1, how='left', on='segment')

vol_df_wide = pd.merge(vol_df_wide, temp2, how='left', on='segment')

vol_df_wide['norm_RMSE_LO_his_LO_casts'] = vol_df_wide['RMSE_LO_his_LO_casts'] / vol_df_wide['LO_his_ranges']

vol_df_wide['norm_RMSE_OBS_LO_his'] = vol_df_wide['RMSE_OBS_LO_his'] / vol_df_wide['obs_ranges']


vol_df_wide.to_pickle((file_dir + 'vol_df_wide.p'))    

# %%

# file_dir = '/Users/dakotamascarenas/Desktop/'

# vol_df = pd.read_pickle((file_dir + 'vol_df.p'))

# vol_df_wide = pd.read_pickle((file_dir + 'vol_df_wide.p'))

vol_df = vol_df.rename(columns = {'month':'Month'})

vol_df_wide = vol_df_wide.rename(columns = {'month':'Month'})

vol_df = vol_df.rename(columns = {'data_type':'Data Type'})

vol_df_wide = vol_df_wide.rename(columns = {'data_type':'Data Type'})


vol_df_wide['LO_his_less_LO_casts'] = vol_df_wide['LO His'] - vol_df_wide['LO Casts']

vol_df_wide['OBS_less_LO_his'] = vol_df_wide['OBS'] - vol_df_wide['LO His']

vol_df_wide['% Dif. (LO His - LO Casts)'] = vol_df_wide['LO_his_less_LO_casts']/vol_df_wide['LO His']*100

vol_df_wide['% Dif. (OBS - LO His)'] = vol_df_wide['OBS_less_LO_his']/vol_df_wide['OBS']*100


pct_dif_melt = pd.melt(vol_df_wide, id_vars=['segment', 'Month'], value_vars=['% Dif. (LO His - LO Casts)', '% Dif. (OBS - LO His)'], var_name = 'Error Type', value_name= '% Difference')

pct_dif_melt.replace([np.inf, -np.inf], np.nan, inplace=True)

# with open((file_dir + 'jj_casts.pkl'), 'rb') as f: 
#     jj_casts = pickle.load(f)

# with open((file_dir + 'ii_casts.pkl'), 'rb') as f: 
#     ii_casts = pickle.load(f)  


# %%

norm_RMSE_array_LO_his_LO_casts = np.empty(np.shape(land_mask))
norm_RMSE_array_LO_his_LO_casts.fill(np.nan)

norm_RMSE_array_OBS_LO_his = np.empty(np.shape(land_mask))
norm_RMSE_array_OBS_LO_his.fill(np.nan)
        

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    
    fig, ax = plt.subplots(1,3,figsize=(16,4), gridspec_kw={'width_ratios': [2, 1, 1]})
    
    sns.lineplot(data=vol_df[vol_df['segment'] == seg_name], x = 'Month', y = 'vol_km3', ax = ax[0], hue = 'Data Type', palette = ['xkcd:azure', 'xkcd:light purple','xkcd:kelly green'], size = 'Data Type', size_order=['OBS', 'LO His', 'LO Casts'], sizes=(1, 3))
    
    ax[1].axline((0, 0), slope=1, color='gray', alpha=0.5)

    ax[2].axline((0, 0), slope=1, color='gray', alpha=0.5)
    
    sns.scatterplot(data=vol_df_wide[vol_df_wide['segment'] == seg_name], x = 'LO His', y = 'LO Casts', hue = 'Month', ax = ax[1], palette='twilight')
    
    sns.scatterplot(data=vol_df_wide[vol_df_wide['segment'] == seg_name], x = 'OBS', y = 'LO His', hue = 'Month', ax = ax[2], palette='twilight')
    
    ax[0].set_title(seg_name + ' Sub-' + str(threshold_val)+ ' mg/L DO Volumes, Monthly ' + str(Ldir['year']))
    
    ax[1].set_title('Sampling Bias Norm-RMSE: %.3f ' % vol_df_wide[vol_df_wide['segment'] ==seg_name]['norm_RMSE_LO_his_LO_casts'].iloc[0])
    
    ax[2].set_title('Model Error Norm-RMSE: %.3f ' % vol_df_wide[vol_df_wide['segment'] ==seg_name]['norm_RMSE_OBS_LO_his'].iloc[0])

    
    fig.tight_layout()
    
    #plt.show()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' +seg_name + '_sub_vol_'+str(threshold_val)+'_mg_L_DO_' + str(Ldir['year'])+'.png')

    
    norm_RMSE_array_LO_his_LO_casts[jjj,iii] = vol_df_wide[vol_df_wide['segment'] ==seg_name]['norm_RMSE_LO_his_LO_casts'].iloc[0]
    
    norm_RMSE_array_OBS_LO_his[jjj,iii] = vol_df_wide[vol_df_wide['segment'] ==seg_name]['norm_RMSE_OBS_LO_his'].iloc[0]
    
    
    
    # fig, ax = plt.subplots(1,1,figsize=(8, 4))
    
    # sns.lineplot(data = pct_dif_melt[pct_dif_melt['segment'] == seg_name], x= 'Month', y = '% Difference', hue = 'Error Type', palette = ['blue', 'green'], alpha = 0.8)
    
    # ax.set_ylim([-200, 200])
    # ax.set_xlim([1, 12])
    # #plt.show()
    
    # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + seg_name +'_sub_vol_pct_diff_'+str(threshold_val)+'_mg_L_DO_' + str(Ldir['year'])+'.png')
    

xlims = [-125.5, -122]

ylims = [47, 50.5]

fig, ax = plt.subplots(1,2,figsize=(16, 8))

cf0 = ax[0].pcolormesh(plon, plat, norm_RMSE_array_LO_his_LO_casts, cmap = 'BuPu') #, vmin = 0.06, vmax = 0.23)  

cf1 = ax[1].pcolormesh(plon, plat, norm_RMSE_array_OBS_LO_his, cmap = 'GnBu')

pfun.add_coast(ax[0])

pfun.add_coast(ax[1])

pfun.dar(ax[0])
pfun.dar(ax[1])

ax[0].set_xlim(xlims)
ax[1].set_xlim(xlims)
ax[0].set_ylim(ylims)
ax[1].set_ylim(ylims)

ax[0].set_title('Sampling Bias (LO His vs. LO Casts) Sub-' + str(threshold_val)+ ' mg/L DO Volume Norm_RMSE ' + str(Ldir['year']))

ax[1].set_title('Model Error (OBS vs. LO His) Sub-' + str(threshold_val)+ ' mg/L DO Volume Norm_RMSE ' + str(Ldir['year']))


fig.colorbar(cf0, ax=ax[0], label = 'Norm-RMSE')

fig.colorbar(cf1, ax=ax[1], label = 'Norm-RMSE')


fig.tight_layout()

#plt.show()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/sub_vol_'+str(threshold_val)+'_mg_L_DO_errors_' + str(Ldir['year'])+'.png')


# %%

xlims = [-125.5, -122]

ylims = [47, 50.5]

for (mon_num, mon_str) in zip(month_num, month_str):
    
    
    pct_dif_LO_his_LO_casts_array = np.empty(np.shape(land_mask))
    pct_dif_LO_his_LO_casts_array.fill(np.nan)

    pct_dif_OBS_LO_his_array = np.empty(np.shape(land_mask))
    pct_dif_OBS_LO_his_array.fill(np.nan)
    
    
    for seg_name in seg_list:
        
        jjj = jjj_dict[seg_name]
        
        iii = iii_dict[seg_name]
        
        pct_dif_LO_his_LO_casts_array[jjj,iii] = vol_df_wide[(vol_df_wide['segment'] ==seg_name) & (vol_df_wide['Month'] == int(mon_num))]['% Dif. (LO His - LO Casts)'].iloc[0]
        
        pct_dif_OBS_LO_his_array[jjj,iii] = vol_df_wide[(vol_df_wide['segment'] ==seg_name) & (vol_df_wide['Month'] == int(mon_num))]['% Dif. (OBS - LO His)'].iloc[0]
        

        
    fig, ax = plt.subplots(1,2,figsize=(16, 8))
    
    cf0 = ax[0].pcolormesh(plon, plat, pct_dif_LO_his_LO_casts_array, cmap = 'RdYlBu', alpha = 0.8, vmin = -200, vmax = 200)  
    
    cf1 = ax[1].pcolormesh(plon, plat, pct_dif_OBS_LO_his_array, cmap = 'RdYlGn', alpha = 0.8, vmin = -200, vmax = 200)
    
    for seg_name in seg_list:
    
        if ii_casts[seg_name][int(mon_num)].size > 0:
        
            for m in range(len(ii_casts[seg_name][int(mon_num)])):
        
                ax[0].plot(Lon[int(ii_casts[seg_name][int(mon_num)][m])], Lat[int(jj_casts[seg_name][int(mon_num)][m])],'o',markeredgecolor='black', markerfacecolor="lightgrey",markersize=5)
                
                ax[1].plot(Lon[int(ii_casts[seg_name][int(mon_num)][m])], Lat[int(jj_casts[seg_name][int(mon_num)][m])],'o',markeredgecolor='black', markerfacecolor="lightgrey",markersize=5)

    
    pfun.add_coast(ax[0])
    
    pfun.add_coast(ax[1])
    
    pfun.dar(ax[0])
    pfun.dar(ax[1])
    
    ax[0].set_xlim(xlims)
    ax[1].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[1].set_ylim(ylims)
    
    ax[0].set_title('% Diff.(LO His vs. LO Casts) Sub-' + str(threshold_val)+ ' mg/L DO Volume ' + mon_str + ' ' + str(Ldir['year']))
    
    ax[1].set_title('% Diff. (OBS vs. LO His) Sub-' + str(threshold_val)+ ' mg/L DO Volume ' + mon_str + ' ' + str(Ldir['year']))
    
    
    fig.colorbar(cf0, ax=ax[0], label = '% Diff')
    
    fig.colorbar(cf1, ax=ax[1], label = '% Diff')
    
    
    fig.tight_layout()
    
    #plt.show()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/sub_vol_pct_diff_'+str(threshold_val)+'_mg_L_DO_errors_' + str(Ldir['year'])+'_00' + mon_num+ '.png')


# %%

# with open((file_dir + 'sub_thick_LO_casts.pkl'), 'rb') as f: 
#     sub_thick_LO_casts = pickle.load(f)

# with open((file_dir + 'sub_thick_LO_his.pkl'), 'rb') as f: 
#     sub_thick_LO_his = pickle.load(f) 
    
# with open((file_dir + 'sub_thick_obs.pkl'), 'rb') as f: 
#     sub_thick_obs = pickle.load(f)  
    
# with open((file_dir + 'sub_thick_LO_casts.pkl'), 'wb') as f: 
#     pickle.dump(sub_thick_LO_casts, f)

# with open((file_dir + 'sub_thick_LO_his.pkl'), 'wb') as f: 
#     pickle.dump(sub_thick_LO_his, f)  
    
# with open((file_dir + 'sub_thick_obs.pkl'), 'wb') as f: 
#     pickle.dump(sub_thick_obs, f)

        
# %%

vol_df['total_vol_km3'] = np.nan

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    vol_array = np.empty(np.shape(dv))
    
    vol_array[:,jjj,iii] = dv[:,jjj,iii]
    
    vol_df.loc[vol_df['segment'] == seg_name, ['total_vol_km3']] = np.sum(vol_array)*1e-9
    
vol_df['Subthreshold Volume [km^3]/Total Volume [km^3]'] = vol_df['vol_km3']/vol_df['total_vol_km3']

# %%


fig, ax = plt.subplots(1,1,figsize=(12,8))

sns.lineplot(data = vol_df[vol_df['Data Type'] == 'OBS'], x = 'Month', y = 'Subthreshold Volume [km^3]/Total Volume [km^3]', hue = 'segment', palette = 'rocket_r', hue_order = ['Tacoma Narrows', 'South Sound', 'Admiralty Inlet', 'Hood Canal', 'Whidbey Basin', 'Main Basin', 'Strait of Juan de Fuca', 'Strait of Georgia'])#, size='segment', size_order = ['Tacoma Narrows', 'South Sound', 'Admiralty Inlet', 'Hood Canal', 'Whidbey Basin', 'Main Basin', 'Strait of Juan de Fuca', 'Strait of Georgia'], sizes=(3, 1))

ax.set_title('2017 Sub-5.0 mg/L [DO] Normalized Volumes - OBS')

plt.legend(title = 'Basin [Order of Increasing Volume]')

plt.grid()

fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/2017_norm_sub_vol.png')



    