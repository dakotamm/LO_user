"""
Finding hypoxic depth and volume using observational data.

Test on mac in ipython:
run obs_hyp_vol_DM -gtx cas6_v0_live -source nceiSalish -otype bottle -year 2017 -test False

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

import os


Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

month_num = ['01','02','03','04','05','06','07','08','09','10','11','12']

month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

segments = ['H1','H2','H3','H4','H5','H6','H7','H8']

#segments = ['M1','M2','M3','M4', 'M5','M6']

# this is really not a good way to code this (below)

cast_start = [datetime(int(year_str),1,1), datetime(int(year_str),2,1), datetime(int(year_str),3,1), datetime(int(year_str),4,1), datetime(int(year_str),5,1),
              datetime(int(year_str),6,1), datetime(int(year_str),7,1), datetime(int(year_str),8,1), datetime(int(year_str),9,1), datetime(int(year_str),10,1),
              datetime(int(year_str),11,1), datetime(int(year_str),12,1)]

cast_end = [datetime(int(year_str),1,31), datetime(int(year_str),2,28), datetime(int(year_str),3,31), datetime(int(year_str),4,30), datetime(int(year_str),5,31),
              datetime(int(year_str),6,30), datetime(int(year_str),7,31), datetime(int(year_str),8,31), datetime(int(year_str),9,30), datetime(int(year_str),10,31),
              datetime(int(year_str),11,30), datetime(int(year_str),12,31)]

sub_vol_dict_obs = {}

var_array_dict = {}

sub_thick_dict_obs = {}

surf_casts_array_dict = {}

surf_casts_array_plot_dict = {}

sub_casts_array_plot_dict = {}

jj_cast_dict_obs = {}

ii_cast_dict_obs = {}

# get segment info
vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
v_df = pd.read_pickle(vol_dir / 'volumes.p')
j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
seg_list = list(v_df.index)

info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')

# tide-averaged??? ************* use same grid for VFC?

dt = pd.Timestamp('2022-01-01 01:30:00')
fn_his = cfun.get_his_fn_from_dt(Ldir, dt)

#grid info
G, S, T = zrfun.get_basic_info(fn_his)
Lon = G['lon_rho'][0, :]
Lat = G['lat_rho'][:, 0]
z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
dz = np.diff(z_w_grid,axis=0)
dv = dz*G['DX']*G['DY']


# %%

threshold_val = 5

var = 'DO_mg_L'


# %%


info_df = pd.read_pickle(info_fn)



# %%

fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / (year_str + '.p')

df = pd.read_pickle(fn)
        
# %%

info_df['ix'] = 0

info_df['iy'] = 0

for cid in info_df.index:

    info_df.loc[cid,'ix'] = zfun.find_nearest_ind(Lon, info_df.loc[cid,'lon'])

    info_df.loc[cid,'iy'] = zfun.find_nearest_ind(Lat, info_df.loc[cid,'lat'])
    
# %%

info_df['segment'] = 'None'

for seg_name in seg_list:
    
    ij_pair = list(zip(i_dict[seg_name],j_dict[seg_name]))

    for cid in info_df.index:        
          
        pair = (info_df.loc[cid,'ix'].tolist(), info_df.loc[cid,'iy'].tolist())
        
        if pair in ij_pair:
            info_df.loc[cid,'segment'] = seg_name
            
        else:
            continue
        
# %%

info_df['ii_cast'] = np.nan

info_df['jj_cast'] = np.nan

for cid in info_df.index:
    
    if G['mask_rho'][info_df.loc[cid,'iy'], info_df.loc[cid,'ix']] == 1:
        
        info_df.loc[cid, 'ii_cast'] = info_df.loc[cid, 'ix']
        
        info_df.loc[cid, 'jj_cast'] = info_df.loc[cid, 'iy']
        
    else:
        continue

    
# %%

df = pd.merge(df, info_df[['ix','iy','ii_cast','jj_cast','segment']], how='left', on=['cid'])

df['DO_mg_L'] = df['DO (uM)']*32/1000
            
    
# %%

for segment in segments:
    
    jjj = j_dict[segment]
    iii = i_dict[segment]
    
    df_use = df[df['segment'] == segment]
    
    info_df_use = info_df[info_df['segment'] == segment]
        
    df_use = df_use[~np.isnan(df_use['jj_cast'])]
    
    df_use = df_use[~np.isnan(df_use['ii_cast'])]
        
    info_df_use = info_df_use[~np.isnan(info_df_use['jj_cast'])]
    
    info_df_use = info_df_use[~np.isnan(info_df_use['ii_cast'])]
    
    var_array_dict[segment] = {}
        
    sub_thick_dict_obs[segment] = {}
    
    surf_casts_array_dict[segment] = {}
    
    surf_casts_array_plot_dict[segment] = {}
            
    sub_vol_dict_obs[segment] = {}
    
    jj_cast_dict_obs[segment] = {}
    
    ii_cast_dict_obs[segment] = {}
    
    sub_casts_array_plot_dict[segment] = {}
    
    
    for mon_num in month_num:
        
        m = int(mon_num) -1
        
        df_use_month = df_use[df_use['time'] <= cast_end[m]]
        
        df_use_month = df_use_month[df_use_month['time'] >= cast_start[m]]
        
        #df_use_month = df_use_month.drop_duplicates(subset=['ix','iy'], keep ='first') # DUPLICATE HANDLING HOW?!?!?!?!

        info_df_use_month = info_df_use[info_df_use['time'] <= cast_end[m]]
        
        info_df_use_month = info_df_use_month[info_df_use_month['time'] >= cast_start[m]]
        
       # info_df_use_month = info_df_use_month.drop_duplicates(subset=['ix','iy'], keep ='first')
        
        
        sub_vol = 0
        
        sub_thick_array = np.empty(np.shape(z_rho_grid))
        sub_thick_array.fill(0)
        
        
        if df_use_month.empty: # if there are no casts in this time period
        
            sub_thick_array.fill(np.nan)
        
            sub_thick = np.sum(sub_thick_array, axis=0)
            
            sub_thick = sub_thick[min(jjj):max(jjj)+1,min(iii):max(iii)+1]
                        
            sub_vol_dict_obs[segment][m] = np.nan
            
            sub_thick_dict_obs[segment][m] = sub_thick
        
            
            
        else: # if there ARE casts in this time period
        
            jj_cast_dict_obs[segment][m] = info_df_use_month['jj_cast'].to_numpy()
            
            ii_cast_dict_obs[segment][m] = info_df_use_month['ii_cast'].to_numpy()


            xx = np.arange(min(iii), max(iii)+1)
            yy = np.arange(min(jjj), max(jjj)+1)
            x, y = np.meshgrid(xx, yy)
                    
            a = np.full([len(yy),len(xx)], -99)
            a[jjj-min(jjj),iii-min(iii)] = -1
            a = np.ma.masked_array(a,a==-99)
        
            b = a.copy()
            b0 = a.copy()
            
            n = 0
            
            for cid in info_df_use_month.index:
                b[int(info_df_use_month.loc[cid, 'jj_cast'])-min(jjj), int(info_df_use_month.loc[cid, 'ii_cast'])-min(iii)] = cid
                b0[int(info_df_use_month.loc[cid, 'jj_cast'])-min(jjj), int(info_df_use_month.loc[cid, 'ii_cast'])-min(iii)] = n
                n +=1 
                
            cast_nums = np.linspace(0,(n-1))
                
            c = b.copy()
            c = np.ma.masked_array(c,c==-1)
            
            c0 = b0.copy()
            c0 = np.ma.masked_array(c0,c0==-1)
                
            xy_water = np.array((x[~a.mask],y[~a.mask])).T
        
            xy_casts = np.array((x[~c.mask],y[~c.mask])).T
            
            xy_casts0 = np.array((x[~c0.mask],y[~c0.mask])).T
        
            tree = KDTree(xy_casts)
            
            tree0 = KDTree(xy_casts0)
            
            tree_query = tree.query(xy_water)[1]
            
            
            surf_casts_array = a.copy()
            
            surf_casts_array[~a.mask] = b[~c.mask][tree_query]
            
            surf_casts_array_dict[segment][m] = surf_casts_array
            
            
            surf_casts_array_plot = a.copy()
            
            surf_casts_array_plot[~a.mask] = b0[~c0.mask][tree_query]
            
            surf_casts_array_plot_dict[segment][m] = surf_casts_array_plot
            
            
            df_sub = df_use_month[df_use_month[var] < threshold_val]
            
        
            if df_sub.empty: # if there are no subthreshold volumes
            
                sub_thick = np.sum(sub_thick_array, axis=0)
                
                sub_thick[G['mask_rho'] == 0] = np.nan # ESSENTIALLY - NEED TO REAPPLY LAND MASK
                
                sub_thick = sub_thick[min(jjj):max(jjj)+1, min(iii):max(iii)+1]
                
                sub_thick = np.ma.masked_where(np.ma.getmask(surf_casts_array), sub_thick)
                
                sub_vol_dict_obs[segment][m] = sub_vol
                
                sub_thick_dict_obs[segment][m] = sub_thick
                
            
            else: # if there ARE subthreshold volumes
            
                info_df_sub = info_df_use_month.copy()
            
                for cid in info_df_use_month.index:
                    
                    if ~(cid in df_sub['cid'].unique()):
                        
                        info_df_sub.drop([cid])
                
                sub_casts_array = surf_casts_array.copy()
                
                sub_casts_array_plot = surf_casts_array_plot.copy()
        
                sub_casts_array = [[ele if ele in df_sub['cid'].unique() else -99 for ele in line] for line in sub_casts_array]
                
                sub_casts_array_plot = [[ele if ele in cast_nums else -99 for ele in line] for line in sub_casts_array]
                
                sub_casts_array = np.array(sub_casts_array)
        
                sub_casts_array =np.ma.masked_array(sub_casts_array,sub_casts_array==-99)
                
                sub_casts_array_plot = np.array(sub_casts_array_plot)
        
                sub_casts_array_plot =np.ma.masked_array(sub_casts_array_plot,sub_casts_array_plot == -99)
                
                sub_casts_array_plot_dict[segment][m] = sub_casts_array_plot
                
                max_z_sub = []
                
                for cid in info_df_sub.index:
                    
                    df_temp = df_sub[df_sub['cid'] == cid]
                    
                    max_z_sub = df_temp['z'].max()
                    
                    idx_sub = np.where(sub_casts_array == cid)
                    
                    sub_array = np.empty(np.shape(z_rho_grid))
                    sub_array.fill(0)
                    
                    for nn in range(len(idx_sub[0])):
                        jjj_sub = idx_sub[0][nn] + min(jjj)
                        iii_sub = idx_sub[1][nn] + min(iii)
                        zzz_sub= np.where(z_rho_grid[:,jjj_sub,iii_sub] <= max_z_sub)
                        if zzz_sub:
                            sub_array[zzz_sub,jjj_sub,iii_sub] = dv[zzz_sub,jjj_sub,iii_sub]
                            sub_thick_array[zzz_sub,jjj_sub,iii_sub] = dz[zzz_sub,jjj_sub,iii_sub]
                            
                sub_vol = sub_vol + np.sum(sub_array)
                
                sub_thick = np.sum(sub_thick_array, axis=0)
                
                sub_thick[G['mask_rho'] == 0] = np.nan # ESSENTIALLY - NEED TO REAPPLY LAND MASK
                
                sub_thick = sub_thick[min(jjj):max(jjj)+1,min(iii):max(iii)+1]
                
                sub_thick = np.ma.masked_where(np.ma.getmask(surf_casts_array), sub_thick)
                
                sub_vol_dict_obs[segment][m] = sub_vol
                
                sub_thick_dict_obs[segment][m] = sub_thick
            
            
             


# %%


for segment in segments:
    
    jjj = j_dict[segment]
    iii = i_dict[segment]
    
    
    min_lat = Lat[min(jjj) - 10]
    max_lat = Lat[max(jjj) + 10]
    
    min_lon = Lon[min(iii) - 10]
    max_lon = Lon[max(iii) + 10]    
    
    
    for (mon_num, mon_str) in zip(month_num,month_str):
        
        m = int(mon_num) - 1
 
        pfun.start_plot(fs=14, figsize=(16,9))
        fig0, axes0 = plt.subplots(nrows=1, ncols=1, squeeze=False)
        
        # if m in sub_casts_array_plot_dict[segment]:
            
        # #     cmap = cm.get_cmap('viridis', len(ii_cast_dict_obs[segment][m]))

        # #     c0 = axes0[0,0].pcolormesh(Lon[np.unique(iii)], Lat[np.unique(jjj)], surf_casts_array_dict[segment][m])#, vmin = 50, vmax = 55) #, facecolor='none', edgecolors='k', cmap='gray')
             
        #    # plt.contour(Lon[np.unique(iii)], Lat[np.unique(jjj)], surf_casts_array_dict[segment][m], 0.1, colors='white')
        
        #     c0 = axes0[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_casts_array_plot_dict[segment][m], cmap='viridis', alpha = 0.8)
        
        # if (m in ii_cast_dict_obs[segment]) and (m in jj_cast_dict_obs[segment]):
                        
        #     for n in range(len(ii_cast_dict_obs[segment][m])):
        
        #         axes0[0,0].plot(Lon[int(ii_cast_dict_obs[segment][m][n])],Lat[int(jj_cast_dict_obs[segment][m][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        # axes0[0,0].set_xlim([min_lon,max_lon])
        # axes0[0,0].set_ylim([min_lat,max_lat])
        # axes0[0,0].tick_params(labelrotation=45)
        # axes0[0,0].set_title('Sub-threshold Cast Areas')
        # pfun.add_coast(axes0[0,0])
        
        c1 = axes0[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)], sub_thick_dict_obs[segment][m], cmap='jet', alpha = 0.8, vmin = 0, vmax = 300)
        
        if (m in ii_cast_dict_obs[segment]) and (m in jj_cast_dict_obs[segment]):
                        
            for n in range(len(ii_cast_dict_obs[segment][m])):
        
                axes0[0,0].plot(Lon[int(ii_cast_dict_obs[segment][m][n])],Lat[int(jj_cast_dict_obs[segment][m][n])],'o', c = 'white', markeredgecolor='black', markersize=10)
                    
        
        axes0[0,0].set_xlim([min_lon,max_lon])
        axes0[0,0].set_ylim([min_lat,max_lat])
        axes0[0,0].tick_params(labelrotation=45)
        axes0[0,0].set_title(mon_str + ' ' + year_str + ' ' + segment + ' Sub-' + str(threshold_val) + ' mg/L DO')
        pfun.add_coast(axes0[0,0])
        
        fig0.colorbar(c1,ax=axes0[0,0], label = 'Subthreshold Thickness [m]')
        
        fig0.tight_layout()
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+segment+ '_' +mon_str+year_str+'_sub_thick_'+str(threshold_val)+'_mg_L_DO_casts_00' + mon_num+'.png')



# %%


pfun.start_plot(fs=14, figsize=(16,9))
fig1, axes1 = plt.subplots(nrows=1, ncols=1, squeeze=False)
plt.grid()




for segment in segments:

    sub_vol_obs = sub_vol_dict_obs[segment]
    
    sub_vol_obs = sorted(sub_vol_obs.items())
            
    x_time, y_vol = zip(*sub_vol_obs)
    
    x_time = np.add(x_time, 1)
    
    y_vol = np.multiply(y_vol, 1e-9)
            
    plt.plot(x_time,y_vol,label = segment)
    
    
axes1[0,0].set_xlabel('Months (2017)')
    
axes1[0,0].set_ylabel('Sub-'+str(threshold_val)+' mg/L DO Volume [km^3]')

plt.legend()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/nceiSalish_H_bottle' + year_str+'_sub_vol_'+str(threshold_val)+'_mg_L_DO.png')

# %%

outdir = '/Users/dakotamascarenas/Desktop/pltz/'

ff_str = ("ffmpeg -r 8 -i " + outdir + "G2_sub_thick_5_mg_L_DO_casts_%04d.png -vcodec libx264 -pix_fmt yuv420p -crf 25" + outdir + "G2_2017.mp4")
os.system(ff_str)

