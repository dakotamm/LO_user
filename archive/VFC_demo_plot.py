"""
VFC diagram

Test on mac in ipython:
run VFC_demo_plot -gtx cas6_v0_live -year 2017 -test False

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
from lo_tools import plotting_functions as pfun
import pickle

import numpy as np

import VFC_functions as vfun

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import cm

from datetime import datetime

import copy


# %%

Ldir = exfun.intro() # this handles the argument passing


# %%

threshold_val = 2

segments = 'basins'

var = 'DO_mg_L'

mon_num = '09'

mon_str = 'Sep'

dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)



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
    
info_df, df = vfun.getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var)

info_df['month'] = info_df['time'].dt.month

df['month'] = df['time'].dt.month

# %%

save_dir = (Ldir['LOo'] / 'extract' / 'vfc' / ('DO_' + str(threshold_val) +'mgL_' + segments + '_months_' + (str(Ldir['year']))) )

file_dir = str(save_dir)



with open((file_dir + '/' + 'sub_thick_obs.pkl'), 'rb') as f: 
    sub_thick_obs = pickle.load(f)  
    
with open((file_dir + '/' + 'sub_thick_LO_his.pkl'), 'rb') as f: 
    sub_thick_LO_his = pickle.load(f)  
    
with open((file_dir + '/' + 'sub_thick_LO_casts.pkl'), 'rb') as f: 
    sub_thick_LO_casts = pickle.load(f)  

    
with open((file_dir + '/' + 'sub_thick_obs.pkl'), 'rb') as f: 
    sub_thick_obs = pickle.load(f)  
    
with open((file_dir + '/' + 'cid_dict.pkl'), 'rb') as f: 
    cid_dict = pickle.load(f)
   
with open((file_dir + '/' + 'surf_casts_array.pkl'), 'rb') as f: 
    surf_casts_array = pickle.load(f)
    
# %%

seg_name = 'Hood Canal'

jjj = jjj_dict[seg_name]
iii = iii_dict[seg_name]


min_lat = Lat[min(jjj) - 10]
max_lat = Lat[max(jjj) + 10]

min_lon = Lon[min(iii) - 10]
max_lon = Lon[max(iii) + 10]


sub_thick_LO_his_plot = np.empty(np.shape(land_mask))

sub_thick_LO_his_plot.fill(np.nan)

sub_thick_LO_his_plot[jjj,iii] = sub_thick_LO_his[seg_name][int(mon_num)]


sub_thick_LO_casts_plot = np.empty(np.shape(land_mask))

sub_thick_LO_casts_plot.fill(np.nan)

sub_thick_LO_casts_plot[jjj,iii] = sub_thick_LO_casts[seg_name][int(mon_num)]


sub_thick_obs_plot = np.empty(np.shape(land_mask))

sub_thick_obs_plot.fill(np.nan)

sub_thick_obs_plot[jjj,iii] = sub_thick_obs[seg_name][int(mon_num)]



info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
        
df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]

land_mask_plot = land_mask.copy()

land_mask_plot[land_mask == 1] = np.nan

#cmap0 = cm.get_cmap('viridis', lut= len(info_df_use.index))


# %%

pfun.start_plot(fs=14, figsize=(15,15))
fig0, axes0 = plt.subplots(nrows=1, ncols=1, squeeze=False)

#axes0[0,0].pcolormesh(plon,plat, land_mask_plot, cmap='Greys', alpha = 0.8, vmin = -0.1, vmax = 0.1)


for cid in info_df_use.index:
            
    axes0[0,0].plot(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
    

axes0[0,0].set_xlim([min_lon,max_lon])
axes0[0,0].set_ylim([min_lat,max_lat])
axes0[0,0].tick_params(labelrotation=45)
axes0[0,0].set_title('Cast Locations ' + mon_str + ' ' + str(Ldir['year']) + ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')



pfun.add_coast(axes0[0,0])
pfun.dar(axes0[0,0])

        

fig0.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/a0.png', bbox_inches='tight')


# %%


pfun.start_plot(fs=14, figsize=(15,15))
fig0, axes0 = plt.subplots(nrows=1, ncols=1, squeeze=False)

c0 = axes0[0,0].pcolormesh(plon,plat, sub_thick_LO_his_plot, cmap='Blues', alpha = 0.8, vmin = 0, vmax = 300)


for cid in info_df_use.index:
            
    axes0[0,0].plot(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
        

axes0[0,0].set_xlim([min_lon,max_lon])
axes0[0,0].set_ylim([min_lat,max_lat])
axes0[0,0].tick_params(labelrotation=45)
axes0[0,0].set_title('Model Output ' + mon_str + ' ' + str(Ldir['year']) + ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')

fig0.colorbar(c0,ax=axes0[0,0], label = 'Subthreshold Thickness [m]')


pfun.add_coast(axes0[0,0])
pfun.dar(axes0[0,0])

        

fig0.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/a1.png', bbox_inches='tight')

# %%

pfun.start_plot(fs=14, figsize=(15,15))
fig0, axes0 = plt.subplots(nrows=1, ncols=1, squeeze=False)

c0 = axes0[0,0].pcolormesh(plon,plat, sub_thick_LO_casts_plot, cmap='Purples', alpha = 0.8, vmin = 0, vmax = 300)

for cid in info_df_use.index:
            
    axes0[0,0].plot(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
    

axes0[0,0].set_xlim([min_lon,max_lon])
axes0[0,0].set_ylim([min_lat,max_lat])
axes0[0,0].tick_params(labelrotation=45)
axes0[0,0].set_title('Model Casts ' + mon_str + ' ' + str(Ldir['year']) + ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')

fig0.colorbar(c0,ax=axes0[0,0], label = 'Subthreshold Thickness [m]')


pfun.add_coast(axes0[0,0])
pfun.dar(axes0[0,0])

        

fig0.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/a2.png', bbox_inches='tight')

# %%

surf_casts_array_full = np.empty(np.shape(land_mask))
surf_casts_array_full.fill(np.nan)
    
surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array[seg_name][int(mon_num)])

sub_thick_array = np.empty(np.shape(z_rho_grid))

sub_casts_array_full = np.empty(np.shape(surf_casts_array_full))

sub_casts_array_full.fill(np.nan)


LO_casts_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' / (str(Ldir['year'])) / (str(info_df_use['segment'].iloc[0]) + '_' + str(info_df_use['time'].dt.date.min()) + '_' + str(info_df_use['time'].dt.date.max()) ) )

df0 = pd.DataFrame()

df0['cid'] = []
    
df0['z_rho'] = []

df0[var] = []

for cid in info_df_use.index:
    
    df_temp = pd.DataFrame()    
    
    fn = (LO_casts_dir) / (str(cid) + '.nc')
    
    if fn.exists(): 
    
        z_rho, var_out = vfun.getLOCastsAttrs(fn) #need to generalize
                
        df_temp['z_rho'] = z_rho
        
        df_temp[var] = var_out
        
        df_temp['cid'] = cid
        
        df0 = pd.concat([df0, df_temp])
        
df_sub = df0[df0[var] < threshold_val]


if df_sub.empty: # if no subthreshold values
    
    sub_vol = 0
    
    sub_thick_array.fill(0)
    
    sub_thick_temp = np.sum(sub_thick_array, axis=0)
    
    sub_thick = sub_thick_temp[jjj,iii]
                            
    sub_casts_array = sub_casts_array_full[jjj,iii]
    
    print('no sub LO casts')

    
else: # if subthreshold values!
                           
     info_df_sub = info_df_use.copy(deep=True)
 
     for cid in info_df_use.index:
         
         if cid not in df_sub['cid'].unique():
             
             info_df_sub = info_df_sub.drop(cid)
     
     sub_casts_array_temp = copy.deepcopy(surf_casts_array[seg_name][int(mon_num)])

     sub_casts_array_temp0 = [[ele if ele in df_sub['cid'].unique() else -99 for ele in line] for line in sub_casts_array_temp]

     sub_casts_array_temp1 = np.array(sub_casts_array_temp0)

     sub_casts_array =np.ma.masked_array(sub_casts_array_temp1,sub_casts_array_temp1==-99)
     
     sub_casts_array_full[min(jjj):max(jjj)+1, min(iii):max(iii)+1] = sub_casts_array.copy()
     
     sub_array = np.empty(np.shape(z_rho_grid))
     sub_array.fill(0)

     sub_thick_array.fill(0)
                  
     

     for cid in info_df_sub.index:
         
         df_temp = df_sub[df_sub['cid']==cid]
         
         z_rho_array = z_rho_grid[:, int(info_df_use.loc[cid,'jj_cast']), int(info_df_use.loc[cid,'ii_cast'])].copy()
         
         n = 0
         
         for depth in z_rho_array:
             
             if depth not in np.asarray(df_temp['z_rho']):
                 
                 z_rho_array[n] = np.nan
                 
             n +=1
                 
         z_rho_array_full = np.repeat(z_rho_array[:,np.newaxis], np.size(z_rho_grid, axis=1), axis=1)
         
         z_rho_array_full_3d = np.repeat(z_rho_array_full[:,:,np.newaxis], np.size(z_rho_grid, axis=2), axis=2)
         
         sub_casts_array_full_3d = np.repeat(sub_casts_array_full[np.newaxis,:,:], np.size(z_rho_grid, axis=0), axis=0)
                                           
         sub_array[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))] = dv[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))].copy()
         
         sub_thick_array[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))] = dz[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))].copy()
                  
         
            
     sub_vol = np.sum(sub_array)
    
     sub_thick_temp = np.sum(sub_thick_array, axis=0)
    
     sub_thick = sub_thick_temp[jjj,iii]
     
# %%

import pinfo
from importlib import reload
reload(pinfo)

# min_lat = [48, 48.4]
# max_lat = [49, 48.7]
# min_lon = [-124, -123.4]
# max_lon = [-122.25,-122.4]

min_lon_sect = Lon[498]
max_lon_sect = Lon[523]
min_lat_sect = Lat[704]
max_lat_sect = Lat[744]

vn = 'oxygen'

x_e = np.linspace(min_lon_sect, max_lon_sect, 10000)
y_e = max_lat_sect * np.ones(x_e.shape)

cmap=pinfo.cmap_dict[vn]


