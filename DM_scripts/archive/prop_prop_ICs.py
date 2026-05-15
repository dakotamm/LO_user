#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:51:33 2023

@author: dakotamascarenas
"""

import VFC_functions as vfun

import pandas as pd

import numpy as np

from lo_tools import forcing_argfun2 as ffun

from lo_tools import zrfun

from lo_tools import plotting_functions as pfun

import matplotlib.pyplot as plt

import pickle

import seaborn as sns
# %%

SMALL_SIZE =12
MEDIUM_SIZE = 16
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%


with open('/Users/dakotamascarenas/Desktop/V_old.pkl', 'rb') as handle:
    V_old = pickle.load(handle)
    

with open('/Users/dakotamascarenas/Desktop/V_new.pkl', 'rb') as handle:
    V_new = pickle.load(handle)
    
with open('/Users/dakotamascarenas/Desktop/G.pkl', 'rb') as handle:
    G = pickle.load(handle)
    
with open('/Users/dakotamascarenas/Desktop/S.pkl', 'rb') as handle:
    S = pickle.load(handle)

with open('/Users/dakotamascarenas/Desktop/T.pkl', 'rb') as handle:
    T = pickle.load(handle)
    
with open('/Users/dakotamascarenas/Desktop/jjj_dict.pkl', 'rb') as handle:
    jjj_dict = pickle.load(handle)
    
with open('/Users/dakotamascarenas/Desktop/iii_dict.pkl', 'rb') as handle:
    iii_dict = pickle.load(handle)
    
info_df_ctd = pd.read_pickle('/Users/dakotamascarenas/Desktop/info_df_ctd.p')

info_df_bottle = pd.read_pickle('/Users/dakotamascarenas/Desktop/info_df_bottle.p')


# %%

var_list = ['salt','temp','NO3','NH4','chlorophyll','TIC','alkalinity','oxygen']

land_mask = G['mask_rho']
Lon = G['lon_rho'][0,:]
Lat = G['lat_rho'][:,0]
plon,plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])
z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
dz = np.diff(z_w_grid,axis=0)
dv = dz*G['DX']*G['DY']
h = G['h']

# %%

seg_array = np.empty(np.shape(z_rho_grid))
seg_array.fill(np.nan)

c= 0

seg_list = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Main Basin', 'Whidbey Basin', 'Hood Canal', 'South Sound', 'Admiralty Inlet', 'Tacoma Narrows']

for seg_name in seg_list:
    
    seg_array[:,jjj_dict[seg_name],iii_dict[seg_name]] = c
    
    c +=1
    


# %%

XX,YY,ZZ = np.meshgrid(np.arange(V_old['salt'][0,:,:,:].shape[2]), np.arange(V_old['salt'][0,:,:,:].shape[1]), np.arange(V_old['salt'][0,:,:,:].shape[0]))
table = np.vstack((XX.flatten(),YY.flatten(), ZZ.flatten())).T
#table_df = pd.DataFrame(table)    

# %%

df = pd.DataFrame()

for var in var_list:
    
    col_old = var + '_old'
    
    col_new = var + '_new'
    
    df[col_old] = V_old[var][0,:,:,:].flatten()
    
    df[col_new] = V_new[var][0,:,:,:].flatten()
    
df['z_rho'] = z_rho_grid.flatten()

df['iii'] = table[:,0]

df['jjj'] = table[:,1]

df['kkk'] = table[:,2]

df['segment'] = seg_array.flatten()
        

# %%

df_nonans = df[~np.isnan(df[col_old])]
df_sea = df_nonans[~np.isnan(df_nonans['segment'])]

# %%

for var in var_list:
    
   # var = 'salt'
    
    fig, ax = plt.subplots(1,1,figsize=(15,15))
        
    col_old = var + '_old'
    
    col_new = var + '_new'
    
    sns.scatterplot(data = df_sea, x = col_old, y = col_new, hue = 'z_rho', palette = 'flare_r')

    if var == 'chlorophyll':
        
        units = '[mg m-3]'
        
        clim_min = 0.022
        
        clim_max = 0.028
        
    elif var == 'salt':
        
        units = '[PSU]'
        
        clim_min = 20
        
        clim_max = 35
        
    elif var == 'temp':
        
        units = '[deg C]'
        
        clim_min = 4
        
        clim_max = 14
        
    elif var == 'NO3':
        
        units = '[uM]'
        
        clim_min = 0
        
        clim_max = 45
        
    elif var == 'NH4':
        
        units = '[uM]'
        
        clim_min = -0.1
        
        clim_max = 0.1
    
    elif var == 'TIC':
    
        units = '[uM]'
    
        clim_min = 1700
    
        clim_max = 2500
        
    elif var == 'alkalinity':
        
        units = '[uM]'
        
        clim_min = 1900
        
        clim_max = 2500
        
    elif var == 'oxygen':
        
        units = '[uM]'
        
        clim_min = 0
        
        clim_max = 350
        
    plt.axline((clim_min,clim_min), slope =1, alpha = 0.5, color = 'k')

    ax.set(xlabel = 'original ' + var + ' [' + units +']', ylabel = 'new ' + var + ' ' + units, xlim = [clim_min, clim_max], ylim = [clim_min, clim_max])
    
    ax.grid(alpha=0.3)
    
    plt.legend(title = 'depth [m]')
    
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + var + '_prop-prop_all.png', transparent = False, dpi=500)
    
   # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + seg_name.replace(" ", "") + '_avgbot20pct_seasonal.png', transparent = False, dpi=500)
    
    
# %%

c= 0

for seg_name in seg_list:
    
    df_seg = df_sea[df_sea['segment'] == c]
    
    min_lat = Lat[min(jjj_dict[seg_name])]
    max_lat = Lat[max(jjj_dict[seg_name])]

    min_lon = Lon[min(iii_dict[seg_name])]
    max_lon = Lon[max(iii_dict[seg_name])]
    
   # df_seg = df_seg[~np.isnan(df_seg['iii'] == nan]

    for var in var_list:
                
        fig, ax = plt.subplots(1,2,figsize=(30, 15))
            
        col_old = var + '_old'
        
        col_new = var + '_new'
        
        sns.scatterplot(data = df_seg, x = col_old, y = col_new, hue = 'z_rho', palette = 'flare_r', ax = ax[0])
        
        if var == 'temp' or var == 'salt':
            
            info_df_ctd_seg = info_df_ctd[info_df_ctd['segment'] == seg_name]
            
            if ~info_df_ctd_seg.empty:

                for cid in info_df_ctd_seg.index:
                            
                    ax[1].plot(info_df_ctd_seg.loc[cid, 'lon'], info_df_ctd_seg.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
                
        else:
            
            info_df_bottle_seg = info_df_bottle[info_df_bottle['segment'] == seg_name]
            
            if ~info_df_bottle_seg.empty:
            
                for cid in info_df_bottle_seg.index:
                            
                    ax[1].plot(info_df_bottle_seg.loc[cid, 'lon'], info_df_bottle_seg.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
        
    
        if var == 'chlorophyll':
            
            units = '[mg m-3]'
            
            clim_min = 0.022
            
            clim_max = 0.028
            
        elif var == 'salt':
            
            units = '[PSU]'
            
            clim_min = 20
            
            clim_max = 35
            
        elif var == 'temp':
            
            units = '[deg C]'
            
            clim_min = 4
            
            clim_max = 14
            
        elif var == 'NO3':
            
            units = '[uM]'
            
            clim_min = 0
            
            clim_max = 45
            
        elif var == 'NH4':
            
            units = '[uM]'
            
            clim_min = -0.1
            
            clim_max = 0.1
        
        elif var == 'TIC':
        
            units = '[uM]'
        
            clim_min = 1700
        
            clim_max = 2500
            
        elif var == 'alkalinity':
            
            units = '[uM]'
            
            clim_min = 1900
            
            clim_max = 2500
            
        elif var == 'oxygen':
            
            units = '[uM]'
            
            clim_min = 0
            
            clim_max = 350
            
        ax[0].axline((clim_min,clim_min), slope =1, alpha = 0.5, color = 'k')
    
        ax[0].set(xlabel = 'original ' + var + ' [' + units +']', ylabel = 'new ' + var + ' ' + units, xlim = [clim_min, clim_max], ylim = [clim_min, clim_max])
        
        ax[0].grid(alpha=0.3)
        
        ax[0].legend(title = 'depth [m]')
        
        ax[1].set(xlim = [min_lon, max_lon], ylim = [min_lat, max_lat])
        
        pfun.add_coast(ax[1])
        pfun.dar(ax[1])
        
        fig.tight_layout()
            
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + var + '_prop-prop_' + seg_name.replace(" ","") + '.png', transparent = False, dpi=500)
        
       # plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + seg_name.replace(" ", "") + '_avgbot20pct_seasonal.png', transparent = False, dpi=500)
       
        c+=1
    
    
    