#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:29:56 2024

@author: dakotamascarenas
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import matplotlib.pyplot as plt
import matplotlib.path as mpth
import xarray as xr
import numpy as np
import pandas as pd
import datetime

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

import seaborn as sns

import scipy.stats as stats

import D_functions as dfun

import pickle

import math

from scipy.interpolate import interp1d

import gsw

import matplotlib.path as mpth

import matplotlib.patches as patches

import cmocean




Ldir = Lfun.Lstart(gridname='cas7')


fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)
lon = dsg.lon_rho.values
lat = dsg.lat_rho.values
m = dsg.mask_rho.values
xp, yp = pfun.get_plon_plat(lon,lat)
depths = dsg.h.values
depths[m==0] = np.nan

lon_1D = lon[0,:]

lat_1D = lat[:,0]

# weird, to fix

mask_rho = np.transpose(dsg.mask_rho.values)
zm = -depths.copy()
zm[np.transpose(mask_rho) == 0] = np.nan
zm[np.transpose(mask_rho) != 0] = -1

zm_inverse = zm.copy()

zm_inverse[np.isnan(zm)] = -1

zm_inverse[zm==-1] = np.nan


X = lon[0,:] # grid cell X values
Y = lat[:,0] # grid cell Y values

plon, plat = pfun.get_plon_plat(lon,lat)


j1 = 570
j2 = 1170
i1 = 220
i2 = 652

# %%

poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf['ix_iy'] = odf['ix'].astype(str).apply(lambda x: x.zfill(4)) + '_' + odf['iy'].astype(str).apply(lambda x: x.zfill(4))


# %%

odf_ixiy_unique = odf.groupby(['ix_iy']).first().reset_index()

# %%



#poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson', 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()






for deep_DO_q in ['deep_DO_q50']:


    odf_depth_mean_deep_DO_less_than_percentile = odf_depth_mean_deep_DO_percentiles[odf_depth_mean_deep_DO_percentiles['val'] <= odf_depth_mean_deep_DO_percentiles[deep_DO_q]]

    cid_deep_DO_less_than_percentile = odf_depth_mean_deep_DO_less_than_percentile['cid']

    odf_use_DO_q = odf_depth_mean[odf_depth_mean['cid'].isin(cid_deep_DO_less_than_percentile)]

    odf_calc_use_DO_q = odf_calc_long[odf_calc_long['cid'].isin(cid_deep_DO_less_than_percentile)]

    odf_use_DO_q = (odf_use_DO_q
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    odf_calc_use_DO_q = (odf_calc_use_DO_q
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )

    
    odf_use_DO_q_AugNov = odf_use_DO_q[odf_use_DO_q['season'] == 'loDO']
    
    odf_calc_use_DO_q_AugNov = odf_calc_use_DO_q[odf_calc_use_DO_q['season'] == 'loDO']

    
    if deep_DO_q == 'deep_DO_q25':
    
        odf_use_q25 = odf_use_DO_q_AugNov
        
        odf_calc_use_q25 = odf_calc_use_DO_q_AugNov
        
    elif deep_DO_q == 'deep_DO_q50':
        
        odf_use_q50 = odf_use_DO_q_AugNov
        
        odf_calc_use_q50 = odf_calc_use_DO_q_AugNov
        
    elif deep_DO_q == 'deep_DO_q75':
        
        odf_use_q75 = odf_use_DO_q_AugNov
        
        odf_calc_use_q75 = odf_calc_use_DO_q_AugNov


# %%

odf_use = (odf_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

# %%


mosaic = [['CT', 'CT'], ['SA', 'SA'], [ 'DO_mg_L', 'DO_mg_L']] #, ['map_source', '.', '.'],]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(9,6), layout='constrained', gridspec_kw=dict(wspace=0.1), sharex = True)



# ax = axd['map_source']
 
# ax.set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
# ax.set_ylim(Y[j1],Y[j2]) # Salish Sea
        
# ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-20, vmax=0, cmap = 'gray', zorder=-5)

# sns.scatterplot(data=odf_ixiy_unique, x='lon', y='lat', ax = ax, color = 'gray', alpha=0.3, label= 'Cast Location')


# pfun.add_coast(ax)

# pfun.dar(ax)

# for site in long_site_list:
    
#     path = path_dict[site]
        
#     if site in ['near_seattle_offshore']:
        
#         patch = patches.PathPatch(path, facecolor='#e04256', edgecolor='white', zorder=1, label='Main Basin')
    
#     elif site in ['point_jefferson']:
            

#         patch = patches.PathPatch(path, facecolor='#e04256', edgecolor='white', zorder=1)
                
#     elif site in ['saratoga_passage_mid']:
        
#         patch = patches.PathPatch(path, facecolor='#4565e8', edgecolor='white', zorder=1, label = 'Sub-Basins')
        
#     else:
        
#         patch = patches.PathPatch(path, facecolor='#4565e8', edgecolor='white', zorder=1)
         
#     ax.add_patch(patch)
    
# ax.text(0.75,0.5, 'PJ', transform=ax.transAxes, fontsize=14, color = '#e04256')

# ax.text(0.54,0.32, 'NS', transform=ax.transAxes, fontsize=12, color = '#e04256')

    
# ax.text(0.62,0.67, 'SP', transform=ax.transAxes, fontsize=14, color = '#4565e8')

# ax.text(0.22,0.29, 'LC', transform=ax.transAxes, fontsize=14, color = '#4565e8')
 
# ax.text(0.49,0.2, 'CI', transform=ax.transAxes, fontsize=14, color = '#4565e8')


# ax.text(0.05,0.025, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')



# ax.legend(loc = 'upper left')

# ax.set_xlim(-123.2, -122.1) 

# ax.set_ylim(47,48.5)


# ax.set_xlabel('')

# ax.set_ylabel('')

# ax.tick_params(axis='x', labelrotation=45)

palette = {'point_jefferson':'#e04256', 'lynch_cove_mid':'#4565e8'}


for var in var_list:
    
    ax = axd[var]
                        
    if 'DO' in var:
        
        label_var = '[DO]'
        
        ymin = 0
        
        ymax = 8
        
        marker = 'o'
        
        unit = r'[mg/L]'
        
    elif 'CT' in var:
        
        label_var = 'Temperature'
        
        ymin = 8
        
        ymax = 14
        
        marker = 'D'
        
        unit = r'[$^{\circ}$C]'
    
    else:
        
        label_var = 'Salinity'
        
        ymin = 28
        
        ymax = 34
        
        marker = 's'
        
        unit = r'[PSU]'

    for site in ['point_jefferson', 'lynch_cove_mid']:
        
        if site == 'point_jefferson':
            
            site_label = 'Point Jefferson'
        
        else:
            
            site_label = 'Lynch Cove'
        
        if var == 'DO_mg_L':
            
            plot_df_q50 = odf_use_q50[(odf_use_q50['site'] == site) & (odf_use_q50['var'] == var) & (odf_use_q50['surf_deep'] == 'deep')] #already filtered to loDO
        
            sns.scatterplot(data=plot_df_q50, x='datetime', y = 'val',  ax=ax, color = palette[site], marker=marker)
            
        elif var == 'CT':
            
            plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == var) & (odf_use['surf_deep'] == 'deep') & (odf_use['season'] == 'loDO')]
            
            sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, color = palette[site], marker=marker)
            
            ax.scatter(x=0, y =0, color = palette[site], marker='o', label = site_label)

            
        else:
            
            plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == var) & (odf_use['surf_deep'] == 'deep') & (odf_use['season'] == 'loDO')]
            
            sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, color = palette[site], marker=marker)
            
    
    if var == 'DO_mg_L':  
    
        ax.axhspan(0,2, color = 'gray', alpha = 0.3, zorder=-5, label='Hypoxia')
        
        ax.legend(loc='upper left')
        
       # ax.text(0.025,0.05, 'd', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
        
    
    elif var == 'CT':
        
        ax.legend(ncol=2, loc='upper left')
        
       # ax.text(0.025,0.05, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
        
    #else:
        
       # ax.text(0.025,0.05, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

            
            
    ax.set_ylim(ymin, ymax) 
            
    ax.set_ylabel('Filtered ' + label_var + ' ' + unit)
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax.set_xlabel('')
        
        
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pj_lc_timeseries_EPOC.png', bbox_inches='tight', dpi=500, transparent=True)
    
