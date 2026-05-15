#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 13:39:58 2025

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

import matplotlib.patheffects as pe




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



poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%
# %%

odf_use = odf_depth_mean.copy()

odf_use = (odf_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

#odf_calc_use = odf_calc_long.copy()



# %%

# odf_use_DO = odf_depth_mean_deep_DO_percentiles.copy()

# odf_use_DO_q50 = odf_use_DO[odf_use_DO['val'] <= odf_use_DO['deep_DO_q50']]

# maybe just show all the DO values here...like don't filter for the time series

# %%

odf_use.loc[odf_use['site'] == 'point_jefferson', 'site_label'] = 'Point Jefferson (PJ) [Main Basin]'

odf_use.loc[odf_use['site'] == 'near_seattle_offshore', 'site_label'] = 'Near Seattle (NS) [Main Basin]'

odf_use.loc[odf_use['site'] == 'carr_inlet_mid', 'site_label'] = 'Carr Inlet (NS) [Sub-Basins: South Sound]'

odf_use.loc[odf_use['site'] == 'saratoga_passage_mid', 'site_label'] = 'Saratoge Passage (SP) [Sub-Basins: Whidbey Basin]'

odf_use.loc[odf_use['site'] == 'lynch_cove_mid', 'site_label'] = 'Lynch Cove (LC) [Sub-Basins: Hood Canal]'

# %%

odf_use.loc[odf_use['season'] == 'grow', 'season_label'] = 'Apr-Jul'

odf_use.loc[odf_use['season'] == 'loDO', 'season_label'] = 'Aug-Nov'

odf_use.loc[odf_use['season'] == 'winter', 'season_label'] = 'Dec-Mar'

# %%

mosaic = [['point_jefferson_deep', 'point_jefferson_surf'], ['near_seattle_offshore_deep', 'near_seattle_offshore_surf'], ['carr_inlet_mid_deep', 'carr_inlet_mid_surf'], ['saratoga_passage_mid_deep', 'saratoga_passage_mid_surf'], ['lynch_cove_mid_deep', 'lynch_cove_mid_surf']]

c=6
for var in ['CT', 'SA', 'DO_mg_L']:
                            
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
        
        unit = r'[g/kg]'
    
    fig, axd = plt.subplot_mosaic(mosaic, sharex=True, sharey=True, figsize=(9,12), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))
    
    for site in long_site_list:
        
        for depth in ['surf', 'deep']:
                
            ax_name = site + '_' + depth
            
            ax = axd[ax_name]
            
            palette = {'Apr-Jul':'#dd9404', 'Aug-Nov':'#e04256', 'Dec-Mar': '#4565e8'}
                    
            plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == var) & (odf_use['surf_deep'] == depth)]
            
            sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, hue='season_label', palette = palette, marker=marker)
            
            #ax.scatter(x=0, y =0, color = palette[site], marker='o', label = site_label)
                
            if var == 'DO_mg_L':  
            
                ax.axhspan(0,2, color = 'gray', alpha = 0.3, zorder=-5, label='Hypoxia')
                
                #ax.legend(loc='upper right')
                
                #ax.text(0.025,0.05, 'd', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
                    
                
            # elif var == 'CT':
                
            #     ax.legend(ncol=2, loc='upper left')
                
            #     ax.text(0.025,0.05, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
                
            # else:
                
            #     ax.text(0.025,0.05, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

                        
                        
            #ax.set_ylim(ymin, ymax) 
                    
            ax.set_ylabel(label_var + ' ' + unit)
            
            ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax.set_xlabel('')
                    
                    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_'+ str(c) +'.png', bbox_inches='tight', dpi=500, transparent=True)
    
    c+=1
            
            