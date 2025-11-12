#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 15:02:13 2025

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

from pygam import LinearGAM, s

import matplotlib.patheffects as pe

# %%




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




poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] #, 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

#poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson '], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

odf_use_seasonal_DO, odf_use_seasonal_CTSA, odf_use_annual_DO, odf_use_annual_CTSA = dfun.calcSeriesAvgs(odf_depth_mean, odf_depth_mean_deep_DO_percentiles, deep_DO_q = 'deep_DO_q50')

# %%

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'point_jefferson', 'site_label'] = 'PJ'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'point_jefferson', 'site_num'] = 1

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'near_seattle_offshore', 'site_num'] = 2

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'saratoga_passage_mid', 'site_num'] = 4

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'carr_inlet_mid', 'site_num'] = 3

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['site'] == 'lynch_cove_mid', 'site_num'] = 5


odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['season'] == 'grow', 'season_label'] = 'Spring (Apr-Jul)'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['season'] == 'loDO', 'season_label'] = 'Low-DO (Aug-Nov)'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['season'] == 'winter', 'season_label'] = 'Winter (Dec-Mar)'


odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['surf_deep'] == 'surf', 'depth_label'] = 'Surface'

odf_use_seasonal_CTSA.loc[odf_use_seasonal_CTSA['surf_deep'] == 'deep', 'depth_label'] = 'Bottom'


# %%

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'point_jefferson', 'site_label'] = 'PJ'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'point_jefferson', 'site_num'] = 1

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'near_seattle_offshore', 'site_num'] = 2

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'saratoga_passage_mid', 'site_num'] = 4

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'carr_inlet_mid', 'site_num'] = 3

odf_use_annual_CTSA.loc[odf_use_annual_CTSA['site'] == 'lynch_cove_mid', 'site_num'] = 5



# %%

red =     "#EF5E3C"   # warm orange-red ##ff4040 #e04256

blue =     "#3A59B3"  # deep blue #4565e8

yellow =     "#C7C445"  # yellow-green '#dd9404'


edgecolors = {'Surface': None, 'Bottom': 'k'}

palette = {'Spring (Apr-Jul)': yellow, 'Low-DO (Aug-Nov)': red, 'Winter (Dec-Mar)': blue}

markers  = {'DO_mg_L': 'o', 'CT':'D', 'SA':'s'}

#palette = {'point_jefferson': '#e04256', 'near_seattle_offshore': '#e04256', 'carr_inlet_mid':'#4565e8', 'saratoga_passage_mid':'#4565e8', 'lynch_cove_mid': '#4565e8'}

#palette = {'point_jefferson': '#e04256', 'near_seattle_offshore': '#e04256', 'carr_inlet_mid':'#4565e8', 'saratoga_passage_mid':'#4565e8', 'lynch_cove_mid': '#4565e8'}

ymins = {'DO_mg_L': 0, 'CT': 7, 'SA': 20}

# ymaxs = {'DO_mg_L': 0, 'CT': 7, 'SA': 20}



jitter = {'Winter (Dec-Mar)': -0.1, 'Spring (Apr-Jul)':0, 'Low-DO (Aug-Nov)': 0.1}

mosaic = [['CT', 'SA', 'DO_mg_L']]


#mosaic = [['surf_DO_mg_L', 'surf_CT', 'surf_SA'], ['deep_DO_mg_L', 'deep_CT', 'deep_SA']]

fig, axd = plt.subplot_mosaic(mosaic, sharex=True, figsize=(9,2.5), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))

for var in var_list:
    
    ax_name = var
    
    ax = axd[ax_name]
    
    for depth in [ 'Surface', 'Bottom']: 
        
        for site in long_site_list:
            
            for season in ['Winter (Dec-Mar)', 'Spring (Apr-Jul)', 'Low-DO (Aug-Nov)']:
                
                plot_df = odf_use_seasonal_CTSA[(odf_use_seasonal_CTSA['var'] == var) & (odf_use_seasonal_CTSA['season_label'] == season) & (odf_use_seasonal_CTSA['site'] == site) & (odf_use_seasonal_CTSA['depth_label'] == depth)]


                if var == 'SA':
                    
                    ax.scatter(plot_df['site_num'] + jitter[season], plot_df['val_mean'], color='gray', s=50, marker= markers[var], edgecolors=edgecolors[depth], label=depth)
                    
                    ax.scatter(plot_df['site_num'] + jitter[season], plot_df['val_mean'], color=palette[season], s=50, marker= markers[var], edgecolors=edgecolors[depth])
                    
                elif var == 'CT':
                    
                    
                    ax.scatter(plot_df['site_num'] + jitter[season], plot_df['val_mean'], color=palette[season], s=50, marker= markers[var], edgecolors=edgecolors[depth], label=season)
                    
                else:
                    
                    ax.scatter(plot_df['site_num'] + jitter[season], plot_df['val_mean'], color=palette[season], s=50, marker= markers[var], edgecolors=edgecolors[depth])

                 
                ax.plot([plot_df['site_num'] + jitter[season], plot_df['site_num'] + jitter[season]],[plot_df['val_ci95lo'], plot_df['val_ci95hi']], color=palette[season], alpha =0.5, zorder = -5, linewidth=1)
              
            # plot_df_ = odf_use_annual_CTSA[(odf_use_annual_CTSA['var'] == var) & (odf_use_annual_CTSA['site'] == site) & (odf_use_annual_CTSA['surf_deep'] == depth)]
            
            # ax.scatter(plot_df_['site_num'], plot_df_['val_mean'], color='gray', s=20, marker= markers[var], edgecolors=edgecolors[depth])
             
            # ax.plot([plot_df_['site_num'], plot_df_['site_num']],[plot_df_['val_ci95lo'], plot_df_['val_ci95hi']], color='gray', alpha =0.5, zorder = -5, linewidth=1)
            
                
    if var == 'DO_mg_L':  
    
        ax.axhspan(0,2, color = 'lightgray', alpha = 0.5, zorder=-6, label='Hypoxia') 
        
        ax.text(0.05,0.05, 'c', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

        
        ax.set_ylabel('Mean [DO] [mg/L]')
        
        ax.legend()
        
        handles_DO_mg_L, labels_DO_mg_L = ax.get_legend_handles_labels()
    
    elif var == 'SA':
        
        ax.set_ylabel('Mean Salinity [g/kg]')
        
        ax.text(0.05,0.05, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

        
        handles, labels = ax.get_legend_handles_labels()
        
        handles_SA = [handles[0], handles[-3]]
        labels_SA = [labels[0], labels[-3]]
        
        ax.legend(handles_SA, labels_SA, loc = 'lower center')        
        
    else:
        
        ax.set_ylabel('Mean Temperature [°C]')

        
        ax.text(0.05,0.05, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')

        
        handles, labels = ax.get_legend_handles_labels()
            
        handles_CT = [handles[0], handles[1], handles[2]]
        labels_CT = [labels[0], labels[1], labels[2]]
        
        ax.legend(handles_CT, labels_CT, loc = 'best')

     
    #ax.text(0.05,0.95, var, transform=ax.transAxes, verticalalignment='top', fontweight = 'bold', color='k')
     
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.3, zorder = -6)

    ax.axhline(0, color='gray', linestyle = '--', zorder = -5) 

    #ax.set_ylabel(var)
     
    ax.set_ylim(ymin=ymins[var]) 
    
    ax.set_xticks([1,2,3,4,5],['PJ', 'NS', 'CI', 'SP', 'LC'])

handles = handles_CT + handles_SA + handles_DO_mg_L

labels = labels_CT + labels_SA + labels_DO_mg_L

fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.01),  # left side
    ncol=len(handles)
    #title='Data Source'
    )

axd['CT'].get_legend().remove()

axd['SA'].get_legend().remove()


axd['DO_mg_L'].get_legend().remove()
        
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_7.png', bbox_inches='tight', dpi=500, transparent=True)