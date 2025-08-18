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

from pygam import LinearGAM, s





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

odf_use_DO = odf_depth_mean_deep_DO_percentiles.copy()

odf_use_DO_q50 = odf_use_DO[odf_use_DO['val'] <= odf_use_DO['deep_DO_q50']]

odf_use_DO_q50 = (odf_use_DO_q50
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

#maybe just show all the DO values here...like don't filter for the time series

# %%

odf_use.loc[odf_use['site'] == 'point_jefferson', 'site_label'] = 'PJ'

odf_use.loc[odf_use['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

odf_use.loc[odf_use['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

odf_use.loc[odf_use['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

odf_use.loc[odf_use['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


odf_use.loc[odf_use['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

odf_use.loc[odf_use['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

odf_use.loc[odf_use['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

odf_use.loc[odf_use['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

odf_use.loc[odf_use['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'

# %%

odf_use_DO_q50.loc[odf_use_DO_q50['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

odf_use_DO_q50.loc[odf_use_DO_q50['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

odf_use_DO_q50.loc[odf_use_DO_q50['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

odf_use_DO_q50.loc[odf_use_DO_q50['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

odf_use_DO_q50.loc[odf_use_DO_q50['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'

# %%

odf_use.loc[odf_use['season'] == 'grow', 'season_label'] = 'Apr-Jul'

odf_use.loc[odf_use['season'] == 'loDO', 'season_label'] = 'Aug-Nov'

odf_use.loc[odf_use['season'] == 'winter', 'season_label'] = 'Dec-Mar'

# %%

mosaic = [['point_jefferson_CT', 'point_jefferson_SA', 'point_jefferson_DO_mg_L'], ['near_seattle_offshore_CT', 'near_seattle_offshore_SA', 'near_seattle_offshore_DO_mg_L'],['carr_inlet_mid_CT', 'carr_inlet_mid_SA', 'carr_inlet_mid_DO_mg_L'], ['saratoga_passage_mid_CT', 'saratoga_passage_mid_SA', 'saratoga_passage_mid_DO_mg_L'],['lynch_cove_mid_CT','lynch_cove_mid_SA', 'lynch_cove_mid_DO_mg_L']]

fig, axd = plt.subplot_mosaic(mosaic, sharex=True, figsize=(9, 12), layout='constrained', gridspec_kw=dict(wspace=0.1, hspace=0.1))

plot_labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']

c=0

for site in ['point_jefferson','near_seattle_offshore', 'carr_inlet_mid', 'saratoga_passage_mid', 'lynch_cove_mid']:

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
            
            ymax = 15
            
            marker = 'D'
            
            unit = r'[$^{\circ}$C]'
        
        else:
            
            label_var = 'Salinity'
            
            ymin = 28
             
            ymax = 32
            
            marker = 's'
            
            unit = r'[g/kg]'
    
                
        ax_name = site + '_' + var
        
        ax = axd[ax_name]
        
        for depth in ['deep']:
            
            for season in ['loDO']:
            
                palette = {'Main Basin':'#e04256', 'Sub-Basins': '#4565e8'}
                
                plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == var) & (odf_use['surf_deep'] == depth) & (odf_use['season'] == season)]
                
                # Define the GAM model
                gam0 = LinearGAM(s(0))
                gam1 = LinearGAM(s(0))

                
                # Fit the model to the data
                gam0.fit(plot_df[plot_df['year']<=1988]['datetime'], plot_df[plot_df['year']<=1988]['val'])
                
                x_pred0 = plot_df[plot_df['year']<=1988]['datetime']
                y_pred0 = gam0.predict(x_pred0)
                
                # Fit the model to the data
                gam0.fit(plot_df[plot_df['year']<=1988]['datetime'], plot_df[plot_df['year']<=1988]['val'])
                
                x_pred0 = plot_df[plot_df['year']<=1988]['datetime']
                y_pred0 = gam0.predict(x_pred0)
                
                # Fit the model to the data
                gam1.fit(plot_df[plot_df['year']>1988]['datetime'], plot_df[plot_df['year']>1988]['val'])
                
                x_pred1 = plot_df[plot_df['year']>1988]['datetime']
                y_pred1 = gam1.predict(x_pred1)
                
                if var == 'DO_mg_L':  
                                    
                    plot_df_DO_q50 = odf_use_DO_q50[(odf_use_DO_q50['site'] == site) & (odf_use_DO_q50['var'] == var) & (odf_use_DO_q50['surf_deep'] == depth) & (odf_use_DO_q50['season'] == season)]
                    
                    # Define the GAM model
                    gam0_DO_q50 = LinearGAM(s(0))
                    gam1_DO_q50 = LinearGAM(s(0))
                    
                    # Fit the model to the data
                    gam0_DO_q50.fit(plot_df_DO_q50[plot_df_DO_q50['year']<=1988]['datetime'], plot_df_DO_q50[plot_df_DO_q50['year']<=1988]['val'])
                    
                    x_pred0_DO_q50 = plot_df_DO_q50[plot_df_DO_q50['year']<=1988]['datetime']

                    y_pred0_DO_q50 = gam0_DO_q50.predict(x_pred0_DO_q50)
                    
                    # Fit the model to the data
                    gam1_DO_q50.fit(plot_df_DO_q50[plot_df_DO_q50['year']>1988]['datetime'], plot_df_DO_q50[plot_df_DO_q50['year']>1988]['val'])
                    
                    x_pred1_DO_q50 = plot_df_DO_q50[plot_df_DO_q50['year']>1988]['datetime']

                    y_pred1_DO_q50 = gam1_DO_q50.predict(x_pred1_DO_q50)
                                                       
                    if site == 'point_jefferson':
                        
                        ax.axhspan(0,2, color = 'gray', alpha = 0.3, zorder=-5, label='hypoxia')
                                                
                        sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, color= 'gray', alpha=0.8, marker=marker, label='all casts')
                        
                        ax.plot(x_pred0, y_pred0, color='grey')
                        
                        ax.plot(x_pred1, y_pred1, color='grey')

                        sns.scatterplot(data=plot_df_DO_q50, x='datetime', y = 'val',  ax=ax,  color =palette[plot_df['site_type'].iloc[0]], marker=marker, label='[DO] <= median') #, label ='cast values <= 50th percentile')
                    
                        ax.plot(x_pred0_DO_q50, y_pred0_DO_q50, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])
                        
                        ax.plot(x_pred1_DO_q50, y_pred1_DO_q50, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])


                        ax.legend(loc='lower right')
                        
                    
                    elif site in ['carr_inlet_mid']:
                        
                        ax.axhspan(0,2, color = 'gray', alpha = 0.3, zorder=-5)
                                                
                        sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, color= 'gray', alpha=0.8, marker=marker, legend=False)
                        
                        ax.plot(x_pred0, y_pred0, color='gray')
                        
                        ax.plot(x_pred1, y_pred1, color='gray')
                        
                        sns.scatterplot(data=plot_df_DO_q50, x='datetime', y = 'val',  ax=ax,  color =palette[plot_df['site_type'].iloc[0]], marker=marker, label='[DO] <= median') #, label ='cast values <= 50th percentile')

                        ax.plot(x_pred0_DO_q50, y_pred0_DO_q50, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])
                        
                        ax.plot(x_pred1_DO_q50, y_pred1_DO_q50, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])


                        ax.legend(loc='lower right')


                    else:
                        
                        ax.axhspan(0,2, color = 'gray', alpha = 0.3, zorder=-5)
                                                
                        sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, color= 'gray', alpha=0.8, marker=marker, legend=False)
                        
                        ax.plot(x_pred0, y_pred0, color='gray')
                        
                        ax.plot(x_pred1, y_pred1, color='gray')
                        
                        sns.scatterplot(data=plot_df_DO_q50, x='datetime', y = 'val',  ax=ax,  color =palette[plot_df['site_type'].iloc[0]], marker=marker, legend=False) #, label ='cast values <= 50th percentile')

                        ax.plot(x_pred0_DO_q50, y_pred0_DO_q50, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])
                        
                        ax.plot(x_pred1_DO_q50, y_pred1_DO_q50, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])
                    
                    
                elif var == 'CT':
                    
                    if site in ['point_jefferson', 'carr_inlet_mid']:

                    
                        sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, hue='site_type', palette = palette, marker=marker)
                                                
                        ax.plot(x_pred0, y_pred0, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])
                        
                        ax.plot(x_pred1, y_pred1, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])
                                        
                        ax.legend(ncol=2, loc='upper left')
                        
                    else:
                        
                        sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, hue='site_type', palette = palette, marker=marker, legend=False)
                        
                        ax.plot(x_pred0, y_pred0, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])
                        
                        ax.plot(x_pred1, y_pred1, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])

                        
                else:
                    
                    sns.scatterplot(data=plot_df, x='datetime', y = 'val',  ax=ax, hue='site_type', palette = palette, marker=marker, legend=False)
                    
                    ax.plot(x_pred0, y_pred0, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])
                    
                    ax.plot(x_pred1, y_pred1, color = 'k') #color=palette[plot_df['site_type'].iloc[0]])
    
                
                #ax.scatter(x=0, y =0, color = palette[site], marker='o', label = site_label)
                    
                        
                    
                # elif var == 'CT':
                    
                #     ax.legend(ncol=2, loc='upper left')
                    
                #     ax.text(0.025,0.05, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
                    
                # else:
                    
                
                if plot_labels[c] in ['a','d','g','j','m']:
                
                    ax.text(0.05,0.05, plot_labels[c] + ' ' + plot_df['site_label'].iloc[0] + ' ->', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
                    
                else:
                    
                    ax.text(0.05,0.05, plot_labels[c], transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')
                
                            
                ax.set_ylim(ymin, ymax)
                
                #if site == 'point_jefferson':
                                    
                ax.set_ylabel(label_var + ' ' + unit)
                    
                # else:
                    
                #     ax.set_ylabel('')
                
                ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
                
                custom_ticks = [
                    datetime.datetime(1930, 1, 1),
                    datetime.datetime(1940, 1, 1),
                    datetime.datetime(1950, 1, 1),
                    datetime.datetime(1960, 1, 1),
                    datetime.datetime(1970, 1, 1),
                    datetime.datetime(1980, 1, 1),
                    datetime.datetime(1990, 1, 1),
                    datetime.datetime(2000, 1, 1),
                    datetime.datetime(2010, 1, 1),
                    datetime.datetime(2020, 1, 1),
                    datetime.datetime(2030, 1, 1),
                ]
                
                ax.set_xticks(custom_ticks)
                
                ax.set_xticklabels(['1930','','','1960','','','1990','','','2020', ''])

                        
                ax.set_xlabel('')
            
        c+=1
                    
                    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_6-GAM_testb.png', bbox_inches='tight', dpi=500, transparent=True)            
            