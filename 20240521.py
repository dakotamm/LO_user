#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:01:40 2024

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


# %%

Ldir = Lfun.Lstart(gridname='cas7')

# %%

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


# %%

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_north', 'saratoga_passage_mid', 'point_jefferson']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'DO_mg_L'] #'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']

# %%

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                      .assign(
                          datetime=(lambda x: pd.to_datetime(x['time'], utc=True)),
                          year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                          month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                          season=(lambda x: pd.cut(x['month'],
                                                  bins=[0,3,6,9,12],
                                                  labels=['winter', 'spring', 'summer', 'fall'])),
                          DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                          date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())),
                          segment=(lambda x: key),
                          decade=(lambda x: pd.cut(x['year'],
                                                  bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                                                  labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
                              )
                      )
    
    for var in var_list:
        
        if var not in odf_dict[key].columns:
            
            odf_dict[key][var] = np.nan
            
    odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade', 'name'],
                                          value_vars=var_list, var_name='var', value_name = 'val')

# %%

odf = pd.concat(odf_dict.values(), ignore_index=True)

# %%

odf['source_type'] = odf['source'] + '_' + odf['otype']

# %%

odf = odf.dropna()

# %%

odf = odf.assign(
    ix=(lambda x: x['lon'].apply(lambda x: zfun.find_nearest_ind(lon_1D, x))),
    iy=(lambda x: x['lat'].apply(lambda x: zfun.find_nearest_ind(lat_1D, x)))
)

# %%

odf['h'] = odf.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)

# %%

odf['yearday'] = odf['datetime'].dt.dayofyear

# %%

odf = odf[odf['val'] >0]

# %%

max_depths_dict = dict()

ox = lon
oy = lat
oxoy = np.concatenate((ox.reshape(-1,1),oy.reshape(-1,1)), axis=1)


for poly in poly_list:

    fnp = Ldir['LOo'] / 'section_lines' / (poly+'.p')
    p = pd.read_pickle(fnp)
    xx = p.x.to_numpy()
    yy = p.y.to_numpy()
    xxyy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
    path = mpth.Path(xxyy)
    
    oisin = path.contains_points(oxoy)
    
    this_depths = depths.flatten()[oisin]
    
    max_depth = np.nanmax(this_depths)
    
    max_depths_dict[poly] = max_depth.copy()
    
# %%
# %%
# %%
# DECADAL

odf_decadal = odf.copy()


for basin in basin_list:
    
    odf_decadal.loc[odf['segment'] == basin, 'min_segment_h'] = -max_depths_dict[basin]

# %%

summer_mask = (odf_decadal['yearday'] > 125) & (odf_decadal['yearday']<= 325)

surf_mask = (odf_decadal['z'] > -5)

deep_non_lc_nso_mask = (odf_decadal['z'] < 0.8*odf_decadal['min_segment_h']) & (odf_decadal['segment'] != 'lynch_cove_mid') & (odf_decadal['segment'] != 'near_seattle_offshore')

deep_lc_mask = (odf_decadal['z'] < 0.4*odf_decadal['min_segment_h']) & (odf_decadal['segment'] == 'lynch_cove_mid')

deep_nso_mask = (odf_decadal['z'] < 0.75*odf_decadal['min_segment_h']) & (odf_decadal['segment'] == 'near_seattle_offshore') #CHANGED 5/21/2024


# %%

odf_decadal.loc[summer_mask, 'summer_non_summer'] = 'summer'

odf_decadal.loc[~summer_mask, 'summer_non_summer'] = 'non_summer'

odf_decadal.loc[surf_mask, 'surf_deep'] = 'surf'

odf_decadal.loc[deep_non_lc_nso_mask, 'surf_deep'] = 'deep'

odf_decadal.loc[deep_lc_mask, 'surf_deep'] = 'deep'

odf_decadal.loc[deep_nso_mask, 'surf_deep'] = 'deep'

# %%

temp0 = odf_decadal[odf_decadal['surf_deep'] != 'nan']

# %%

odf_decadal_depth_mean = temp0.groupby(['segment','surf_deep', 'summer_non_summer', 'year', 'var','cid']).mean(numeric_only=True).reset_index().dropna()

# %%

annual_counts = (odf_decadal_depth_mean
                      .dropna()
                      #.set_index('datetime')
                      .groupby(['segment','year','summer_non_summer', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )

# %%

odf_decadal_use = odf_decadal_depth_mean.groupby(['segment', 'surf_deep', 'summer_non_summer', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})

# %%

odf_decadal_use.columns = odf_decadal_use.columns.to_flat_index().map('_'.join)

odf_decadal_use = odf_decadal_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# %%

odf_decadal_use = (odf_decadal_use
                  # .drop(columns=['date_ordinal_std'])
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .reset_index() 
                  .dropna()
                  .assign(
                          #segment=(lambda x: key),
                          # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                          # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                          # season=(lambda x: pd.cut(x['month'],
                          #                          bins=[0,3,6,9,12],
                          #                          labels=['winter', 'spring', 'summer', 'fall'])),
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

# %%

odf_decadal_use = pd.merge(odf_decadal_use, annual_counts, how='left', on=['segment','surf_deep','summer_non_summer','year','var'])

# %%

odf_decadal_use = odf_decadal_use[odf_decadal_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_decadal_use['val_ci95hi'] = odf_decadal_use['val_mean'] + 1.96*odf_decadal_use['val_std']/np.sqrt(odf_decadal_use['cid_count'])

odf_decadal_use['val_ci95lo'] = odf_decadal_use['val_mean'] - 1.96*odf_decadal_use['val_std']/np.sqrt(odf_decadal_use['cid_count'])

# %%

mk_bool_decadal = {}

mk_p_decadal = {}

for basin in basin_list:
    
    mk_bool_decadal[basin] = {}
    
    mk_p_decadal[basin] = {}

    
    for var in var_list:
        
        mk_bool_decadal[basin][var] = {}
        
        mk_p_decadal[basin][var] = {}
        
        for season in ['summer', 'non_summer']:
            
            mk_bool_decadal[basin][var][season] = {}

            mk_p_decadal[basin][var][season] = {}
        
            for depth in ['surf', 'deep']:
                
                mk_df = odf_decadal_use[(odf_decadal_use['segment'] == basin) & (odf_decadal_use['var'] == var) & (odf_decadal_use['summer_non_summer'] == season) & (odf_decadal_use['surf_deep'] == depth)]
                
                mk_df = mk_df.sort_values(by='datetime')
                
                reject_null, p_value, Z = dfun.mann_kendall(mk_df['val_mean'])
                
                mk_bool_decadal[basin][var][season][depth] = reject_null
                
                mk_p_decadal[basin][var][season][depth] = p_value
                
                print(basin + ' ' + var + ' ' + season + ' ' + depth + ': ' + str(reject_null) + ' ' + str(np.round(p_value,decimals=4)))
                
# %%

for basin in basin_list:       
    
    for var in var_list:
        
        for season in ['summer', 'non_summer']:
            
            for depth in ['surf', 'deep']:
                
                mask = (odf_decadal_depth_mean['segment'] == basin) &  (odf_decadal_depth_mean['var'] == var) & (odf_decadal_depth_mean['summer_non_summer'] == season) & (odf_decadal_depth_mean['surf_deep'] == depth)

                odf_decadal_depth_mean.loc[mask, 'mk_bool'] = mk_bool_decadal[basin][var][season][depth]
                
                odf_decadal_depth_mean.loc[mask, 'mk_p'] = mk_p_decadal[basin][var][season][depth]
                
# %%

# for var in var_list:
        
#     if var =='SA':
                
#         marker = 's'
        
#         ymin = 15
        
#         ymax = 35
        
#         label = 'Salinity [PSU]'
                    
#         color_deep = 'blue'
        
#         color_surf = 'lightblue'
                
#     elif var == 'CT':
        
#         marker = '^'
        
#         ymin = 4
        
#         ymax = 22
        
#         label = 'Temperature [deg C]'
                    
#         color_deep = 'red'
        
#         color_surf = 'pink'
        
#     else:
        
#         marker = 'o'
        
#         ymin = 0
        
#         ymax = 15
        
#         color = 'black'
        
#         label = 'DO [mg/L]'
        
#         color_deep = 'black'
        
#         color_surf = 'gray'
        
#     colors = {'deep':color_deep, 'surf':color_surf}
    
    
    

#     for big_basin in ['mb','wb','hc','ss']:
        
#         if big_basin == 'mb':
            
#             sites = ['point_jefferson', 'near_seattle_offshore']
            
#         elif big_basin == 'wb':
            
#             sites = ['saratoga_passage_north', 'saratoga_passage_mid']
            
#         elif big_basin == 'hc':
            
#             sites = ['lynch_cove_mid']
            
#         elif big_basin == 'ss':
            
#             sites = ['carr_inlet_mid']
            
            
            
#         mosaic = []
        
#         c=0
    
#         for site in sites:
            
#             if c==0:
            
#                 new_list = ['map']
                
#             else:
                
#                 new_list = ['.']
            
#             for season in ['non_summer', 'summer']:
                                
#                 new_list.append(site + '_' + season)
                                
#             mosaic.append(new_list)
            
#             c+=1
            
            
#         fig_height = len(sites)*5
        
                    
#         fig, ax = plt.subplot_mosaic(mosaic, layout='constrained', figsize = (15,fig_height))
        
#         plot_df_map = odf_decadal[(odf_decadal['segment'].isin(sites))].groupby('cid').first().reset_index()

#         sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='segment', palette='Set2', ax = ax['map'], s = 100, alpha=0.5)

#         #sns.scatterplot(data=plot_df_mean, x='lon', y='lat', hue='decade', palette='Set2', marker='s', sizes=20)

#         ax['map'].autoscale(enable=False)

#         pfun.add_coast(ax['map'])

#         pfun.dar(ax['map'])

#         ax['map'].set_xlim(-123.2, -122.1)

#         ax['map'].set_ylim(47,48.5)

#         # ax['map'].set_title(season + ' sampling locations')

#         ax['map'].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
        
        
#         for site in sites:
            
            
#             for season in ['non_summer', 'summer']:
                
#                 ax_name = site + '_' + season
                                                
#                 plot_df = odf_decadal_use[(odf_decadal_use['segment'] == site) & (odf_decadal_use['var'] == var) & (odf_decadal_use['summer_non_summer'] == season)]
                        
#                 sns.scatterplot(data=plot_df, x='datetime', y ='val_mean', hue = 'surf_deep', palette=colors, ax=ax[ax_name], alpha=0.7, legend = False)
                            
#                 for idx in plot_df.index:
                    
#                     ax[ax_name].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
                                    
#                 ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
                
#                 ax[ax_name].set_ylabel(label)
                
#                 ax[ax_name].set_ylim(ymin,ymax)
                
#                 ax[ax_name].set_title(site + ' ' + season)
            
#                 if var == 'DO_mg_L':
                    
#                     ax[ax_name].axhspan(0,2, color = 'lightgray', alpha = 0.2)
                    
#                 ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2030,12,31)])
                
#                 ax[ax_name].set_xlabel('Year')
                
#                 ax[ax_name].text(0, 1, 'max depth = ' + str(np.round(max_depths_dict[site])), horizontalalignment='left', verticalalignment='top', transform=ax[ax_name].transAxes)
                                    
            
#                 for depth in ['surf', 'deep']:
                    
#                     reject_null = mk_bool_decadal[site][var][season][depth]
                    
#                     p_value = mk_p_decadal[site][var][season][depth]
                    
#                     if reject_null == True:
                        
#                         color = 'm'
                        
#                     else:
#                         color = 'k'
                                    
#                     if depth == 'surf':
                        
#                         ax[ax_name].text(1,1, depth + ' MK: ' + str(reject_null) + ' ' + str(np.round(p_value, 3)), horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=color)
                    
#                     elif depth == 'deep':
                        
#                         ax[ax_name].text(1,0.9, depth + ' MK: ' + str(reject_null) + ' ' + str(np.round(p_value, 3)), horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=color)

                        
#         fig.suptitle(big_basin)
                                                    
#         plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + big_basin + '_' + var + '_surf_deep_decadal.png', bbox_inches='tight', dpi=500)
        
# %%

site = 'point_jefferson'

mosaic = []

c=0

for var in var_list:
    
    new_list = []
    
    for season in ['non_summer','summer']:
        
        new_list.append(var + '_' + season)
                        
    mosaic.append(new_list)
    
    c+=1
    
fig, ax = plt.subplot_mosaic(mosaic, layout='constrained', figsize = (12,8))
        

for var in var_list:
        
    if var =='SA':
                
        marker = 's'
        
        ymin = 15
        
        ymax = 35
        
        label = 'Salinity [PSU]'
                    
        color_deep = 'blue'
        
        color_surf = 'lightblue'
                
    elif var == 'CT':
        
        marker = '^'
        
        ymin = 4
        
        ymax = 22
        
        label = 'Temperature [deg C]'
                    
        color_deep = 'red'
        
        color_surf = 'pink'
        
    else:
        
        marker = 'o'
        
        ymin = 0
        
        ymax = 15
        
        color = 'black'
        
        label = 'DO [mg/L]'
        
        color_deep = 'black'
        
        color_surf = 'gray'
        
    colors = {'deep':color_deep, 'surf':color_surf}
    
    
    
    for season in ['non_summer', 'summer']:
        
        ax_name = var + '_' + season
                                        
        plot_df = odf_decadal_use[(odf_decadal_use['segment'] == site) & (odf_decadal_use['var'] == var) & (odf_decadal_use['summer_non_summer'] == season)]
                
        sns.scatterplot(data=plot_df, x='datetime', y ='val_mean', hue = 'surf_deep', palette=colors, ax=ax[ax_name], alpha=0.7, legend = False)
                    
        for idx in plot_df.index:
            
            ax[ax_name].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
                            
        ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax[ax_name].set_ylabel(label)
        
        ax[ax_name].set_ylim(ymin,ymax)
        
        #ax[ax_name].set_title(site + ' ' + season)
    
        if var == 'DO_mg_L':
            
            ax[ax_name].axhspan(0,2, color = 'lightgray', alpha = 0.2)
            
        ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2030,12,31)])
        
        ax[ax_name].set_xlabel('Year')
        
        ax[ax_name].text(0, 1, 'max depth = ' + str(np.round(max_depths_dict[site])), horizontalalignment='left', verticalalignment='top', transform=ax[ax_name].transAxes)
                            
    
        for depth in ['surf', 'deep']:
            
            reject_null = mk_bool_decadal[site][var][season][depth]
            
            p_value = mk_p_decadal[site][var][season][depth]
            
            if reject_null == True:
                
                color = 'm'
                
            else:
                color = 'k'
                            
            if depth == 'surf':
                
                ax[ax_name].text(1,1, depth + ' MK: ' + str(reject_null) + ' ' + str(np.round(p_value, 3)), horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=color)
            
            elif depth == 'deep':
                
                ax[ax_name].text(1,0.9, depth + ' MK: ' + str(reject_null) + ' ' + str(np.round(p_value, 3)), horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=color)


ax['SA_non_summer'].set_title('Non-Summer')

ax['SA_summer'].set_title('Summer')


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_all_surf_deep_decadal.png', bbox_inches='tight', dpi=500)

