#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:50:07 2024

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

poly_list = ['carr_inlet_mid', 'hat_island', 'hood_canal_mouth', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_north', 'saratoga_passage_mid', 'point_jefferson']

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
            
    odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade'],
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


for basin in basin_list:
    
    odf.loc[odf['segment'] == basin, 'min_segment_h'] = -max_depths_dict[basin]

# %%

grow_mask = (odf['yearday'] > 125) & (odf['yearday']<= 325)

surf_mask = (odf['z'] > 0.1*odf['min_segment_h'])

deep_non_lc_mask = (odf['z'] < 0.8*odf['min_segment_h']) & (odf['segment'] != 'lynch_cove_mid')

deep_lc_mask = (odf['z'] < 0.4*odf['min_segment_h']) & (odf['segment'] == 'lynch_cove_mid')

# %%

odf.loc[grow_mask, 'grow_no_grow'] = 'grow'

odf.loc[~grow_mask, 'grow_no_grow'] = 'no_grow'

odf.loc[surf_mask, 'surf_deep'] = 'surf'

odf.loc[deep_non_lc_mask, 'surf_deep'] = 'deep'

odf.loc[deep_lc_mask, 'surf_deep'] = 'deep'

# %%

temp0 = odf[odf['surf_deep'] != 'nan']

# %%

temp1 = temp0.groupby(['segment','surf_deep', 'grow_no_grow', 'year', 'var','cid']).mean(numeric_only=True).reset_index().dropna()

# %%

annual_counts = (temp1
                      .dropna()
                      #.set_index('datetime')
                      .groupby(['segment','year','grow_no_grow', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )

# %%

odf_use = temp1.groupby(['segment', 'surf_deep', 'grow_no_grow', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})

# %%

odf_use.columns = odf_use.columns.to_flat_index().map('_'.join)

odf_use = odf_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# %%

odf_use = (odf_use
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

odf_use = pd.merge(odf_use, annual_counts, how='left', on=['segment','surf_deep','grow_no_grow','year','var'])

# %%

odf_use = odf_use[odf_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_use['val_ci95hi'] = odf_use['val_mean'] + 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])

odf_use['val_ci95lo'] = odf_use['val_mean'] - 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])

# %%

mosaic =[['SA_no_grow', 'SA_grow'], ['CT_no_grow', 'CT_grow'], ['DO_mg_L_no_grow', 'DO_mg_L_grow']]


for basin in basin_list:
    
    fig, ax = plt.subplot_mosaic(mosaic, figsize = (10,10), layout='constrained')
    
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
            
            ymin = 7
            
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
            
        
        for gng in ['grow', 'no_grow']:
            
            ax_name = var + '_' + gng
            
            plot_df = odf_use[(odf_use['var'] == var) & (odf_use['segment'] == basin) & (odf_use['grow_no_grow'] == gng)]
                    
            sns.scatterplot(data=plot_df, x='datetime', y ='val_mean', hue = 'surf_deep', palette=colors, ax=ax[ax_name], alpha=0.7)
                        
            for idx in plot_df.index:
                
                ax[ax_name].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
                                
            ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[ax_name].set_ylabel(label)
            
            ax[ax_name].set_ylim(ymin,ymax)
            
            ax[ax_name].set_title(gng)
        
            if var == 'DO_mg_L':
                
                ax[ax_name].axhspan(0,2, color = 'lightgray', alpha = 0.2)
                
            ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2030,12,31)])
            
            ax[ax_name].set_xlabel('Year')
                            
    fig.suptitle(basin)
                                                
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_surf_deep_decadal.png', bbox_inches='tight', dpi=500)
            
            
            
            
            
            
    

# %%

# odf_grow = odf[(odf['yearday'] > 200) & (odf['yearday'] <= 300)]

# # %%

# odf_grow_bottom = odf_grow[(odf_grow['z'] < 0.8*odf_grow['min_segment_h']) & (~odf_grow['segment'].isin(['lynch_cove_mid', 'elliott_bay', 'off_port_madison', 'near_alki', 'port_susan_mid', 'ps', 'wb', 'mb', 'ss', 'hc']))]

# # %%

# odf_grow_lynch_cove_mid = odf_grow[(odf_grow['z'] < 0.4*odf_grow['min_segment_h']) & (odf_grow['segment'] == 'lynch_cove_mid')]  # bottom 50%

# # %%

# #odf_grow_elliott_bay= odf_grow[(odf_grow['z'] < 0.4*odf_grow['min_segment_h']) & (odf_grow['segment'] == 'elliott_bay')]  # bottom 50%


# # %%

# odf_use = pd.concat([odf_grow_bottom, odf_grow_lynch_cove_mid], ignore_index=True)

# # %%

# temp0 = odf_use.groupby(['segment','decade','year','var','cid']).mean(numeric_only=True).reset_index().dropna()

# # %%

# annual_counts = (temp0
#                      .dropna()
#                      #.set_index('datetime')
#                      .groupby(['segment','year','var']).agg({'cid' :lambda x: x.nunique()})
#                      .reset_index()
#                      .rename(columns={'cid':'cid_count'})
#                      )

# # %%

# odf_grow_bottom_lc = temp0.groupby(['segment', 'decade', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})

# # HOOD CANAL WHOLE CAST VALUE?!?!?! what it is right now

# # %%

# odf_grow_bottom_lc.columns = odf_grow_bottom_lc.columns.to_flat_index().map('_'.join)

# odf_grow_bottom_lc = odf_grow_bottom_lc.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# # %%

# odf_grow_bottom_lc = (odf_grow_bottom_lc
#                   # .drop(columns=['date_ordinal_std'])
#                   .rename(columns={'date_ordinal_mean':'date_ordinal'})
#                   .reset_index() 
#                   .dropna()
#                   .assign(
#                           #segment=(lambda x: key),
#                           # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
#                           # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
#                           # season=(lambda x: pd.cut(x['month'],
#                           #                          bins=[0,3,6,9,12],
#                           #                          labels=['winter', 'spring', 'summer', 'fall'])),
#                           datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
#                           )
#                   )

# # %%

# odf_grow_bottom_lc = pd.merge(odf_grow_bottom_lc, annual_counts, how='left', on=['segment','year','var'])

# # %%

# odf_grow_bottom_lc = odf_grow_bottom_lc[odf_grow_bottom_lc['cid_count'] >1] #redundant but fine (see note line 234)

# odf_grow_bottom_lc['val_ci95hi'] = odf_grow_bottom_lc['val_mean'] + 1.96*odf_grow_bottom_lc['val_std']/np.sqrt(odf_grow_bottom_lc['cid_count'])

# odf_grow_bottom_lc['val_ci95lo'] = odf_grow_bottom_lc['val_mean'] - 1.96*odf_grow_bottom_lc['val_std']/np.sqrt(odf_grow_bottom_lc['cid_count'])


# # %%
# ##########

# odf_grow_surface = odf_grow[(odf_grow['z'] > 0.1*odf_grow['min_segment_h'])] # & (~odf_grow['segment'].isin(['lynch_cove_mid', 'elliott_bay', 'off_port_madison', 'near_alki', 'ps', 'wb', 'mb', 'ss', 'hc']))]

# # %%

# temp1 = odf_grow_surface.groupby(['segment','decade','year','var','cid']).mean(numeric_only=True).reset_index().dropna()

# # %%

# odf_grow_surface = temp1.groupby(['segment', 'decade', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})

# # %%

# odf_grow_surface.columns = odf_grow_surface.columns.to_flat_index().map('_'.join)

# odf_grow_surface = odf_grow_surface.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# # %%

# odf_grow_surface = (odf_grow_surface
#                   # .drop(columns=['date_ordinal_std'])
#                   .rename(columns={'date_ordinal_mean':'date_ordinal'})
#                   .reset_index() 
#                   .dropna()
#                   .assign(
#                           #segment=(lambda x: key),
#                           # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
#                           # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
#                           # season=(lambda x: pd.cut(x['month'],
#                           #                          bins=[0,3,6,9,12],
#                           #                          labels=['winter', 'spring', 'summer', 'fall'])),
#                           datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
#                           )
#                   )

# # %%

# odf_grow_surface = pd.merge(odf_grow_surface, annual_counts, how='left', on=['segment','year','var'])

# # %%

# odf_grow_surface = odf_grow_surface[odf_grow_surface['cid_count'] >1] #redundant but fine (see note line 234)

# odf_grow_surface['val_ci95hi'] = odf_grow_surface['val_mean'] + 1.96*odf_grow_surface['val_std']/np.sqrt(odf_grow_surface['cid_count'])

# odf_grow_surface['val_ci95lo'] = odf_grow_surface['val_mean'] - 1.96*odf_grow_surface['val_std']/np.sqrt(odf_grow_surface['cid_count'])


# # %%

# for basin in odf_grow_surface['segment'].unique():
            
#     fig, ax = plt.subplots(figsize=(8,8), nrows=3, sharex=True)
            
#     plt.rc('font', size=14)
    
#     c=0
    
#     for var in var_list:
        
#         if var =='SA':
            
#             marker = 's'
            
#             ymin = 22
            
#             ymax = 32
            
#             label = 'Salinity [PSU]'
            
#             color = 'blue'
            
#             color_less = '#393070'
            
#             color_more = '#ABCF43'
                    
#         elif var == 'CT':
            
#             marker = '^'
            
#             ymin = 7
            
#             ymax = 17
            
#             label = 'Temperature [deg C]'
            
#             color = 'red'
            
#             color_less = '#466EA9'
            
#             color_more = '#9C3C49'
            
#         else:
            
#             marker = 'o'
            
#             ymin = 0
            
#             ymax = 12
            
#             color = 'black'
            
#             label = 'DO [mg/L]'
            
#             color_more = '#3E1F95'
            
#             color_less = '#EAB63A'

#         plot_df = odf_grow_surface[(odf_grow_surface['var'] == var) & (odf_grow_surface['segment'] == basin)]
        
#         plot_df_mean = plot_df.set_index('datetime').sort_index()
        
#        # rolling_mean = plot_df_mean['val_mean'].rolling(window='3650D', min_periods=1).mean()
        
#         sns.scatterplot(data=plot_df, x='datetime', y ='val_mean', ax=ax[c], color=color)
        
#         #rolling_mean.plot(label='Decadal Rolling Mean', ax=ax[c], color = color)
        
#         for idx in plot_df.index:
            
#             ax[c].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
                            
#         ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
#         ax[c].set_ylabel(label)
        
#         ax[c].set_ylim(ymin,ymax)
    
#         if var == 'DO_mg_L':
            
#             ax[c].axhspan(0,2, color = 'lightgray', alpha = 0.2)
            
#         ax[c].set_xlim([datetime.date(1930,1,1), datetime.date(2030,12,31)])
            
#         c+=1
        
#     ax[0].set_title(basin + ' surface grow season')
    
#     ax[-1].set_xlabel('Date')
                            
#     fig.tight_layout()
    
#     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_grow_surface_annual_mean.png', bbox_inches='tight', dpi=500)

# %%



