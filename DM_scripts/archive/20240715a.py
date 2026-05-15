#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:11:45 2024

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

poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_nc', 'ecology_his', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO_mg_L', 'DO (uM)']

# %%

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                     .assign(
                         datetime=(lambda x: pd.to_datetime(x['time'])),
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

#odf = odf[odf['var'].isin(['DO_mg_L', 'CT', 'SA'])]

#odf = odf[(odf['val'] >= 0) & (odf['val'] <50)]
    
# %%

#lc_exclude = odf[(odf['segment'] == 'lynch_cove_shallow') & (odf['z'] < -45)]

# %%

#odf = odf[~odf['cid'].isin(lc_exclude['cid'].unique())]


# %%

odf['source_type'] = odf['source'] + '_' + odf['otype']

# %%

for decade in odf['decade'].unique():
    
    for season in ['winter','spring','summer','fall']:
        
        for basin in basin_list:
            
            for var in var_list:
            
                fig, (ax, axx)  = plt.subplots(ncols = 2, figsize = (20,20))
                
                plt.rc('font', size=14)
                
                plot_df = odf[(odf['segment'] == basin) & (odf['season'] == season) & (odf['decade'] == decade) & (odf['var'] == var)].dropna(subset=['val'])
                
                if not plot_df.empty:
                
                    plot_df_map = plot_df.groupby('cid').first().reset_index()
                    
                    sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='source_type', hue_order = ['collias_bottle', 'ecology_his_bottle', 'ecology_his_ctd', 'ecology_nc_ctd', 'ecology_nc_bottle', 'kc_bottle', 'kc_whidbey_ctd', 'kc_taylor_bottle', 'nceiSalish_bottle', 'kc_ctd'], alpha = 0.5, ax = ax, legend = False)
                    
                    sns.scatterplot(data=plot_df, x='val', y='z', hue='source_type', hue_order = ['collias_bottle', 'ecology_his_bottle', 'ecology_his_ctd', 'ecology_nc_ctd', 'ecology_nc_bottle', 'kc_bottle', 'kc_whidbey_ctd', 'kc_taylor_bottle', 'nceiSalish_bottle', 'kc_ctd'], alpha=0.5, linewidth=0, ax = axx)
                    
                    ax.autoscale(enable=False)
                    
                    pfun.add_coast(ax)
                    
                    pfun.dar(ax)
                    
                    ax.set_xlim(-123.2, -122.1)
                    
                    ax.set_ylim(47,48.5)
                    
                    ax.set_title(decade + ' ' + season + ' ' + var.replace(' ', '_').replace('(','').replace(')',''))
                    
                    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                    
                    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/ps_' + decade + 's_' + season + '_' + var.replace(" ", "_").replace('(','').replace(')','') +'_raw.png', bbox_inches='tight', dpi=500)

# %%

# for year in odf['year'].unique():
    
#     for season in ['winter','spring','summer','fall']:

#         for basin in basin_list:
            
#             for var in var_list:
                
#                 fig, (ax, axx)  = plt.subplots(ncols = 2, figsize = (20,20))
                
#                 plt.rc('font', size=14)
                
#                 plot_df = odf[(odf['segment'] == basin) & (odf['season'] == season) & (odf['year'] == year) & (odf['var'] == var)].dropna(subset=['val'])
                
#                 if not plot_df.empty:
                
#                     plot_df_map = plot_df.groupby('cid').first().reset_index()
                    
#                     sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='source_type', hue_order = ['collias_bottle', 'ecology_bottle', 'ecology_ctd', 'kc_bottle', 'kc_whidbey_ctd', 'kc_taylor_bottle', 'nceiSalish_bottle', 'kc_ctd'], alpha = 0.5, ax = ax, legend = False)
                    
#                     sns.scatterplot(data=plot_df, x='val', y='z', hue='source_type', hue_order = ['collias_bottle', 'ecology_bottle', 'ecology_ctd', 'kc_bottle', 'kc_whidbey_ctd', 'kc_taylor_bottle', 'nceiSalish_bottle', 'kc_ctd'], alpha=0.5, linewidth=0, ax = axx)
                    
#                     ax.autoscale(enable=False)
                    
#                     pfun.add_coast(ax)
                    
#                     pfun.dar(ax)
                    
#                     ax.set_xlim(-123.2, -122.1)
                    
#                     ax.set_ylim(47,48.5)
                    
#                     ax.set_title(str(year) + ' ' + season + ' ' + var.replace(' ', '_').replace('(','').replace(')',''))
                    
#                     #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                                    
#                     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/ps_' + str(year) + '_' + season + '_' + var.replace(" ", "_").replace('(','').replace(')','') +'_raw.png', bbox_inches='tight', dpi=500)

