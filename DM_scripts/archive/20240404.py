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

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['kc_whidbey'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

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

lc_exclude = odf[(odf['segment'] == 'lynch_cove_shallow') & (odf['z'] < -45)]

# %%

odf = odf[~odf['cid'].isin(lc_exclude['cid'].unique())]

# %%

for year in odf['year'].unique():
    
    for season in ['winter','summer','spring','fall']:

        for basin in basin_list:
            
            mosaic = [['map', 'DO_mg_L', 'CT', 'SA'],
                      ['map', 'NO3 (uM)', 'Chl (mg m-3)', '']]
            
            fig, axd = plt.subplot_mosaic(mosaic, layout="constrained", figsize = (10,5)) 
            
            plt.rc('font', size=14)
            
            plot_df = odf[(odf['segment'] == basin) & (odf['season'] == season) & (odf['year'] == year)]
            
            plot_df_map = plot_df.groupby('cid').first().reset_index()
             
            #plot_df = odf_decade[(odf_decade['segment'] == basin) & (odf_decade['decade'] == decade) & (odf_decade['var'] == 'DO_mg_L')].groupby('cid').first().reset_index()
            
           # plot_df_mean = odf[(odf['segment'] == basin) & (odf['otype'] == 'bottle') & (odf['var'] == 'DO_mg_L')].groupby('decade').agg({'lat':'mean', 'lon':'mean'}).reset_index()
            
            sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='cid', palette='Set2', ax = axd['map'], legend=False) #, alpha=0.5)
            
            #sns.scatterplot(data=plot_df_mean, x='lon', y='lat', hue='decade', palette='Set2', marker='s', sizes=20)
            
            axd['map'].autoscale(enable=False)
            
            pfun.add_coast(axd['map'])
            
            pfun.dar(axd['map'])
            
            axd['map'].set_xlim(-123.2, -122.1)
            
            axd['map'].set_ylim(47,48.5)
            
            axd['map'].set_title(str(year) + ' ' + season)
        
            #fig.tight_layout()
            
            
            sns.scatterplot(data=plot_df[plot_df['var'] == 'DO_mg_L'], x='val', y='z', hue='cid', palette='Set2', ax = axd['DO_mg_L'], legend=False) #, alpha=0.5)
            
            sns.scatterplot(data=plot_df[plot_df['var'] == 'CT'], x='val', y='z', hue='cid', palette='Set2', ax = axd['CT'], legend=False)
            
            sns.scatterplot(data=plot_df[plot_df['var'] == 'SA'], x='val', y='z', hue='cid', palette='Set2', ax = axd['SA'], legend=False)
            
            sns.scatterplot(data=plot_df[plot_df['var'] == 'NO3 (uM)'], x='val', y='z', hue='cid', palette='Set2', ax = axd['NO3 (uM)'], legend=False)
            
            sns.scatterplot(data=plot_df[plot_df['var'] == 'Chl (mg m-3)'], x='val', y='z', hue='cid', palette='Set2', ax = axd['Chl (mg m-3)'], legend=False)



            
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/kc_whidbey_ctd_data_' + str(year) + '_' + season +'.png', bbox_inches='tight', dpi=500)
            
# %%


