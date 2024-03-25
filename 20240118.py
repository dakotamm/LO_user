"""
Created on Thu Jan 18 13:56:22 2024

@author: dakotamascarenas

Creating a per-decade map of sampling data. Focus first on bottles, then maybe move to CTDs...

***

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

poly_list = ['sog_n', 'sog_s_wo_si', 'si', 'soj','sji','mb', 'wb', 'hc_wo_lc', 'lc', 'ss']

odf_dict = dfun.getPolyData(Ldir, poly_list)

with open('/Users/dakotamascarenas/Desktop/big_dict.pkl', 'wb') as f:
    pickle.dump(odf_dict, f)
    
    # THIS IS CURRENTLY SET TO INCLUDE ECOLOGY BOTTLE DO!!!


# %%


# with open('/Users/dakotamascarenas/Desktop/big_dict.pkl', 'rb') as f:
#     odf_dict = pickle.load(f)


# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO_mg_L', 'DO (uM)']


# %%

# manage data without period labels or depth labels - not the cleanest way to do that

# also make data "long" off the bat

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
                         decade=(lambda x: ((x['year']/10).apply(np.floor)*10).astype('int64'))
                             )
                     )
    
    for var in var_list:
        
        if var not in odf_dict[key].columns:
            
            odf_dict[key][var] = np.nan
            
    odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade'],
                                         value_vars=var_list, var_name='var', value_name = 'val')
    
# %%

# make df??? not until after counts...honestly doesn't really matter i guess...

odf = pd.concat(odf_dict.values(), ignore_index=True)
    
# %%

# plot 1 - make maps of decadal sampling locations

for basin in basin_list:
    
    fig, ax = plt.subplots(figsize = (10,10))
    
    plt.rc('font', size=14)
    
    plot_df = odf[(odf['segment'] == basin) & (odf['otype'] == 'bottle') & (odf['var'] == 'DO_mg_L')].groupby('cid').first().reset_index()
    
    plot_df_mean = odf[(odf['segment'] == basin) & (odf['otype'] == 'bottle') & (odf['var'] == 'DO_mg_L')].groupby('decade').agg({'lat':'mean', 'lon':'mean'}).reset_index()
    
    sns.scatterplot(data=plot_df, x='lon', y='lat', hue='decade', palette='Set2')
    
    #sns.scatterplot(data=plot_df_mean, x='lon', y='lat', hue='decade', palette='Set2', marker='s', sizes=20)
    
    ax.autoscale(enable=False)
    
    pfun.add_coast(ax)
    
    pfun.dar(ax)
    
    ax.set_title(basin + ' bottle DO sampling locations')

    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_bottle_DO_sampling_locations_by_decade.png', dpi=500)
    
    # for decade in plot_df['decade'].unique():
        
    #     fig, ax = plt.subplots(figsize = (10,10))
        
    #     plt.rc('font', size=14)
        
    #     plot_df_decade = plot_df[plot_df['decade'] == decade]
        
    #     plot_df_mean_decade = plot_df_mean[plot_df_mean['decade'] == decade]
        
    #     sns.scatterplot(data=plot_df_decade, x='lon', y='lat', hue='decade', palette='Set2', alpha=0.7)
        
    #     #sns.scatterplot(data=plot_df_mean_decade, x='lon', y='lat', hue='decade', palette='Set2', marker='s', sizes=20)
        
    #     ax.autoscale(enable=False)
        
    #     pfun.add_coast(ax)
        
    #     pfun.dar(ax)
        
    #     ax.set_title(basin + ' ' + str(decade) + 's bottle DO sampling locations')
        
    #     fig.tight_layout()
        
    #     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + str(decade) + 's_bottle_DO_sampling_locations_by_decade.png', dpi=500)
    
    
        











            