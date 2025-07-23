#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:35:24 2025

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


poly_list = ['point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L', 'CT','SA'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)


odf_use = odf_depth_mean.copy()


# %%

odf_use = (odf_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )

# %%

odf_use_filt = odf_use[(odf_use['surf_deep'] == 'deep') & (odf_use['var'] == 'DO_mg_L')]

    

# %%



ts_lines_df = pd.DataFrame()

for season in odf_use_filt['season'].unique():
    
    temp_df = odf_use_filt[odf_use_filt['season'] == season]
    
    x = temp_df['date_ordinal']
    
    x_dt = temp_df['datetime']
    
    y = temp_df['val']
    
    result = stats.theilslopes(y,x,alpha=0.05)
    B1 = result.slope
    B0 = result.intercept
    
    high_sB1 = result.high_slope
    low_sB1 = result.low_slope
    
    
    #x_line = np.linspace(x.min(), x.max(), 100)
    
    
    y_line = B0 + B1*x
    
    middle_y = (y_line.max() - y_line.min())/2 + y_line.min()
        
    middle_x = (x.max() - x.min())/2 + x.min()
    
    y_hi_line = high_sB1*(x - middle_x) + middle_y
    
    y_lo_line = low_sB1*(x - middle_x) + middle_y
    
    y_zero_line = (y_line.max() - y_line.min())/2 + y_line.min() + 0*y_line
    
    line_df = pd.DataFrame({'date_ordinal': x, 'y_line': y_line, 'y_lo_line': y_lo_line, 'y_hi_line': y_hi_line, 'y_zero_line': y_zero_line})
    
    line_df['season'] = season
    
    line_df['slope_datetime'] = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_dt.max().year - x_dt.min().year)

    line_df['slope_datetime_s_hi'] = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_dt.max().year - x_dt.min().year)
    line_df['slope_datetime_s_lo'] = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_dt.max().year - x_dt.min().year)

    
    ts_lines_df = pd.concat([ts_lines_df, line_df])
    
    
    
ts_lines_df = (ts_lines_df
              .dropna()
              .assign(
                      datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                      )
              )

    
ts_lines_df['95CI_hi'] = ts_lines_df['slope_datetime_s_hi'] - ts_lines_df['slope_datetime'] 

ts_lines_df['95CI_lo'] = ts_lines_df['slope_datetime'] - ts_lines_df['slope_datetime_s_lo']

ts_lines_df['slope_datetime_cent'] = ts_lines_df['slope_datetime']*100

ts_lines_df['slope_datetime_s_hi_cent'] = ts_lines_df['slope_datetime_s_hi']*100

ts_lines_df['slope_datetime_s_lo_cent'] = ts_lines_df['slope_datetime_s_lo']*100

ts_lines_df['95CI_hi_cent'] = ts_lines_df['95CI_hi']*100

ts_lines_df['95CI_lo_cent'] = ts_lines_df['95CI_lo']*100


ts_lines_df['slope_datetime_cent_str'] = ts_lines_df['slope_datetime_cent'].apply(lambda x: f"{x:.2f}")

ts_lines_df['95CI_hi_cent_str'] = ts_lines_df['95CI_hi_cent'].apply(lambda x: f"{x:.2f}")

ts_lines_df['95CI_lo_cent_str'] = ts_lines_df['95CI_lo_cent'].apply(lambda x: f"{x:.2f}")



ts_lines_df['slope_label'] = ts_lines_df['slope_datetime_cent_str'] + ' [mg/L]/century +' + ts_lines_df['95CI_hi_cent_str'] + '/-' + ts_lines_df['95CI_lo_cent_str']


# %%

pj_deep = pd.merge(odf_use_filt, ts_lines_df, how='left', on = ['date_ordinal', 'datetime', 'season'])

# %%


pj_deep.loc[pj_deep['season'] == 'loDO', 'Season'] = 'Aug-Nov [Low DO]'

pj_deep.loc[pj_deep['season'] == 'grow', 'Season'] = 'Apr-Jul [Spring Bloom]'

pj_deep.loc[pj_deep['season'] == 'winter', 'Season'] = 'Dec-Mar [Winter]'


pj_deep['val_label'] = pj_deep['val'].apply(lambda x: f"{x:.2f}")

# %%

pj_deep.to_json('/Users/dakotamascarenas/Desktop/pj_deep_slopes.json', orient='records', date_format = 'iso')

# %%

fig,ax = plt.subplots()
 
ax.plot(ts_lines_df[ts_lines_df['season'] == 'Apr-Jul [Spring Bloom]']['datetime'], ts_lines_df[ts_lines_df['season'] == 'Apr-Jul [Spring Bloom]']['y_line'])

plt.savefig('/Users/dakotamascarenas/Desktop/tset_yline.png', bbox_inches='tight', dpi=500, transparent=False)

