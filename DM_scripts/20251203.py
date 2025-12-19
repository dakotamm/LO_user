#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 10:52:53 2025

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

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)


# %%

site_list =  odf['site'].unique()

# %%

hyp_vals = odf_depth_mean[(odf_depth_mean['var'] == 'DO_mg_L') & (odf_depth_mean['val'] <= 2)]

# %%

num_hyp_vals_per_year = (hyp_vals
                         .groupby(['site','year', 'season', 'surf_deep']).agg({'yearday' :lambda x: x.nunique()})
                         .reset_index()
                         .rename(columns={'yearday':'num_hyp_days'})
                         )

# %%

fig, ax = plt.subplots(figsize=(8,4))

plot_df = num_hyp_vals_per_year[(num_hyp_vals_per_year['site'] == 'lynch_cove_mid') & (num_hyp_vals_per_year['surf_deep'] == 'deep')]

sns.scatterplot(data=plot_df, x='year', y='num_hyp_days', hue='season', hue_order=['grow', 'loDO', 'winter'], palette={'grow':'gold','loDO':'red','winter':'blue'}, ax=ax, legend=True)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/lc_num_hyp_days_per_year.png', bbox_inches='tight', dpi=500, transparent=True)

# %%

fig, ax = plt.subplots(figsize=(8,4))

plot_df = num_hyp_vals_per_year[(num_hyp_vals_per_year['site'] == 'lynch_cove_mid') & (num_hyp_vals_per_year['surf_deep'] == 'deep') & (num_hyp_vals_per_year['season'] == 'loDO')]

sns.scatterplot(data=plot_df, x='year', y='num_hyp_days', hue='season', hue_order=['grow', 'loDO', 'winter'], palette={'grow':'gold','loDO':'red','winter':'blue'}, ax=ax, legend=True)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/lc_loDO_num_hyp_days_per_year.png', bbox_inches='tight', dpi=500, transparent=True)

# %%

num_hyp_vals_per_year_full = (hyp_vals
                         .groupby(['site','year', 'surf_deep']).agg({'yearday' :lambda x: x.nunique()})
                         .reset_index()
                         .rename(columns={'yearday':'num_hyp_days'})
                         )


# %%

fig, ax = plt.subplots(figsize=(8,4))

plot_df = num_hyp_vals_per_year_full[(num_hyp_vals_per_year_full['site'] == 'lynch_cove_mid') & (num_hyp_vals_per_year_full['surf_deep'] == 'deep')]

sns.scatterplot(data=plot_df, x='year', y='num_hyp_days', ax=ax, legend=True)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/lc_num_hyp_days_per_year_full.png', bbox_inches='tight', dpi=500, transparent=True)

# %%

odf_depth_mean_dt = (odf_depth_mean
                      .dropna()
                      .assign(
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )

# %%

fig, ax = plt.subplots(figsize=(8,4))

plot_df = odf_depth_mean_dt[(odf_depth_mean_dt['site'] == 'lynch_cove_mid') & (odf_depth_mean_dt['var'] == 'DO_mg_L') & (odf_depth_mean_dt['surf_deep'] == 'deep')]

sns.scatterplot(data=plot_df, x='datetime', y='val')

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/lc_time_series_bottom_DO.png', bbox_inches='tight', dpi=500, transparent=True)

# %%

num_sampling_days_per_year = (odf_depth_mean
                         .groupby(['site','year', 'season','surf_deep']).agg({'yearday' :lambda x: x.nunique()})
                         .reset_index()
                         .rename(columns={'yearday':'num_sampling_days'})
                         )


# %%

fig, ax = plt.subplots(figsize=(8,4))

plot_df = num_sampling_days_per_year[(num_sampling_days_per_year['site'] == 'lynch_cove_mid') & (num_sampling_days_per_year['surf_deep'] == 'deep')]

sns.scatterplot(data=plot_df, x='year', y='num_sampling_days', hue='season', hue_order=['grow', 'loDO', 'winter'], palette={'grow':'gold','loDO':'red','winter':'blue'}, ax=ax, legend=True)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/lc_num_sampling_days_per_year.png', bbox_inches='tight', dpi=500, transparent=True)

# %%

num_sampling_days_per_year_full = (odf_depth_mean
                         .groupby(['site','year','surf_deep']).agg({'yearday' :lambda x: x.nunique()})
                         .reset_index()
                         .rename(columns={'yearday':'num_sampling_days'})
                         )

# %%

fig, ax = plt.subplots(figsize=(8,4))

plot_df = num_sampling_days_per_year_full[(num_sampling_days_per_year_full['site'] == 'lynch_cove_mid') & (num_sampling_days_per_year_full['surf_deep'] == 'deep')]

sns.scatterplot(data=plot_df, x='year', y='num_sampling_days', ax=ax, legend=True)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/lc_num_sampling_days_per_year_full.png', bbox_inches='tight', dpi=500, transparent=True)

# %%

fig, ax = plt.subplots(figsize=(8,4))

plot_df = num_sampling_days_per_year[(num_sampling_days_per_year['site'] == 'lynch_cove_mid') & (num_sampling_days_per_year['surf_deep'] == 'deep') & (num_sampling_days_per_year['season'] == 'loDO')]

sns.scatterplot(data=plot_df, x='year', y='num_sampling_days', hue='season', hue_order=['grow', 'loDO', 'winter'], palette={'grow':'gold','loDO':'red','winter':'blue'}, ax=ax, legend=True)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/lc_loDO_num_sampling_days_per_year.png', bbox_inches='tight', dpi=500, transparent=True)