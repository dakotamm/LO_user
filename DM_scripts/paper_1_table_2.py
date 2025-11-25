#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 16:00:01 2025

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

odf_use_seasonal_DO, odf_use_seasonal_CTSA, odf_use_annual_DO, odf_use_annual_CTSA = dfun.calcSeriesAvgs(odf_depth_mean, odf_depth_mean_deep_DO_percentiles, deep_DO_q = 'deep_DO_q50', filter_out=True)

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

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'point_jefferson', 'site_label'] = 'PJ'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'point_jefferson', 'site_num'] = 1

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'near_seattle_offshore', 'site_num'] = 2

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'saratoga_passage_mid', 'site_num'] = 4

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'carr_inlet_mid', 'site_num'] = 3

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['site'] == 'lynch_cove_mid', 'site_num'] = 5


odf_use_seasonal_DO.loc[odf_use_seasonal_DO['season'] == 'grow', 'season_label'] = 'Spring (Apr-Jul)'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['season'] == 'loDO', 'season_label'] = 'Low-DO (Aug-Nov)'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['season'] == 'winter', 'season_label'] = 'Winter (Dec-Mar)'


odf_use_seasonal_DO.loc[odf_use_seasonal_DO['surf_deep'] == 'surf', 'depth_label'] = 'Surface'

odf_use_seasonal_DO.loc[odf_use_seasonal_DO['surf_deep'] == 'deep', 'depth_label'] = 'Bottom'

# %%

loDO_deep_mean_ci_CTSA = odf_use_seasonal_CTSA[(odf_use_seasonal_CTSA['var'].isin(['CT','SA'])) & (odf_use_seasonal_CTSA['surf_deep'] == 'deep') & (odf_use_seasonal_CTSA['season'] == 'loDO')]

loDO_deep_mean_ci_CTSA['95ci_less'] =  loDO_deep_mean_ci_CTSA['val_mean'] - loDO_deep_mean_ci_CTSA['val_ci95lo']

loDO_deep_mean_ci_CTSA['95ci_more'] =  loDO_deep_mean_ci_CTSA['val_ci95hi'] - loDO_deep_mean_ci_CTSA['val_mean']

loDO_deep_mean_ci_CTSA = loDO_deep_mean_ci_CTSA[['site', 'site_label', 'depth_label','var', 'val_mean', '95ci_less','95ci_more', 'cid_count']]

# %%

loDO_deep_mean_ci_DO = odf_use_seasonal_DO[(odf_use_seasonal_DO['var']== 'DO_mg_L') & (odf_use_seasonal_DO['surf_deep'] == 'deep') & (odf_use_seasonal_DO['season'] == 'loDO')]

loDO_deep_mean_ci_DO['95ci_less'] =  loDO_deep_mean_ci_DO['val_mean'] - loDO_deep_mean_ci_DO['val_ci95lo']

loDO_deep_mean_ci_DO['95ci_more'] =  loDO_deep_mean_ci_DO['val_ci95hi'] - loDO_deep_mean_ci_DO['val_mean']

loDO_deep_mean_ci_DO = loDO_deep_mean_ci_DO[['site', 'site_label', 'depth_label', 'var', 'val_mean', '95ci_less','95ci_more', 'cid_count']]

# %%

loDO_deep_mean_ci = pd.concat([loDO_deep_mean_ci_CTSA, loDO_deep_mean_ci_DO])

# %%

loDO_deep_mean_ci_PJLC = loDO_deep_mean_ci[loDO_deep_mean_ci['site'].isin(['point_jefferson', 'lynch_cove_mid'])]

# %%

mean_ci_CTSA = odf_use_seasonal_CTSA[(odf_use_seasonal_CTSA['var'].isin(['CT','SA']))]

mean_ci_CTSA['95ci_less'] =  mean_ci_CTSA['val_mean'] - mean_ci_CTSA['val_ci95lo']

mean_ci_CTSA['95ci_more'] =  mean_ci_CTSA['val_ci95hi'] - mean_ci_CTSA['val_mean']

mean_ci_CTSA = mean_ci_CTSA[['site', 'site_label', 'depth_label','var', 'surf_deep', 'season', 'val_mean', '95ci_less','95ci_more', 'cid_count']]

mean_ci_DO = odf_use_seasonal_DO[(odf_use_seasonal_DO['var']== 'DO_mg_L')]

mean_ci_DO['95ci_less'] =  mean_ci_DO['val_mean'] - mean_ci_DO['val_ci95lo']

mean_ci_DO['95ci_more'] =  mean_ci_DO['val_ci95hi'] - mean_ci_DO['val_mean']

mean_ci_DO = mean_ci_DO[['site', 'site_label', 'depth_label','var', 'surf_deep', 'season', 'val_mean', '95ci_less','95ci_more', 'cid_count']]

mean_ci = pd.concat([mean_ci_CTSA, mean_ci_DO])

# %%

mean_ci_disp = mean_ci.copy()

mean_ci_disp['95ci_str'] = mean_ci_disp['95ci_less'].round(2).astype(str)

mean_ci_disp['val_mean_str'] = mean_ci_disp['val_mean'].round(2).astype(str)

mean_ci_disp['val_95_ci'] = mean_ci_disp['val_mean_str'] + ' +/-' + mean_ci_disp['95ci_str']

# %%

mean_ci_disp['Site'] = mean_ci_disp['site_label']

mean_ci_disp.loc[mean_ci_disp['season'] == 'grow', 'Season'] = 'Apr-Jul'

mean_ci_disp.loc[mean_ci_disp['season'] == 'loDO', 'Season'] = 'Aug-Nov'

mean_ci_disp.loc[mean_ci_disp['season'] == 'winter', 'Season'] = 'Dec-Mar'

mean_ci_disp = mean_ci_disp[mean_ci_disp['season'] != 'allyear']

mean_ci_disp.loc[mean_ci_disp['var'] == 'DO_mg_L', 'var'] = 'DO [mg/L]'

mean_ci_disp.loc[mean_ci_disp['var'] == 'CT', 'var'] = 'CT [°C]'

mean_ci_disp.loc[mean_ci_disp['var'] == 'SA', 'var'] = 'SA [g/kg]'

mean_ci_disp['Depth'] = mean_ci_disp['depth_label']


mean_ci_disp['Var.'] = mean_ci_disp['var']

mean_ci_disp['Mean Value (95% CI)'] = mean_ci_disp['val_95_ci']

mean_ci_disp['n'] = mean_ci_disp['cid_count']

# %%

mean_ci_disp_use = mean_ci_disp[['Site', 'Season','Var.','Depth', 'Mean Value (95% CI)','n']]


# %%

season_avg = mean_ci.groupby(['site','var','surf_deep'])['val_mean'].mean().reset_index()

depth_avg = mean_ci.groupby(['site','var','season'])['val_mean'].mean().reset_index()

site_depth_avg = mean_ci.groupby(['var','season'])['val_mean'].mean().reset_index()

site_avg = mean_ci.groupby(['var','season', 'surf_deep'])['val_mean'].mean().reset_index()


# %%

mean_ci_disp_use_wide = mean_ci_disp_use.pivot(index=['Site','Season'], columns= ['Depth', 'Var.'])

mean_ci_disp_use_wide.columns = mean_ci_disp_use_wide.columns.reorder_levels([2, 1, 0])

# %%

site_order = ['PJ', 'NS', 'CI', 'SP', 'LC']
season_order = ['Dec-Mar', 'Apr-Jul', 'Aug-Nov']



var_order = ['CT [°C]', 'SA [g/kg]', 'DO [mg/L]']
depth_order = ['Surface', 'Bottom']
stat_order = ['Mean Value (95% CI)', 'n']

mean_ci_disp_use_wide = (
    mean_ci_disp_use_wide
    .sort_index(
        level=['Site', 'Season'],
        key=lambda idx: (
            idx.map({v: i for i, v in enumerate(site_order)}) 
            if idx.name == 'Site' else 
            idx.map({v: i for i, v in enumerate(season_order)}) 
        )
    )
)


# Extract current column MultiIndex
cols = mean_ci_disp_use_wide.columns

# Get positions in the custom order
var_pos = pd.Index(var_order).get_indexer(cols.get_level_values('Var.'))
depth_pos = pd.Index(depth_order).get_indexer(cols.get_level_values('Depth'))
stat_pos = pd.Index(stat_order).get_indexer(cols.get_level_values(None))

# Combine into a DataFrame for sorting
sort_df = pd.DataFrame({
    'Var.': var_pos,
    'Depth': depth_pos,
    'Stat.': stat_pos
})

# Get new column order based on sort priorities
sorted_indexer = sort_df.sort_values(
    by=['Var.', 'Depth', 'Stat.']
).index

# Apply new column order
mean_ci_disp_use_wide = mean_ci_disp_use_wide.iloc[:, sorted_indexer]

# %%

mean_ci_disp_use_wide.to_excel('/Users/dakotamascarenas/Desktop/paper_1_table_2.xlsx')  