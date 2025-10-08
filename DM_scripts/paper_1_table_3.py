#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:23:25 2025

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




poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] # 5 sites + 4 basins

#poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

site_list =  odf['site'].unique()




odf_use = odf_depth_mean.copy()

odf_calc_use = odf_calc_long.copy()

all_stats_filt = dfun.buildStatsDF(odf_use, site_list, odf_calc_use=odf_calc_use, odf_depth_mean_deep_DO_percentiles=odf_depth_mean_deep_DO_percentiles)



# %%

c=0

all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_label'] = 'PJ'

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_label'] = 'NS'

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_label'] = 'SP'

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_label'] = 'CI'

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_label'] = 'LC'


all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_type'] = 'Main Basin'

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_type'] = 'Main Basin'

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_type'] = 'Sub-Basins'

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_type'] = 'Sub-Basins'

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_type'] = 'Sub-Basins'


all_stats_filt.loc[all_stats_filt['site'] == 'point_jefferson', 'site_num'] = 1

all_stats_filt.loc[all_stats_filt['site'] == 'near_seattle_offshore', 'site_num'] = 2

all_stats_filt.loc[all_stats_filt['site'] == 'saratoga_passage_mid', 'site_num'] = 3

all_stats_filt.loc[all_stats_filt['site'] == 'carr_inlet_mid', 'site_num'] = 4

all_stats_filt.loc[all_stats_filt['site'] == 'lynch_cove_mid', 'site_num'] = 5

# %%

all_stats_filt['95CI_hi'] = all_stats_filt['slope_datetime_s_hi'] - all_stats_filt['slope_datetime'] 

all_stats_filt['95CI_lo'] = all_stats_filt['slope_datetime'] -all_stats_filt['slope_datetime_s_lo']

# %%

all_stats_disp = all_stats_filt[all_stats_filt['var'].isin(['deep_DO_mg_L','deep_CT','deep_SA', 'surf_DO_mg_L', 'surf_CT', 'surf_SA', 'deep_DO_sol', 'surf_DO_sol'])]

all_stats_disp.loc[all_stats_disp['var'].isin(['deep_DO_mg_L','deep_CT','deep_SA', 'deep_DO_sol']), 'Depth'] = 'Bottom'

all_stats_disp.loc[all_stats_disp['var'].isin(['surf_DO_mg_L','surf_CT','surf_SA', 'surf_DO_sol']), 'Depth'] = 'Surface'

all_stats_disp.loc[all_stats_disp['var'].isin(['deep_DO_mg_L','surf_DO_mg_L']), 'var'] = 'DO [mg/L]'

all_stats_disp.loc[all_stats_disp['var'].isin(['deep_CT','surf_CT']), 'var'] = 'CT [°C]'

all_stats_disp.loc[all_stats_disp['var'].isin(['deep_SA','surf_SA']), 'var'] = 'SA [g/kg]'

all_stats_disp.loc[all_stats_disp['var'].isin(['deep_DO_sol','surf_DO_sol']), 'var'] = 'Sol.-Based DO [mg/L]'




# %%

all_stats_disp['slope_datetime_cent'] = all_stats_disp['slope_datetime']*100

all_stats_disp['slope_datetime_s_hi_cent'] = all_stats_disp['slope_datetime_s_hi']*100

all_stats_disp['slope_datetime_s_lo_cent'] = all_stats_disp['slope_datetime_s_lo']*100

all_stats_disp['95CI_hi_cent'] = all_stats_disp['95CI_hi']*100

all_stats_disp['95CI_lo_cent'] = all_stats_disp['95CI_lo']*100



all_stats_disp['95CI_hi_cent_str'] = all_stats_disp['95CI_hi_cent'].round(2).astype(str)

all_stats_disp['95CI_lo_cent_str'] = all_stats_disp['95CI_lo_cent'].round(2).astype(str)

all_stats_disp['slope_datetime_cent_str'] = all_stats_disp['slope_datetime_cent'].round(2).astype(str)


all_stats_disp['trend_95_ci'] = all_stats_disp['slope_datetime_cent_str'] + ' +' + all_stats_disp['95CI_hi_cent_str'] + '/-' + all_stats_disp['95CI_lo_cent_str']

# %%

all_stats_disp['Site'] = all_stats_disp['site_label']

all_stats_disp.loc[all_stats_disp['season'] == 'grow', 'Season'] = 'Apr-Jul'

all_stats_disp.loc[all_stats_disp['season'] == 'loDO', 'Season'] = 'Aug-Nov'

all_stats_disp.loc[all_stats_disp['season'] == 'winter', 'Season'] = 'Dec-Mar'

all_stats_disp = all_stats_disp[all_stats_disp['season'] != 'allyear']


all_stats_disp['Var.'] = all_stats_disp['var']

all_stats_disp['Trend Slope (95% CI)'] = all_stats_disp['trend_95_ci']





# %%

all_stats_disp_use = all_stats_disp[['Site', 'Season','Var.','Depth', 'Trend Slope (95% CI)','p','n']]

# %%


all_stats_disp_use_wide = all_stats_disp_use.pivot(index=['Site','Season'], columns= ['Depth', 'Var.'])

all_stats_disp_use_wide.columns = all_stats_disp_use_wide.columns.reorder_levels([2, 1, 0])

# %%

site_order = ['PJ', 'NS', 'CI', 'SP', 'LC']
season_order = ['Dec-Mar', 'Apr-Jul', 'Aug-Nov']



var_order = ['CT [°C]', 'SA [g/kg]', 'DO [mg/L]', 'Sol.-Based DO [mg/L]']
depth_order = ['Surface', 'Bottom']
stat_order = ['Trend Slope w/ 95% CI [unit/cent.]', 'p', 'n']

all_stats_disp_use_wide = (
    all_stats_disp_use_wide
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
cols = all_stats_disp_use_wide.columns

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
all_stats_disp_use_wide = all_stats_disp_use_wide.iloc[:, sorted_indexer]



# %%

all_stats_disp_use_wide.to_excel('/Users/dakotamascarenas/Desktop/paper_1_table_3.xlsx')  
