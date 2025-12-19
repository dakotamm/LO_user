#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:35:24 2025

@author: dakotamascarenas

updated for Estuaries & Coasts submission on 20251111
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

poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf.loc[odf['source'].isin(['kc_his', 'kc_whidbeyBasin', 'kc_pointJefferson', 'kc']), 'data_source'] = 'King County (KC)'

odf.loc[odf['source'].isin(['ecology_nc', 'ecology_his']), 'data_source'] = 'WA Dept. of Ecology (Eco.)'

odf.loc[odf['source'].isin(['collias']), 'data_source'] = 'Collias (Col.)'

odf.loc[odf['source'].isin(['nceiSalish']), 'data_source'] = 'NCEI Salish Sea (NCEI)'


odf['site'] = odf['segment']


odf.loc[odf['otype'] == 'ctd', 'sampling_type'] = 'CTD+DO'

odf.loc[odf['otype'] == 'bottle', 'sampling_type'] = 'Bottle'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_his'), 'sampling_type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_pointJefferson') & (odf['year'] <= 1998), 'sampling_type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'ecology_his') & (odf['year'] <= 1988), 'sampling_type'] = 'Sonde (unknown type)'

# %%

ps_casts_DF = odf.copy()

ps_casts_DF = ps_casts_DF[['cid', 'lon', 'lat', 'time', 'datetime', 'date_ordinal', 'decade', 'year', 'season', 'month', 'yearday', 'z', 'var',
       'val', 'ix', 'iy', 'h', 'data_source', 'sampling_type']]

ps_casts_DF.to_pickle('/Users/dakotamascarenas/Desktop/Mascarenas_etal_2025/ps_casts_DF.p')

# %%

# %%

# %%











# %%

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)



# %%

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)

# %%

# %%

odf.loc[odf['source'].isin(['kc_his', 'kc_whidbeyBasin', 'kc_pointJefferson', 'kc']), 'data_source'] = 'King County (KC)'

odf.loc[odf['source'].isin(['ecology_nc', 'ecology_his']), 'data_source'] = 'WA Dept. of Ecology (Eco.)'

odf.loc[odf['source'].isin(['collias']), 'data_source'] = 'Collias (Col.)'

odf.loc[odf['source'].isin(['nceiSalish']), 'data_source'] = 'NCEI Salish Sea (NCEI)'


#odf['site'] = odf['segment']


odf.loc[odf['otype'] == 'ctd', 'sampling_type'] = 'CTD+DO'

odf.loc[odf['otype'] == 'bottle', 'sampling_type'] = 'Bottle'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_his'), 'sampling_type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_pointJefferson') & (odf['year'] <= 1998), 'sampling_type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'ecology_his') & (odf['year'] <= 1988), 'sampling_type'] = 'Sonde (unknown type)'

# %%

#site_casts_DF = odf[odf['site'] == 'point_jefferson']

site_casts_DF = odf[['site', 'cid', 'lon', 'lat', 'time', 'datetime', 'date_ordinal', 'surf_deep', 'decade', 'year', 'season', 'month', 'yearday', 'z', 'var',
       'val', 'ix', 'iy', 'h', 'min_segment_h', 'data_source', 'sampling_type']]

site_casts_DF.to_pickle('/Users/dakotamascarenas/Desktop/Mascarenas_etal_2025/site_casts_DF.p')

# %%

# %%
odf_use = odf_depth_mean.copy()



# %%

odf_use = (odf_use
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


# %%

with open('/Users/dakotamascarenas/Desktop/Mascarenas_etal_2025/site_polygon_dict.p', 'wb') as fp:
    pickle.dump(path_dict, fp)
    
# %%

site_depth_avg_var_DF = odf_use[['site', 'datetime', 'date_ordinal', 'year', 'season', 'month', 'yearday', 'cid', 'lon', 'lat', 'surf_deep', 'z', 'var', 'val', 'ix', 'iy', 'h','min_segment_h']]

site_depth_avg_var_DF.to_pickle('/Users/dakotamascarenas/Desktop/Mascarenas_etal_2025/site_depth_avg_var_DF.p')


# %%

# %%

# %%

with open('/Users/dakotamascarenas/Desktop/Mascarenas_etal_2025/X.p', 'wb') as fp:
    pickle.dump(X, fp)
    
with open('/Users/dakotamascarenas/Desktop/Mascarenas_etal_2025/Y.p', 'wb') as fp:
    pickle.dump(Y, fp)
    
with open('/Users/dakotamascarenas/Desktop/Mascarenas_etal_2025/zm_inverse.p', 'wb') as fp:
    pickle.dump(zm_inverse, fp)
    
with open('/Users/dakotamascarenas/Desktop/Mascarenas_etal_2025/plon.p', 'wb') as fp:
    pickle.dump(plon, fp)
    
with open('/Users/dakotamascarenas/Desktop/Mascarenas_etal_2025/plat.p', 'wb') as fp:
    pickle.dump(plat, fp)
