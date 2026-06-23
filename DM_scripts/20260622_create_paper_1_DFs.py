#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 14:35:24 2025

@author: dakotamascarenas

updated for Estuaries & Coasts resubmission on 20260622

20260622 update (R1 revision):
  - Extended year_list to np.arange(1930, 2027) to bring in the most up-to-date
    obs (ecology_nc & kc now reach 2025; kc & kc_whidbeyBasin reach 2026 -- 2026
    is a partial/in-progress year).
  - Properly include nceiSalish: its files live under obs/nceiSalish/bottle_ctd/
    (not bottle/ or ctd/), so the previous otype_list=['bottle','ctd'] silently
    dropped every nceiSalish read via the except: pass in getPolyData. Added
    'bottle_ctd' to otype_list so NCEI Salish Sea (WOAC) data actually loads.
    Other sources have no bottle_ctd dir, so they just hit the swallowed
    FileNotFoundError -- no other behavior changes.
  - Added a sampling_type rule for otype=='bottle_ctd' (set to 'Bottle'; nceiSalish
    DO/nutrients are bottle/Winkler-derived). Change to 'CTD+DO' if preferred.
  - Repointed all output pickles to Desktop/Mascarenas_etal_2026_R1_working/.
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


# output directory for this run (R1 revision working dir)
out_dir = '/Users/dakotamascarenas/Desktop/Mascarenas_etal_2026_R1_working/'

# obs source / otype / year configuration (shared by both getPolyData calls)
source_list = ['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson', 'prism']
otype_list = ['bottle', 'ctd', 'bottle_ctd']  # 'bottle_ctd' is nceiSalish's dir
year_list = np.arange(1930, 2027)  # through 2026 (2026 partial / in progress)

# PRISM (ctd) and nceiSalish (bottle_ctd) are the SAME WOAC cruises, with duplicate
# CTD profiles in 2008-2018. To avoid double-counting, keep only PRISM's non-overlap
# years (< prism_overlap_year) and let nceiSalish cover 2008 onward. PRISM has CT/SA
# only (no DO).
prism_overlap_year = 2008


def drop_prism_overlap(odf_dict, cutoff_year=prism_overlap_year):
    """Remove PRISM rows in the nceiSalish overlap period (year >= cutoff_year)."""
    for poly in odf_dict:
        d = odf_dict[poly]
        yr = pd.to_datetime(d['time'], utc=True).dt.year
        odf_dict[poly] = d[~((d['source'] == 'prism') & (yr >= cutoff_year))].copy()
    return odf_dict


def drop_kc_pj_exact_dups(odf_dict):
    """The 'kc' and 'kc_pointJefferson' sources both record the KSBP01 (Point
    Jefferson) station, so casts at the same timestamp AND location appear twice.
    Keep the dedicated kc_pointJefferson copy and drop the matching kc cast (matched
    on time-to-minute + lon/lat rounded to 3 decimals)."""
    for poly in odf_dict:
        d = odf_dict[poly]
        key = (pd.to_datetime(d['time'], utc=True).dt.strftime('%Y-%m-%d %H:%M')
               + '_' + d['lon'].round(3).astype(str) + '_' + d['lat'].round(3).astype(str))
        kcpj_keys = set(key[d['source'] == 'kc_pointJefferson'])
        drop = (d['source'] == 'kc') & key.isin(kcpj_keys)
        odf_dict[poly] = d[~drop].copy()
    return odf_dict


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

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=source_list, otype_list=otype_list, year_list=year_list)

odf_dict = drop_prism_overlap(odf_dict)

odf_dict = drop_kc_pj_exact_dups(odf_dict)


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf.loc[odf['source'].isin(['kc_his', 'kc_whidbeyBasin', 'kc_pointJefferson', 'kc']), 'data_source'] = 'King County (KC)'

odf.loc[odf['source'].isin(['ecology_nc', 'ecology_his']), 'data_source'] = 'WA Dept. of Ecology (Eco.)'

odf.loc[odf['source'].isin(['collias']), 'data_source'] = 'Collias (Col.)'

odf.loc[odf['source'].isin(['nceiSalish', 'prism']), 'data_source'] = 'Salish Cruises/PRISM (SCDP/P)'

#odf.loc[odf['source'].isin(['prism']), 'data_source'] = 'PRISM'


odf['site'] = odf['segment']


odf.loc[odf['otype'] == 'ctd', 'sampling_type'] = 'CTD+DO'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'prism'), 'sampling_type'] = 'CTD'  # PRISM CTD has CT/SA only, no DO

odf.loc[odf['otype'] == 'bottle', 'sampling_type'] = 'Bottle'

odf.loc[odf['otype'] == 'bottle_ctd', 'sampling_type'] = 'Bottle/CTD+DO'  # nceiSalish (WOAC); change to 'CTD+DO' if preferred

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_his'), 'sampling_type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_pointJefferson') & (odf['year'] <= 1998), 'sampling_type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'ecology_his') & (odf['year'] <= 1988), 'sampling_type'] = 'Sonde (unknown type)'

# %%

ps_casts_DF = odf.copy()

ps_casts_DF = ps_casts_DF[['cid', 'lon', 'lat', 'time', 'datetime', 'date_ordinal', 'decade', 'year', 'season', 'month', 'yearday', 'z', 'var',
       'val', 'ix', 'iy', 'h', 'data_source', 'sampling_type']]

ps_casts_DF.to_pickle(out_dir + 'ps_casts_DF.p')

# %%

# %%

# %%












# %%

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=source_list, otype_list=otype_list, year_list=year_list)

odf_dict = drop_prism_overlap(odf_dict)

odf_dict = drop_kc_pj_exact_dups(odf_dict)


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

odf.loc[odf['source'].isin(['nceiSalish', 'prism']), 'data_source'] = 'Salish Cruises/PRISM (SCDP/P)'

#odf.loc[odf['source'].isin(['prism']), 'data_source'] = 'PRISM'


#odf['site'] = odf['segment']


odf.loc[odf['otype'] == 'ctd', 'sampling_type'] = 'CTD+DO'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'prism'), 'sampling_type'] = 'CTD'  # PRISM CTD has CT/SA only, no DO

odf.loc[odf['otype'] == 'bottle', 'sampling_type'] = 'Bottle'

odf.loc[odf['otype'] == 'bottle_ctd', 'sampling_type'] = 'Bottle/CTD+DO'  # nceiSalish (WOAC); change to 'CTD+DO' if preferred

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_his'), 'sampling_type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'kc_pointJefferson') & (odf['year'] <= 1998), 'sampling_type'] = 'Sonde (unknown type)'

odf.loc[(odf['otype'] == 'ctd') & (odf['source'] == 'ecology_his') & (odf['year'] <= 1988), 'sampling_type'] = 'Sonde (unknown type)'

# %%

#site_casts_DF = odf[odf['site'] == 'point_jefferson']

site_casts_DF = odf[['site', 'cid', 'lon', 'lat', 'time', 'datetime', 'date_ordinal', 'surf_deep', 'decade', 'year', 'season', 'month', 'yearday', 'z', 'var',
       'val', 'ix', 'iy', 'h', 'min_segment_h', 'data_source', 'sampling_type']]

site_casts_DF.to_pickle(out_dir + 'site_casts_DF.p')

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

with open(out_dir + 'site_polygon_dict.p', 'wb') as fp:
    pickle.dump(path_dict, fp)

# %%

site_depth_avg_var_DF = odf_use[['site', 'datetime', 'date_ordinal', 'year', 'season', 'month', 'yearday', 'cid', 'lon', 'lat', 'surf_deep', 'z', 'var', 'val', 'ix', 'iy', 'h','min_segment_h']]

site_depth_avg_var_DF.to_pickle(out_dir + 'site_depth_avg_var_DF.p')

# %%

# additional longShortClean outputs needed by paper_1_table_*.py (so the tables can
# run off these pickles instead of rebuilding from raw obs via getPolyData)
odf_depth_mean.to_pickle(out_dir + 'odf_depth_mean.p')

odf_calc_long.to_pickle(out_dir + 'odf_calc_long.p')

odf_depth_mean_deep_DO_percentiles.to_pickle(out_dir + 'odf_depth_mean_deep_DO_percentiles.p')


# %%

# %%

# %%

with open(out_dir + 'X.p', 'wb') as fp:
    pickle.dump(X, fp)

with open(out_dir + 'Y.p', 'wb') as fp:
    pickle.dump(Y, fp)

with open(out_dir + 'zm_inverse.p', 'wb') as fp:
    pickle.dump(zm_inverse, fp)

with open(out_dir + 'plon.p', 'wb') as fp:
    pickle.dump(plon, fp)

with open(out_dir + 'plat.p', 'wb') as fp:
    pickle.dump(plat, fp)
