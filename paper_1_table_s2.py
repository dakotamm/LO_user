#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:19:58 2025

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

from pyproj import Geod

from shapely.geometry import Polygon





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

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%

odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)


# %%

area_dict = {}

centroid_dict = {}

depth_dict = {}

years_dict = {}


for site in poly_list:
    
    path = path_dict[site]
    
    coords = path.vertices
    
    lats = coords[:,1]
    lons = coords[:,0]
    
    geod = Geod(ellps='WGS84')


    # Compute:
    area, perim = geod.polygon_area_perimeter(lons, lats)
    
    area_dict[site] = area*1e-6
    
    
    p = Polygon(list(zip(lons, lats)))

    
    centroid = p.centroid
    
    centroid_dict[site] = np.array([centroid.x, centroid.y])
    
    
    depth_dict[site] = odf[odf['site'] == site]['min_segment_h'].unique().astype(float)
    
    
    year_min = odf[(odf['site'] == site) & (odf['surf_deep'] == 'deep')]['year'].min()
    
    year_max = odf[(odf['site'] == site) & (odf['surf_deep'] == 'deep')]['year'].max()
    
    years_dict[site] = np.array([year_min, year_max])

    
    
