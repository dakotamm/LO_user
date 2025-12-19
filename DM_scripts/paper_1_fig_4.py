#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:35:03 2025

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

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_his', 'kc_whidbeyBasin', 'nceiSalish', 'kc_pointJefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_1D, lat_1D, depths, lon, lat, poly_list, path_dict, basin_list)

# %%


odf, odf_depth_mean, odf_calc_long, odf_depth_mean_deep_DO_percentiles, long_site_list, short_site_list, big_basin_list, site_list = dfun.longShortClean(odf)


# %%

odf_depth_mean_0 = odf_depth_mean.copy()

cid_deep = odf_depth_mean_0.loc[odf_depth_mean_0['surf_deep'] == 'deep', 'cid']

# %%

odf_depth_mean_0 = odf_depth_mean_0[odf_depth_mean_0['cid'].isin(cid_deep)]

# NOTE THIS IS JUST TAKING FULL DEPTH CASTS, DON'T NECESSARILY NEED THIS FOR DO SOL

# %%

odf_use = odf_depth_mean.copy()


odf_use = (odf_use
                  # .drop(columns=['date_ordinal_std'])
                  #.rename(columns={'date_ordinal_mean':'date_ordinal'})
                  #.reset_index() 
                  .dropna()
                  .assign(
                          #segment=(lambda x: key),
                          # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                          # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                          # season=(lambda x: pd.cut(x['month'],
                          #                          bins=[0,3,6,9,12],
                          #                          labels=['winter', 'spring', 'summer', 'fall'])),
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


# %%

odf_use_DO = odf_use[odf_use['var'] == 'DO_mg_L'].reset_index(drop=True)

# %%

DO_min_idx = odf_use_DO.groupby(['site','var','year']).idxmin()['val'].to_numpy()

# %%

odf_use_DO_min = odf_use_DO[odf_use_DO.index.isin(DO_min_idx)]
    
# %%

odf_use_DO_deep = odf_use[(odf_use['var'] == 'DO_mg_L') & (odf_use['surf_deep'] == 'deep')].reset_index(drop=True)

# %%

DO_min_deep_idx = odf_use_DO_deep.groupby(['site','var','year']).idxmin()['val'].to_numpy()

# %%

odf_use_DO_min_deep = odf_use_DO_deep[odf_use_DO_deep.index.isin(DO_min_deep_idx)]

# %%

odf_DO_deep = odf[(odf['var'] == 'DO_mg_L') & (odf['cid'].isin(cid_deep))]

odf_cast_mins = odf_DO_deep.groupby('cid').min(numeric_only=True).reset_index()

odf_cast_mins['min_z'] = odf_cast_mins['z']

# %%

odf_z_at_min_val = odf_DO_deep.loc[odf_DO_deep.groupby('cid')['val'].idxmin()]

odf_test = pd.merge(odf_z_at_min_val, odf_cast_mins[['cid','min_z']], on = 'cid', how='left')

# %%

odf_test_test = odf_test[odf_test['z'] <= odf_test['min_z']*0.95]

# %%


odf_val_at_min_z = odf_DO_deep.loc[odf_DO_deep.groupby('cid')['z'].idxmin()]


# %%

odf_Aug_Nov = odf_use_DO_min_deep[odf_use_DO_min_deep['month'].isin([8,9,10,11])]

# %%

odf_val_at_min_z['val_at_min_z'] = odf_val_at_min_z['val']

odf_cast_mins['val_cast_mins'] = odf_cast_mins['val']



odf_double_mins = pd.merge(odf_val_at_min_z, odf_cast_mins[['cid', 'val_cast_mins']], how='left', on = ['cid'])

# %%

red =     "#EF5E3C"   # warm orange-red ##ff4040 #e04256

blue =     "#3A59B3"  # deep blue #4565e8

mosaic = [['depth', 'yearday', 'yearday']]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(9,3), layout='constrained', gridspec_kw=dict(wspace=0.1))


    

ax = axd['yearday']

plot_df = odf_use_DO_min_deep.copy()



sns.scatterplot(data = plot_df, x='year', y = 'yearday',  color = 'gray', ax=ax)

sns.scatterplot(data = plot_df[plot_df['val'] <2], x='year', y = 'yearday',  color = red, label='value <2 [mg/L] (hypoxic)', ax=ax)




ax.set_ylim(0,366)

ax.set_ylabel('Yearday Occurence of Min. DO') 

ax.set_xlabel('Year')


ax.axhspan(213,335, color = 'lightgray', alpha = 0.5, label = 'Low-DO (Aug-Nov)', zorder=-4) #july31/august1-september30/oct1

ax.legend()

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.text(0.025,0.05, 'b', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')






ax = axd['depth']

sns.scatterplot(data = odf_double_mins, x='val_at_min_z', y = 'val_cast_mins', alpha=0.1, color='gray', ax=ax)

#ax.scatter(odf_val_at_min_z['val'], odf_cast_mins['val'], alpha=0.1, color = 'gray')

#ax.hist2d(odf_val_at_min_z['val'], odf_cast_mins['val'], bins=100, cmap='inferno', cmin=1)
 
#ax.colorbar(label='Cast Count')

ax.set_xlabel('DO at Max. Cast Depth [mg/L]')

ax.set_ylabel('Min. Cast DO [mg/L]')

#ax.axhspan(0,2, color = 'lightgray', alpha = 0.5)

#ax.axvspan(0,2, color = 'lightgray', alpha = 0.5)



ax.axis('square') 

#ax.set_ylim(0,18)

ax.set_xlim(xmin=0) 

ax.set_ylim(ymin=0) 




ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5) 

ax.text(0.05,0.05, 'a', transform=ax.transAxes, fontsize=14, fontweight='bold', color = 'k')


    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/paper_1_fig_4.png', bbox_inches='tight', dpi=500, transparent=False)