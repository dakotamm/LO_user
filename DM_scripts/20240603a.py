#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:13:01 2024

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



# %%

Ldir = Lfun.Lstart(gridname='cas7')

# %%

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

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson', 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'DO_mg_L'] #'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']

# %%

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                      .assign(
                          datetime=(lambda x: pd.to_datetime(x['time'], utc=True)),
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
            
    odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade', 'name'],
                                          value_vars=var_list, var_name='var', value_name = 'val')
    
# %%

odf = pd.concat(odf_dict.values(), ignore_index=True)

# %%

odf['source_type'] = odf['source'] + '_' + odf['otype']

# %%

odf = odf.dropna()

# %%

short_exclude_sites = ['BUD002', 'QMH002', 'PMA001', 'OCH014', 'DYE004', 'SUZ001', 'HLM001', 'PNN001', 'PSS010', 'TOT002', 'TOT001', 'HND001','ELD001', 'ELD002', 'CSE002', 'CSE001', 'HCB010', 'SKG003','HCB006', 'HCB008', 'HCB009', 'CMB006', 'EAG001', 'HCB013', 'POD007']

big_basin_list = ['mb', 'wb', 'ss', 'hc']

long_site_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']


short_mask_ecology = (odf['segment'].isin(big_basin_list)) & (odf['source'] == 'ecology_nc') & (~odf['name'].isin(short_exclude_sites)) & (odf['year'] >= 1998)

short_mask_point_jefferson = (odf['segment'] == 'mb') & (odf['name'] =='KSBP01') & (odf['year'] >= 1998)

long_mask = (odf['segment'].isin(long_site_list))

# %%

odf.loc[short_mask_ecology, 'short_long'] = 'short'

odf.loc[short_mask_point_jefferson, 'short_long'] = 'short'

odf.loc[long_mask, 'short_long'] = 'long'

# %%

odf = odf[odf['short_long'] != 'nan']

# %%

short_site_list = odf[odf['short_long'] == 'short']['name'].unique()


# %%

odf = odf.assign(
    ix=(lambda x: x['lon'].apply(lambda x: zfun.find_nearest_ind(lon_1D, x))),
    iy=(lambda x: x['lat'].apply(lambda x: zfun.find_nearest_ind(lat_1D, x)))
)

# %%

odf['h'] = odf.apply(lambda x: -depths[x['iy'], x['ix']], axis=1)

# %%

odf['yearday'] = odf['datetime'].dt.dayofyear

# %%

odf = odf[odf['val'] >0]


# %%

max_depths_dict = dict()

ox = lon
oy = lat
oxoy = np.concatenate((ox.reshape(-1,1),oy.reshape(-1,1)), axis=1)


for poly in poly_list:

    path = path_dict[poly]
    
    oisin = path.contains_points(oxoy)
    
    this_depths = depths.flatten()[oisin]
    
    max_depth = np.nanmax(this_depths)
    
    max_depths_dict[poly] = max_depth.copy()
    
# %%


for basin in basin_list:
    
    odf.loc[odf['segment'] == basin, 'min_segment_h'] = -max_depths_dict[basin]
    
# %%

#odf.loc[odf['name'] == 'BUD002', 'h'] = -14

# %%

summer_mask = (odf['yearday'] > 125) & (odf['yearday']<= 325)

surf_mask = (odf['z'] >= -5)


long_deep_non_lc_nso_mask = (odf['z'] < 0.8*odf['min_segment_h']) & (odf['segment'] != 'lynch_cove_mid') & (odf['segment'] != 'near_seattle_offshore') & (odf['short_long'] == 'long')

long_deep_lc_mask = (odf['z'] < 0.4*odf['min_segment_h']) & (odf['segment'] == 'lynch_cove_mid') & (odf['short_long'] == 'long')

long_deep_nso_mask = (odf['z'] < 0.75*odf['min_segment_h']) & (odf['segment'] == 'near_seattle_offshore') & (odf['short_long'] == 'long') #CHANGED 5/21/2024


short_deep_mask = (odf['z'] < 0.8*odf['h']) & (odf['short_long'] == 'short')



supershallowsite_mask_long = (odf['min_segment_h'] >=-10) & (odf['short_long'] == 'long')

shallowsite_mask_long = (odf['min_segment_h'] < -10) & (odf['min_segment_h'] >= -60) & (odf['short_long'] == 'long')

deepsite_mask_long = (odf['min_segment_h'] < -60) & (odf['short_long'] == 'long')


supershallowsite_mask_short = (odf['h'] >=-10) & (odf['short_long'] == 'short')

shallowsite_mask_short = (odf['h'] < -10) & (odf['h'] >= -60) & (odf['short_long'] == 'short')

deepsite_mask_short = (odf['h'] < -60) & (odf['short_long'] == 'short')


# %%

odf.loc[summer_mask, 'summer_non_summer'] = 'summer'

odf.loc[~summer_mask, 'summer_non_summer'] = 'non_summer'

odf.loc[surf_mask, 'surf_deep'] = 'surf'

odf.loc[long_deep_non_lc_nso_mask, 'surf_deep'] = 'deep'

odf.loc[long_deep_lc_mask, 'surf_deep'] = 'deep'

odf.loc[long_deep_nso_mask, 'surf_deep'] = 'deep'

odf.loc[short_deep_mask, 'surf_deep'] = 'deep'

odf.loc[supershallowsite_mask_long, 'site_depth'] = 'supershallow'

odf.loc[supershallowsite_mask_short, 'site_depth'] = 'supershallow'

odf.loc[shallowsite_mask_long, 'site_depth'] = 'shallow'

odf.loc[shallowsite_mask_short, 'site_depth'] = 'shallow'

odf.loc[deepsite_mask_long, 'site_depth'] = 'deep'

odf.loc[deepsite_mask_short, 'site_depth'] = 'deep'


# %%

odf.loc[odf['short_long'] == 'short', 'site'] = odf[odf['short_long'] == 'short']['name']

odf.loc[odf['short_long'] == 'long', 'site'] = odf[odf['short_long'] == 'long']['segment']

# %%

temp = odf.groupby(['site','cid']).min().reset_index()

cid_exclude = temp[(temp['site'].isin(['HCB005', 'HCB007', 'lynch_cove_mid'])) & (temp['z'] < -50)]['cid']

odf = odf[~odf['cid'].isin(cid_exclude)]

# %%

temp0 = odf[odf['surf_deep'] != 'nan']

# %%

odf_depth_mean = temp0.groupby(['site','surf_deep', 'summer_non_summer', 'year', 'var','cid']).mean(numeric_only=True).reset_index().dropna()


# %%

odf_depth_mean_0 = odf_depth_mean.copy()

cid_deep = odf_depth_mean_0.loc[odf_depth_mean_0['surf_deep'] == 'deep', 'cid']

# %%

odf_depth_mean_0 = odf_depth_mean_0[odf_depth_mean_0['cid'].isin(cid_deep)]

# %%

odf_dens = odf_depth_mean_0.pivot(index = ['site', 'year', 'summer_non_summer', 'date_ordinal', 'cid'], columns = ['surf_deep', 'var'], values ='val')

# %%

odf_dens.columns = odf_dens.columns.to_flat_index().map('_'.join)

odf_dens = odf_dens.reset_index()

# %%

odf_dens['surf_dens'] = gsw.density.sigma0(odf_dens['surf_SA'], odf_dens['surf_CT'])

odf_dens['deep_dens'] = gsw.density.sigma0(odf_dens['deep_SA'], odf_dens['deep_CT'])

# %%

odf_dens['strat_sigma'] = odf_dens['deep_dens'] - odf_dens['surf_dens']


# %%

odf_dens = odf_dens[['site', 'year','summer_non_summer', 'date_ordinal', 'cid', 'strat_sigma']]



# %%

annual_counts = (odf_depth_mean
                      .dropna()
                      #.set_index('datetime')
                      .groupby(['site','year','summer_non_summer', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )

# %%

odf_use = odf_depth_mean.groupby(['site', 'surf_deep', 'summer_non_summer', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})

# %%

odf_use.columns = odf_use.columns.to_flat_index().map('_'.join)

odf_use = odf_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# %%

odf_use = (odf_use
                  # .drop(columns=['date_ordinal_std'])
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
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

odf_use = pd.merge(odf_use, annual_counts, how='left', on=['site','surf_deep','summer_non_summer','year','var'])

# %%

odf_use = odf_use[odf_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_use['val_ci95hi'] = odf_use['val_mean'] + 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])

odf_use['val_ci95lo'] = odf_use['val_mean'] - 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])


# %%

annual_counts_dens = (odf_depth_mean
                      .dropna()
                      #.set_index('datetime')
                      .groupby(['site','year','summer_non_summer']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )

# %%

odf_dens_use = odf_dens.groupby(['site', 'summer_non_summer', 'year']).agg({'strat_sigma':['mean', 'std'], 'date_ordinal':['mean']})

# %%

odf_dens_use.columns = odf_dens_use.columns.to_flat_index().map('_'.join)

odf_dens_use = odf_dens_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# %%

odf_dens_use = (odf_dens_use
                  # .drop(columns=['date_ordinal_std'])
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
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

odf_dens_use = pd.merge(odf_dens_use, annual_counts_dens, how='left', on=['site','summer_non_summer','year'])

# %%

odf_dens_use = odf_dens_use[odf_dens_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_dens_use['val_ci95hi'] = odf_dens_use['strat_sigma_mean'] + 1.96*odf_dens_use['strat_sigma_std']/np.sqrt(odf_dens_use['cid_count'])

odf_dens_use['val_ci95lo'] = odf_dens_use['strat_sigma_mean'] - 1.96*odf_dens_use['strat_sigma_std']/np.sqrt(odf_dens_use['cid_count'])

# %%

alpha = 0.05

for site in odf_use['site'].unique():
    
    
    for season in ['summer', 'non_summer']:
        
        mask = (odf_dens_use['site'] == site) & (odf_dens_use['summer_non_summer'] == season)
        
        plot_df = odf_dens_use[mask]
        
        x = plot_df['date_ordinal']
        
        x_plot = plot_df['datetime']
        
        y = plot_df['strat_sigma_mean']
        
        result = stats.linregress(x, y)
        
        B1 = result.slope
        
        B0 = result.intercept
        
        odf_dens_use.loc[mask, 'linreg_B1'] = result.slope
        
        odf_dens_use.loc[mask, 'linreg_B0'] = result.intercept
        
        odf_dens_use.loc[mask, 'linreg_r'] = result.rvalue
        
        odf_dens_use.loc[mask, 'linreg_p'] = result.pvalue
        
        odf_dens_use.loc[mask, 'linreg_sB1'] = result.stderr
        
        odf_dens_use.loc[mask, 'linreg_sB0'] = result.intercept_stderr
        
        # length of the original dataset
        n = len(x)
        # degrees of freedom
        dof = n - 2
        
        # t-value for alpha/2 with n-2 degrees of freedom
        t = stats.t.ppf(1-alpha/2, dof)
        
        slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
        
        odf_dens_use.loc[mask, 'slope_var_datetime'] = slope_datetime #per year
        
        
        for depth in ['surf', 'deep']:
            
            for var in var_list:
                
                mask = (odf_use['site'] == site) & (odf_use['summer_non_summer'] == season) & (odf_use['surf_deep'] == depth) & (odf_use['var'] == var)
                
                plot_df = odf_use[mask]
                
                x = plot_df['date_ordinal']
                
                x_plot = plot_df['datetime']
                
                y = plot_df['val_mean']
                
                result = stats.linregress(x, y)
                
                B1 = result.slope
                
                B0 = result.intercept
                
                odf_use.loc[mask, 'linreg_B1'] = result.slope
                
                odf_use.loc[mask, 'linreg_B0'] = result.intercept
                
                odf_use.loc[mask, 'linreg_r'] = result.rvalue
                
                odf_use.loc[mask, 'linreg_p'] = result.pvalue
                
                odf_use.loc[mask, 'linreg_sB1'] = result.stderr
                
                odf_use.loc[mask, 'linreg_sB0'] = result.intercept_stderr
                
                # length of the original dataset
                n = len(x)
                # degrees of freedom
                dof = n - 2
                
                # t-value for alpha/2 with n-2 degrees of freedom
                t = stats.t.ppf(1-alpha/2, dof)
                
                slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
        
                odf_use.loc[mask, 'slope_var_datetime'] = slope_datetime #per year
                
# %%

odf_val_at_min_z = odf[odf['var'] == 'DO_mg_L'].loc[odf[odf['var'] == 'DO_mg_L'].groupby('cid')['z'].idxmin()].dropna()

odf_val_at_min_z['-h'] = odf_val_at_min_z['h']*-1



fig, ax = plt.subplots(figsize=(5,5))

sns.scatterplot(data=odf_val_at_min_z, x = '-h', y='z', alpha=0.2)

#sns.scatterplot(data=odf_val_at_min_z[odf_val_at_min_z['val'] < 2], x = '-h', y='z', color='red', alpha=0.2)



ax.set_xlabel('water column depth [m]')

ax.set_ylabel('cast depth [m]')

#ax.set_ylim(-,18)

#ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)



plt.savefig('/Users/dakotamascarenas/Desktop/pltz/test.png', bbox_inches='tight', dpi=500)

# %%

# plot with surf T

summer_deep_hyp_sites = odf[(odf['var'] == 'DO_mg_L') & (odf['val'] < 2) & (odf['surf_deep'] == 'deep') & (odf['summer_non_summer'] == 'summer')]['site'].unique()

dec_summer_deep_DO_sites = odf_use[(odf_use['linreg_p'] < alpha) & (odf_use['slope_var_datetime'] <0) & (odf_use['var'] == 'DO_mg_L') & (odf_use['summer_non_summer'] == 'summer') & (odf_use['surf_deep'] == 'deep')]['site'].unique()

inc_summer_surf_CT_sites = odf_use[(odf_use['linreg_p'] < alpha) & (odf_use['slope_var_datetime'] >0) & (odf_use['var'] == 'CT') & (odf_use['summer_non_summer'] == 'summer') & (odf_use['surf_deep'] == 'surf')]['site'].unique()

inc_summer_strat_sites = odf_dens_use[(odf_dens_use['linreg_p'] < alpha) & (odf_dens_use['slope_var_datetime'] >0) & (odf_dens_use['summer_non_summer'] == 'summer')]['site'].unique()

# %%

fig, axes = plt.subplots(figsize=(16,6), ncols=5, sharey=True, sharex=True)

c = 0

for ax in axes:

    ax.pcolormesh(plon, plat, zm, linewidth=0.5, vmin=-100, vmax=0, cmap=plt.get_cmap(cmocean.cm.ice))
    
    ax.set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
    ax.set_ylim(Y[j1],Y[j2]) # Salish Sea

    for basin in big_basin_list:
        
        if basin == 'hc':
            
            color = 'orange'
        
        elif basin == 'wb':
            
            color = 'red'
            
        elif basin == 'ss':
            
            color = 'purple'
            
        else:
            
            color = 'blue'
        
        path = path_dict[basin]
        
        patch = patches.PathPatch(path, facecolor='none', edgecolor ='k', alpha=0.5)
        
        ax.add_patch(patch)
    
    
    ax.pcolormesh(plon, plat, zm_inverse, linewidth=0.5, vmin=-1000, vmax=0, cmap = 'gray')
    
    for site in long_site_list:
        
        path = path_dict[site]
        
        if c == 0:
            
            if site == 'point_jefferson':
            
        
                patch = patches.PathPatch(path, facecolor='lightgray', alpha=0.5, edgecolor='black', label='>60-year history', hatch = '///')
                
            else:
                
                patch = patches.PathPatch(path, facecolor='lightgray', alpha=0.5, edgecolor='black', hatch = '///')
            
            
        else: 
            
            patch = patches.PathPatch(path, facecolor='lightgray', alpha=0.5, edgecolor='gray', hatch ='///')
            
        if c == 1:
            
            if site in summer_deep_hyp_sites: 
                        
                patch = patches.PathPatch(path, facecolor='red', alpha=0.5, edgecolor='red', hatch = '///')
                
        elif c == 2:
            
            if site in dec_summer_deep_DO_sites:
                
                patch = patches.PathPatch(path, facecolor='orange', alpha=0.5, edgecolor='orange', hatch = '///')
                
        elif c == 3: 
            
            if site in inc_summer_surf_CT_sites:
                
                patch = patches.PathPatch(path, facecolor='magenta', alpha=0.5, edgecolor='magenta', hatch = '///')
                
        elif c == 4:
            
            if site in inc_summer_surf_CT_sites:
                
                patch = patches.PathPatch(path, facecolor='purple', alpha=0.5, edgecolor='purple', hatch = '///')
                
        ax.add_patch(patch)
    
    
    lat_lon_df = odf[(odf['site'].isin(short_site_list))].groupby('site').first().reset_index()
    
    for site in short_site_list:
        
        if c == 0:
            
            if site == 'KSBP01':
            
                sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='lightgray', alpha=0.5, edgecolor='black', ax = ax, s=30,  label='~20-year history')
            
            else:
                
                sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='lightgray', alpha=0.5, edgecolor='black', ax = ax,  s=30)
        
        else:
            
            sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='lightgray', alpha=0.5, edgecolor='gray', ax = ax,  s=30)
     
        if c == 1:
            
            if site in summer_deep_hyp_sites: 
                
                if site == 'HCB007':
                        
                    sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='red', alpha=0.5, edgecolor='red', ax = ax,  s=30, label = 'deep hypoxia')
                    
                else:
                    
                    sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='red', alpha=0.5, edgecolor='red', ax = ax,  s=30)
                
        elif c == 2:
            
            if site in dec_summer_deep_DO_sites:
                
                if site == 'OAK004':
                
                    sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='orange', alpha=0.5, edgecolor='orange', ax = ax,  s=30, label = 'decreasing deep DO')
                    
                else:
                    
                    sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='orange', alpha=0.5, edgecolor='orange', ax = ax, s=30)
                
        elif c == 3: 
            
            if site in inc_summer_surf_CT_sites:
                
                if site == 'ADM003':
                
                    sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='magenta', alpha=0.5, edgecolor='magenta', ax = ax,  s=30, label = 'surface warming')
                    
                else:
                    
                    sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='magenta', alpha=0.5, edgecolor='magenta', ax = ax, s=30)
                
        elif c == 4:
            
            if site in inc_summer_strat_sites:
                                
                
                if site == 'HCB007':
            
                    sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='purple', alpha=0.5, edgecolor='purple', ax = ax,  s=30, label = 'increasing stratification')
        
                
                else:
                
                    sns.scatterplot(data=lat_lon_df[lat_lon_df['site'] == site], x='lon', y='lat', facecolor='purple', alpha=0.5, edgecolor='purple', ax = ax,  s=30)


    pfun.add_coast(ax) 
    
    pfun.dar(ax)
    
    ax.set_xlim(-123.3, -122.1)
    
    ax.set_ylim(47,48.5)
    
    ax.set(xlabel=None)
     
    ax.set(ylabel=None)
    
    ax.tick_params(axis='x', labelrotation=45)
    
    ax.legend(loc='upper left')
    
    c+=1

    



plt.savefig('/Users/dakotamascarenas/Desktop/pltz/all_trends.png', bbox_inches='tight', dpi=500)

# %%

for site in ['point_jefferson']:
    
    mosaic = []
        
    for var in ['DO_mg_L', 'CT', 'SA']:
        
        new_list = []
        
        for season in ['summer']:
            
            new_list.append(var + '_' + season)
                            
        mosaic.append(new_list)
        
    mosaic.append(['strat_sigma_summer'])
        
        
    fig, ax = plt.subplot_mosaic(mosaic, layout='constrained', figsize = (8,8))
    
    for var in var_list:
            
        if var =='SA':
                    
            marker = 's'
            
            ymin = 15
            
            ymax = 35
            
            label = 'Salinity [PSU]'
                        
            color_deep = 'blue'
            
            color_surf = 'lightblue'
            
                    
        elif var == 'CT':
            
            marker = '^'
            
            ymin = 4
            
            ymax = 22
            
            label = 'Temperature [deg C]'
                        
            color_deep = 'red'
            
            color_surf = 'pink'
            
        else:
            
            marker = 'o'
            
            ymin = 0
            
            ymax = 15
            
            color = 'black'
            
            label = 'DO [mg/L]'
            
            color_deep = 'black'
            
            color_surf = 'gray'
            
        colors = {'deep':color_deep, 'surf':color_surf}
    
    
    for season in ['summer']:
        
        ax_name = 'strat_sigma_' + season
        
        plot_df = odf_dens_use[(odf_dens_use['site'] == site) & (odf_dens_use['summer_non_summer'] == season)]
                
        sns.scatterplot(data=plot_df, x='datetime', y ='strat_sigma_mean', color = 'green', ax=ax[ax_name], alpha=0.7, legend = False)
                    
        for idx in plot_df.index:
            
            ax[ax_name].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
            
            
            
        x = plot_df['date_ordinal']
        
        x_plot = plot_df['datetime']
        
        y = plot_df['strat_sigma_mean']
        
        p = plot_df['linreg_p'].unique()[0]
        
        B0 = plot_df['linreg_B0'].unique()[0]
        
        B1 = plot_df['linreg_B1'].unique()[0]
        
        slope_datetime = plot_df['slope_var_datetime'].unique()[0]


        
        if p <= alpha:
            
            color = 'green'
        
            ax[ax_name].plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=color, alpha = 0.7)
        
            slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)

        
            ax[ax_name].text(0.99,0.99, 'slope = ' + str(np.round(slope_datetime,3)) + ' /yr', horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=color)
        
        
        
        ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax[ax_name].set_ylabel('Strat ~ Surf-Deep [sigma]')
        
        ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2024,12,31)])
        
        ax[ax_name].set_xlabel('Year')
        
        ax[ax_name].set_ylim(-5,10)
        
        
       
    
    for var in ['DO_mg_L', 'CT', 'SA']:
            
        if var =='SA':
                    
            marker = 's'
            
            ymin = 15
            
            ymax = 35
            
            label = 'Salinity [PSU]'
                        
            color_deep = 'blue'
            
            color_surf = 'lightblue'
                    
        elif var == 'CT':
            
            marker = '^'
            
            ymin = 4
            
            ymax = 22
            
            label = 'Temperature [deg C]'
                        
            color_deep = 'red'
            
            color_surf = 'pink'
            
        else:
            
            marker = 'o'
            
            ymin = 0
            
            ymax = 15
            
            color = 'black'
            
            label = 'DO [mg/L]'
            
            color_deep = 'black'
            
            color_surf = 'gray'
            
        colors = {'deep':color_deep, 'surf':color_surf}
        
        
        
        for season in ['summer']:
            
            ax_name = var + '_' + season
                                            
            plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == var) & (odf_use['summer_non_summer'] == season)]
                    
            sns.scatterplot(data=plot_df, x='datetime', y ='val_mean', hue = 'surf_deep', palette=colors, ax=ax[ax_name], alpha=0.7, legend = False)
                        
            for idx in plot_df.index:
                
                ax[ax_name].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
            
            
            for depth in ['surf', 'deep']:
            
                x = plot_df[plot_df['surf_deep'] == depth]['date_ordinal']
                
                x_plot = plot_df[plot_df['surf_deep'] == depth]['datetime']
                
                y = plot_df[plot_df['surf_deep'] == depth]['val_mean']
            
                
            
                p = plot_df[plot_df['surf_deep'] == depth]['linreg_p'].unique()[0]
                
                B0 = plot_df[plot_df['surf_deep'] == depth]['linreg_B0'].unique()[0]
                
                B1 = plot_df[plot_df['surf_deep'] == depth]['linreg_B1'].unique()[0]
                
                slope_datetime = plot_df[plot_df['surf_deep'] == depth]['slope_var_datetime'].unique()[0]
                
                
                if p <= alpha:
                    
                    color = colors[depth]
                    
                    ax[ax_name].plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=color, alpha =0.7)
                
                    slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)

                 
                    if depth == 'surf':
                        
                        y = 0.99
                         
                    else:
                        
                        y =0.90
                        
                    
                    ax[ax_name].text(0.99,y, 'slope = ' + str(np.round(slope_datetime,3)) + ' /yr', horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=color)
            
            
                                
            ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[ax_name].set_ylabel(label)
            
            ax[ax_name].set_ylim(ymin,ymax)
            
        
            if var == 'DO_mg_L':
                
                ax[ax_name].axhspan(0,2, color = 'lightgray', alpha = 0.2)
                
            ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2024,12,31)])
            
            ax[ax_name].set_xlabel('Year')
            
         
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_surf_deep_decadal.png', bbox_inches='tight', dpi=500)
    



# %%



fig, ax = plt.subplots(figsize=(5,5))

plot_df= odf_use[(odf_use['site'].isin(short_site_list)) & (odf_use['summer_non_summer'] == 'summer')]

plot_df = plot_df.pivot(index=['site', 'year', 'date_ordinal'], columns = ['surf_deep', 'var'], values= 'val_mean')

plot_df.columns = plot_df.columns.to_flat_index().map('_'.join)

plot_df = plot_df.reset_index()


sns.scatterplot(data = plot_df, x='surf_CT', y='deep_DO_mg_L')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

ax.set_ylabel('Deep DO [mg/L]')

ax.set_xlabel('Surface Temperature [deg C]')


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/short_surfCT_v_deepDO.png', bbox_inches='tight', dpi=500)

