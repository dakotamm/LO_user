#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:59:49 2024

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


# %%

poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] # 5 sites

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

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

    fnp = Ldir['LOo'] / 'section_lines' / (poly+'.p')
    p = pd.read_pickle(fnp)
    xx = p.x.to_numpy()
    yy = p.y.to_numpy()
    xxyy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
    path = mpth.Path(xxyy)
    
    oisin = path.contains_points(oxoy)
    
    this_depths = depths.flatten()[oisin]
    
    max_depth = np.nanmax(this_depths)
    
    max_depths_dict[poly] = max_depth.copy()
    
# %%
# %%
# %%
# DECADAL

odf_decadal = odf.copy()


for basin in basin_list:
    
    odf_decadal.loc[odf['segment'] == basin, 'min_segment_h'] = -max_depths_dict[basin]

# %%

summer_mask = (odf_decadal['yearday'] > 125) & (odf_decadal['yearday']<= 325)

surf_mask = (odf_decadal['z'] > -5)

deep_non_lc_nso_mask = (odf_decadal['z'] < 0.8*odf_decadal['min_segment_h']) & (odf_decadal['segment'] != 'lynch_cove_mid') & (odf_decadal['segment'] != 'near_seattle_offshore')

deep_lc_mask = (odf_decadal['z'] < 0.4*odf_decadal['min_segment_h']) & (odf_decadal['segment'] == 'lynch_cove_mid')

deep_nso_mask = (odf_decadal['z'] < 0.75*odf_decadal['min_segment_h']) & (odf_decadal['segment'] == 'near_seattle_offshore') #CHANGED 5/21/2024


# %%

odf_decadal.loc[summer_mask, 'summer_non_summer'] = 'summer'

odf_decadal.loc[~summer_mask, 'summer_non_summer'] = 'non_summer'

odf_decadal.loc[surf_mask, 'surf_deep'] = 'surf'

odf_decadal.loc[deep_non_lc_nso_mask, 'surf_deep'] = 'deep'

odf_decadal.loc[deep_lc_mask, 'surf_deep'] = 'deep'

odf_decadal.loc[deep_nso_mask, 'surf_deep'] = 'deep'

# %%

temp0 = odf_decadal[odf_decadal['surf_deep'] != 'nan']

# %%

odf_decadal_depth_mean = temp0.groupby(['segment','surf_deep', 'summer_non_summer', 'year', 'var','cid']).mean(numeric_only=True).reset_index().dropna()


# %%

odf_decadal_depth_mean_0 = odf_decadal_depth_mean.copy()

cid_deep = odf_decadal_depth_mean_0.loc[odf_decadal_depth_mean_0['surf_deep'] == 'deep', 'cid']

# %%

odf_decadal_depth_mean_0 = odf_decadal_depth_mean_0[odf_decadal_depth_mean_0['cid'].isin(cid_deep)]

# %%

odf_dens = odf_decadal_depth_mean_0.pivot(index = ['segment', 'year', 'summer_non_summer', 'date_ordinal', 'cid'], columns = ['surf_deep', 'var'], values ='val')

# %%

odf_dens.columns = odf_dens.columns.to_flat_index().map('_'.join)

odf_dens = odf_dens.reset_index()

# %%

odf_dens['surf_dens'] = gsw.density.sigma0(odf_dens['surf_SA'], odf_dens['surf_CT'])

odf_dens['deep_dens'] = gsw.density.sigma0(odf_dens['deep_SA'], odf_dens['deep_CT'])

# %%

odf_dens['strat_sigma'] = odf_dens['deep_dens'] - odf_dens['surf_dens']

# %%

odf_DO_pred = odf_dens.copy()

# %%

A_0 = 5.80818 #all in umol/kg

A_1 = 3.20684

A_2 = 4.11890

A_3 = 4.93845

A_4 = 1.01567

A_5 = 1.41575

B_0 = -7.01211e-3

B_1 = -7.25958e-3

B_2 = -7.93334e-3

B_3 = -5.54491e-3

C_0 = -1.32412e-7

# %%

odf_DO_pred['T_s'] = np.log((298.15 - odf_DO_pred['surf_CT'])/(273.15 - odf_DO_pred['surf_CT']))

odf_DO_pred['C_o_*'] = np.exp(A_0 + A_1*odf_DO_pred['T_s'] + A_2*odf_DO_pred['T_s']**2 + A_3*odf_DO_pred['T_s']**3 + A_4*odf_DO_pred['T_s']**4 + A_5*odf_DO_pred['T_s']**5 + 
                       odf_DO_pred['surf_SA']*(B_0 + B_1*odf_DO_pred['T_s'] + B_2*odf_DO_pred['T_s']**2 + B_3*odf_DO_pred['T_s']**3) + C_0*odf_DO_pred['surf_SA']**2)

odf_DO_pred['DO_sol'] =  (odf_DO_pred['surf_dens']/1000 + 1)*32/1000

# %%

odf_dens = odf_dens[['segment', 'year','summer_non_summer', 'date_ordinal', 'cid', 'strat_sigma']]

# %%

odf_DO_pred = odf_DO_pred[['segment', 'year','summer_non_summer', 'date_ordinal', 'cid', 'DO_sol']]




# %%

annual_counts = (odf_decadal_depth_mean
                      .dropna()
                      #.set_index('datetime')
                      .groupby(['segment','year','summer_non_summer', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )

# %%

odf_decadal_use = odf_decadal_depth_mean.groupby(['segment', 'surf_deep', 'summer_non_summer', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})

# %%

odf_decadal_use.columns = odf_decadal_use.columns.to_flat_index().map('_'.join)

odf_decadal_use = odf_decadal_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# %%

odf_decadal_use = (odf_decadal_use
                  # .drop(columns=['date_ordinal_std'])
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .reset_index() 
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

odf_decadal_use = pd.merge(odf_decadal_use, annual_counts, how='left', on=['segment','surf_deep','summer_non_summer','year','var'])

# %%

odf_decadal_use = odf_decadal_use[odf_decadal_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_decadal_use['val_ci95hi'] = odf_decadal_use['val_mean'] + 1.96*odf_decadal_use['val_std']/np.sqrt(odf_decadal_use['cid_count'])

odf_decadal_use['val_ci95lo'] = odf_decadal_use['val_mean'] - 1.96*odf_decadal_use['val_std']/np.sqrt(odf_decadal_use['cid_count'])


# %%

annual_counts_dens= (odf_decadal_depth_mean_0
                      .dropna()
                      #.set_index('datetime')
                      .groupby(['segment','year','summer_non_summer']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )
# %%

odf_dens_use = odf_dens.groupby(['segment', 'summer_non_summer', 'year']).agg({'strat_sigma':['mean', 'std'], 'date_ordinal':['mean']})

# %%

odf_dens_use.columns = odf_dens_use.columns.to_flat_index().map('_'.join)

odf_dens_use = odf_dens_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# %%

odf_dens_use = (odf_dens_use
                  # .drop(columns=['date_ordinal_std'])
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .reset_index() 
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

odf_dens_use = pd.merge(odf_dens_use, annual_counts_dens, how='left', on=['segment','summer_non_summer','year'])

# %%

odf_dens_use = odf_dens_use[odf_dens_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_dens_use['val_ci95hi'] = odf_dens_use['strat_sigma_mean'] + 1.96*odf_dens_use['strat_sigma_std']/np.sqrt(odf_dens_use['cid_count'])

odf_dens_use['val_ci95lo'] = odf_dens_use['strat_sigma_mean'] - 1.96*odf_dens_use['strat_sigma_std']/np.sqrt(odf_dens_use['cid_count'])


# %%

annual_counts_sol= (odf_decadal_depth_mean_0
                      .dropna()
                      #.set_index('datetime')
                      .groupby(['segment','year','summer_non_summer']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )
# %%

odf_DO_pred_use = odf_DO_pred.groupby(['segment', 'summer_non_summer', 'year']).agg({'DO_sol':['mean', 'std'], 'date_ordinal':['mean']})

# %%

odf_DO_pred_use.columns = odf_DO_pred_use.columns.to_flat_index().map('_'.join)

odf_DO_pred_use = odf_DO_pred_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# %%

odf_DO_pred_use = (odf_DO_pred_use
                  # .drop(columns=['date_ordinal_std'])
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .reset_index() 
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

odf_DO_pred_use = pd.merge(odf_DO_pred_use, annual_counts_sol, how='left', on=['segment','summer_non_summer','year'])

# %%

odf_DO_pred_use = odf_DO_pred_use[odf_DO_pred_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_DO_pred_use['val_ci95hi'] = odf_DO_pred_use['DO_sol_mean'] + 1.96*odf_DO_pred_use['DO_sol_std']/np.sqrt(odf_DO_pred_use['cid_count'])

odf_DO_pred_use['val_ci95lo'] = odf_DO_pred_use['DO_sol_mean'] - 1.96*odf_DO_pred_use['DO_sol_std']/np.sqrt(odf_DO_pred_use['cid_count'])

# %%

# %%
for site in basin_list:
    
    
    mosaic = [['strat_sigma_non_summer', 'strat_sigma_summer'],
              ['DO_sol_non_summer', 'DO_sol_summer']]
    
    c=0
    
    for var in var_list:
        
        new_list = []
        
        for season in ['non_summer','summer']:
            
            new_list.append(var + '_' + season)
                            
        mosaic.append(new_list)
        
        c+=1
        
    fig, ax = plt.subplot_mosaic(mosaic, layout='constrained', figsize = (24,16))
    
    
    for season in ['non_summer', 'summer']:
        
        ax_name = 'strat_sigma_' + season
        
        plot_df = odf_dens_use[(odf_dens_use['segment'] == site) & (odf_dens_use['summer_non_summer'] == season)]
                
        sns.scatterplot(data=plot_df, x='datetime', y ='strat_sigma_mean', color = 'green', ax=ax[ax_name], alpha=0.7, legend = False)
                    
        for idx in plot_df.index:
            
            ax[ax_name].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
            
            
            
        x = plot_df['date_ordinal']
        
        x_plot = plot_df['datetime']
        
        y = plot_df['strat_sigma_mean']
        
        result = stats.linregress(x, y)
        
        B1 = result.slope
        
        B0 = result.intercept
        
        r = result.rvalue
        
        p = result.pvalue
        
        sB1 = result.stderr
        
        sB0 = result.intercept_stderr
        
        # our alpha for 95% confidence
        alpha = 0.05
        
        # length of the original dataset
        n = len(x)
        # degrees of freedom
        dof = n - 2
        
        # t-value for alpha/2 with n-2 degrees of freedom
        t = stats.t.ppf(1-alpha/2, dof)
        
        # # compute the upper and lower limits on our B1 (slope) parameter
        # B1_upper = B1 + t * sB1
        # B1_lower = B1 - t * sB1
        
        # # compute the corresponding upper and lower B0 values (y intercepts)
        # B0_upper = y.mean() - B1_upper*x.mean()
        # B0_lower = y.mean() - B1_lower*x.mean()
        
        # # an array of x values
        # p_x = np.linspace(x.min(),x.max(),100)
        
        # # using our model parameters to predict y values
        # p_y = B0 + B1*p_x
        
        # # compute the upper and lower limits at each of the p_x values
        # p_y_lower = p_y - t * sB0
        # p_y_upper = p_y + t * sB0
        
        # # Plot the mean line, we only need two points to define a line, use xmin and xmax
        # ax[ax_name].plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
        
        # # Plot the mean x line
        # ax[ax_name].axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
        
        if p >= alpha:
            
            color = 'k'
            
        else:
            
            color = 'm'
        
        # Plot the linear regression model
        ax[ax_name].plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=color, alpha = 0.7)
        
        slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
        
        # # Plot the upper and lower confidence limits for the standard error of the gradient (slope)
        # ax[ax_name].plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
        # ax[ax_name].plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
        
        # # Plot confidence limits on our predicted Y values
        # ax[ax_name].plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
        # ax[ax_name].plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
        
        ax[ax_name].text(1,1, 'p = ' + str(np.round(p, 3)) + ', slope = ' + str(np.round(slope_datetime,3)) + ' /yr', horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=color)
                            
        
        
        
        ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax[ax_name].set_ylabel('Strat ~ Surf-Deep [sigma]')
        
        ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2030,12,31)])
        
        ax[ax_name].set_xlabel('Year')
        
        ax[ax_name].set_ylim(-5,10)
        
        
        
        
        
        
        ax_name = 'DO_sol_' + season
        
        plot_df = odf_DO_pred_use[(odf_DO_pred_use['segment'] == site) & (odf_DO_pred_use['summer_non_summer'] == season)]
                
        sns.scatterplot(data=plot_df, x='datetime', y ='DO_sol_mean', color = 'purple', ax=ax[ax_name], alpha=0.7, legend = False)
                    
        for idx in plot_df.index:
            
            ax[ax_name].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
            
            
            
        x = plot_df['date_ordinal']
        
        x_plot = plot_df['datetime']
        
        y = plot_df['DO_sol_mean']
        
        result = stats.linregress(x, y)
        
        B1 = result.slope
        
        B0 = result.intercept
        
        r = result.rvalue
        
        p = result.pvalue
        
        sB1 = result.stderr
        
        sB0 = result.intercept_stderr
        
        # our alpha for 95% confidence
        alpha = 0.05
        
        # length of the original dataset
        n = len(x)
        # degrees of freedom
        dof = n - 2
        
        # t-value for alpha/2 with n-2 degrees of freedom
        t = stats.t.ppf(1-alpha/2, dof)
        
        # # compute the upper and lower limits on our B1 (slope) parameter
        # B1_upper = B1 + t * sB1
        # B1_lower = B1 - t * sB1
        
        # # compute the corresponding upper and lower B0 values (y intercepts)
        # B0_upper = y.mean() - B1_upper*x.mean()
        # B0_lower = y.mean() - B1_lower*x.mean()
        
        # # an array of x values
        # p_x = np.linspace(x.min(),x.max(),100)
        
        # # using our model parameters to predict y values
        # p_y = B0 + B1*p_x
        
        # # compute the upper and lower limits at each of the p_x values
        # p_y_lower = p_y - t * sB0
        # p_y_upper = p_y + t * sB0
        
        # # Plot the mean line, we only need two points to define a line, use xmin and xmax
        # ax[ax_name].plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
        
        # # Plot the mean x line
        # ax[ax_name].axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
        
        if p >= alpha:
            
            color = 'gray'
            
        else:
            
            color = 'm'
        
        # Plot the linear regression model
        ax[ax_name].plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=color, alpha = 0.7)
        
        slope_datetime = (B0 + B1*x.max() - B0 + B1*x.min())/(x_plot.max().year - x_plot.min().year)
        
        # # Plot the upper and lower confidence limits for the standard error of the gradient (slope)
        # ax[ax_name].plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
        # ax[ax_name].plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
        
        # # Plot confidence limits on our predicted Y values
        # ax[ax_name].plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
        # ax[ax_name].plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')
        
        ax[ax_name].text(1,1, 'p = ' + str(np.round(p, 3)) + ', slope = ' + str(np.round(slope_datetime,3)) + ' /yr', horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=color)
                            
        
        
        
        ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax[ax_name].set_ylabel('DO Sol [mg/L]')
        
        ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2030,12,31)])
        
        ax[ax_name].set_xlabel('Year')
        
        ax[ax_name].set_ylim(0,1)
        
        
       
            
    
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
        
        
        
        for season in ['non_summer', 'summer']:
            
            ax_name = var + '_' + season
                                            
            plot_df = odf_decadal_use[(odf_decadal_use['segment'] == site) & (odf_decadal_use['var'] == var) & (odf_decadal_use['summer_non_summer'] == season)]
                    
            sns.scatterplot(data=plot_df, x='datetime', y ='val_mean', hue = 'surf_deep', palette=colors, ax=ax[ax_name], alpha=0.7, legend = False)
                        
            for idx in plot_df.index:
                
                ax[ax_name].plot([plot_df.loc[idx,'datetime'], plot_df.loc[idx,'datetime']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color='lightgray', alpha =0.7)
            
            for depth in ['surf', 'deep']:
            
                x = plot_df[plot_df['surf_deep'] == depth]['date_ordinal']
                
                x_plot = plot_df[plot_df['surf_deep'] == depth]['datetime']
                
                y = plot_df[plot_df['surf_deep'] == depth]['val_mean']
                
                result = stats.linregress(x, y)
                
                B1 = result.slope
                
                B0 = result.intercept
                
                r = result.rvalue
                
                p = result.pvalue
                
                sB1 = result.stderr
                
                sB0 = result.intercept_stderr
                
                # our alpha for 95% confidence
                alpha = 0.05
                
                # length of the original dataset
                n = len(x)
                # degrees of freedom
                dof = n - 2
                
                # t-value for alpha/2 with n-2 degrees of freedom
                t = stats.t.ppf(1-alpha/2, dof)
                
                # # compute the upper and lower limits on our B1 (slope) parameter
                # B1_upper = B1 + t * sB1
                # B1_lower = B1 - t * sB1
                
                # # compute the corresponding upper and lower B0 values (y intercepts)
                # B0_upper = y.mean() - B1_upper*x.mean()
                # B0_lower = y.mean() - B1_lower*x.mean()
                
                # # an array of x values
                # p_x = np.linspace(x.min(),x.max(),100)
                
                # # using our model parameters to predict y values
                # p_y = B0 + B1*p_x
                
                # # compute the upper and lower limits at each of the p_x values
                # p_y_lower = p_y - t * sB0
                # p_y_upper = p_y + t * sB0
                
                # # Plot the mean line, we only need two points to define a line, use xmin and xmax
                # ax[ax_name].plot([x.min(), x.max()], [y.mean(), y.mean()] , '--m', label='Mean Y')
                
                # # Plot the mean x line
                # ax[ax_name].axvline(x.mean(),c='k', linestyle='--', label='Mean X Value')
                
                if p >= alpha:
                    
                    color = 'gray'
                    
                    
                else:
                    
                    color = 'm'
                
                # Plot the linear regression model
                ax[ax_name].plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=color, alpha =0.7)
                
                slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
                
                # # Plot the upper and lower confidence limits for the standard error of the gradient (slope)
                # ax[ax_name].plot([x.min(), x.max()], [B0_upper + B1_upper*x.min(), B0_upper + B1_upper*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
                # ax[ax_name].plot([x.min(), x.max()], [B0_lower + B1_lower*x.min(), B0_lower + B1_lower*x.max()] , '--r', label='Upper B0 confidence limit (95%)')
                
                # # Plot confidence limits on our predicted Y values
                # ax[ax_name].plot(p_x, p_y_upper, ':b', label='Upper Y prediction interval (95%)')
                # ax[ax_name].plot(p_x, p_y_lower, ':b', label='Lower Y prediction interval (95%)')                    

                
                
                    
                if depth == 'surf':
                    
                    y = 1
                    
                else:
                    
                    y =0.95
                
                ax[ax_name].text(1,y, 'p ' + str(depth) + ' = ' + str(np.round(p, 3))+ ', slope = ' + str(np.round(slope_datetime,3)) + ' /yr', horizontalalignment='right', verticalalignment='top', transform=ax[ax_name].transAxes, color=color)
            
            
                                
            ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[ax_name].set_ylabel(label)
            
            ax[ax_name].set_ylim(ymin,ymax)
            
            #ax[ax_name].set_title(site + ' ' + season)
        
            if var == 'DO_mg_L':
                
                ax[ax_name].axhspan(0,2, color = 'lightgray', alpha = 0.2)
                
            ax[ax_name].set_xlim([datetime.date(1930,1,1), datetime.date(2030,12,31)])
            
            ax[ax_name].set_xlabel('Year')
            
         
    
    ax['strat_sigma_non_summer'].set_title('Non-Summer')
    
    ax['strat_sigma_summer'].set_title('Summer')
    
    #ax['strat_psu_non_summer'].text(0.01, 0.99, 'max depth = ' + str(np.round(max_depths_dict[site])), horizontalalignment='left', verticalalignment='top', transform=ax[ax_name].transAxes)
    
    
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_wDOsol_surf_deep_decadal.png', bbox_inches='tight', dpi=500)





