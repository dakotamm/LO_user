#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:57:47 2024

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


poly_list = ['mb', 'wb', 'ss', 'hc'] #,'admiralty_sill', 'budd_inlet', 'carr_inlet_mid', 'dana_passage', 'hat_island', 'hazel_point', 'hood_canal_mouth', 'lynch_cove_mid', 'near_seattle_offshore', 'near_edmonds', 'port_susan_mid', 'saratoga_passage_north', 'saratoga_passage_mid']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['ecology_nc'], otype_list=['bottle', 'ctd'], year_list=np.arange(1998,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'DO_mg_L'] #, 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']

# %%

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                      .assign(
                          datetime=(lambda x: pd.to_datetime(x['time'])),
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

station_list = odf['name'].unique()


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

odf_annual = odf.copy()


# %%

summer_mask = (odf_annual['yearday'] > 125) & (odf_annual['yearday']<= 325)

surf_mask = (odf_annual['z'] > -5)

deep_mask_no_lc = (odf_annual['z'] < 0.8*odf_annual['h']) & (odf_annual['name'] != 'HCB007')

deep_mask_lc = (odf_annual['z'] < 0.4*odf_annual['h']) & (odf_annual['name'] == 'HCB007')


# %%

odf_annual.loc[summer_mask, 'summer_non_summer'] = 'summer'

odf_annual.loc[~summer_mask, 'summer_non_summer'] = 'non_summer'

odf_annual.loc[surf_mask, 'surf_deep'] = 'surf'

odf_annual.loc[deep_mask_no_lc, 'surf_deep'] = 'deep'

odf_annual.loc[deep_mask_lc, 'surf_deep'] = 'deep'


# %%

temp0 = odf_annual[odf_annual['surf_deep'] != 'nan']

# %%

odf_annual_depth_mean = temp0.groupby(['segment', 'name','surf_deep', 'summer_non_summer', 'year', 'var','cid']).mean(numeric_only=True).reset_index().dropna()

# %%

# annual_counts = (temp1
#                       .dropna()
#                       #.set_index('datetime')
#                       .groupby(['segment','year','summer_non_summer', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
#                       .reset_index()
#                       .rename(columns={'cid':'cid_count'})
#                       )


# %%

odf_annual_depth_mean = (odf_annual_depth_mean
                  # .drop(columns=['date_ordinal_std'])
                  # .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  # .reset_index() 
                  # .dropna()
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

odf_annual_depth_mean_0 = odf_annual_depth_mean.copy()

cid_deep = odf_annual_depth_mean_0.loc[odf_annual_depth_mean_0['surf_deep'] == 'deep', 'cid']

# %%

odf_annual_depth_mean_0 = odf_annual_depth_mean_0[odf_annual_depth_mean_0['cid'].isin(cid_deep)]

# %%

odf_dens = odf_annual_depth_mean_0.pivot(index = ['segment', 'name', 'year', 'summer_non_summer', 'date_ordinal', 'cid'], columns = ['surf_deep', 'var'], values ='val')

# %%

odf_dens.columns = odf_dens.columns.to_flat_index().map('_'.join)

odf_dens = odf_dens.reset_index()

# %%

odf_dens['surf_dens'] = gsw.density.sigma0(odf_dens['surf_SA'], odf_dens['surf_CT'])

odf_dens['deep_dens'] = gsw.density.sigma0(odf_dens['deep_SA'], odf_dens['deep_CT'])

# %%

odf_dens['strat_sigma'] = odf_dens['deep_dens'] - odf_dens['surf_dens']

# %%


short_sites = ['QMH002', 'PMA001', 'OCH014', 'DYE004', 'SUZ001', 'HLM001', 'PNN001', 'PSS010', 'TOT002', 'TOT001', 'HND001','ELD001', 'ELD002', 'CSE002', 'CSE001', 'HCB010', 'SKG003','HCB006', 'HCB008', 'HCB009']

odf_annual_depth_mean = odf_annual_depth_mean[~odf_annual_depth_mean['name'].isin(short_sites)]

sites = odf_annual_depth_mean['name'].unique()

for var in ['DO_mg_L']:
    
    mosaic = [['map']]
    
    fig, ax = plt.subplot_mosaic(mosaic, layout='constrained', figsize = (4,8))
            
    plot_df_map = odf_annual_depth_mean[odf_annual_depth_mean['var'] == var].groupby('name').first().reset_index()

    sns.scatterplot(data=plot_df_map, x='lon', y='lat', ax = ax['map'], s = 100, alpha=1)

    #sns.scatterplot(data=plot_df_mean, x='lon', y='lat', hue='decade', palette='Set2', marker='s', sizes=20)

    ax['map'].autoscale(enable=False)

    pfun.add_coast(ax['map'])

    pfun.dar(ax['map'])

    ax['map'].set_xlim(-123.2, -122.1)

    ax['map'].set_ylim(47,48.5)

    #ax['map'].set_title(var + ' TRUE MK Trends Annual, Depth and Season')

    #ax['map'].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + var + '_ecology_sampling_locations.png', bbox_inches='tight', dpi=500)


# %%

annual_counts = (odf_annual_depth_mean
                      .dropna()
                      #.set_index('datetime')
                      .groupby(['segment','name', 'year','summer_non_summer', 'surf_deep', 'var']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )

# %%

odf_annual_use = odf_annual_depth_mean.groupby(['segment', 'name', 'surf_deep', 'summer_non_summer', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})

# %%

odf_annual_use.columns = odf_annual_use.columns.to_flat_index().map('_'.join)

odf_annual_use = odf_annual_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!

# %%

odf_annual_use = (odf_annual_use
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

odf_annual_use = pd.merge(odf_annual_use, annual_counts, how='left', on=['segment','name', 'surf_deep','summer_non_summer','year','var'])

# %%

odf_annual_use = odf_annual_use[odf_annual_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_annual_use['val_ci95hi'] = odf_annual_use['val_mean'] + 1.96*odf_annual_use['val_std']/np.sqrt(odf_annual_use['cid_count'])

odf_annual_use['val_ci95lo'] = odf_annual_use['val_mean'] - 1.96*odf_annual_use['val_std']/np.sqrt(odf_annual_use['cid_count'])


# %%

annual_counts_dens = (odf_annual_depth_mean_0
                      .dropna()
                      #.set_index('datetime')
                      .groupby(['segment','name', 'year','summer_non_summer']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )
# %%

odf_dens_use = odf_dens.groupby(['segment', 'name', 'summer_non_summer', 'year']).agg({'strat_sigma':['mean', 'std'], 'date_ordinal':['mean']})

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

odf_dens_use = pd.merge(odf_dens_use, annual_counts_dens, how='left', on=['segment','name', 'summer_non_summer','year'])

# %%

odf_dens_use = odf_dens_use[odf_dens_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_dens_use['val_ci95hi'] = odf_dens_use['strat_sigma_mean'] + 1.96*odf_dens_use['strat_sigma_std']/np.sqrt(odf_dens_use['cid_count'])

odf_dens_use['val_ci95lo'] = odf_dens_use['strat_sigma_mean'] - 1.96*odf_dens_use['strat_sigma_std']/np.sqrt(odf_dens_use['cid_count'])

# %%
for site in sites:
    
    
    mosaic = [['strat_sigma_non_summer', 'strat_sigma_summer']]
    
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
        
        plot_df = odf_dens_use[(odf_dens_use['name'] == site) & (odf_dens_use['summer_non_summer'] == season)]
                
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
        
        ax[ax_name].set_xlim([datetime.date(1998,1,1), datetime.date(2024,12,31)])
        
        ax[ax_name].set_xlabel('Year')
        
        ax[ax_name].set_ylim(-5,10)
        
        
       
            
    
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
                                            
            plot_df = odf_annual_use[(odf_annual_use['name'] == site) & (odf_annual_use['var'] == var) & (odf_annual_use['summer_non_summer'] == season)]
                    
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
                    
                    color = 'k'
                    
                    
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
                
            ax[ax_name].set_xlim([datetime.date(1998,1,1), datetime.date(2024,12,31)])
            
            ax[ax_name].set_xlabel('Year')
            
         
    
    ax['strat_sigma_non_summer'].set_title('Non-Summer')
    
    ax['strat_sigma_summer'].set_title('Summer')
    
    #ax['strat_psu_non_summer'].text(0.01, 0.99, 'max depth = ' + str(np.round(max_depths_dict[site])), horizontalalignment='left', verticalalignment='top', transform=ax[ax_name].transAxes)
    
    
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + site + '_wdens_surf_deep_annual.png', bbox_inches='tight', dpi=500)