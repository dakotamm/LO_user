#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:14:48 2024

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

poly_list = ['ps']#['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson', 'mb', 'hc', 'ss', 'wb'] # 5 sites + 4 basins

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology_his', 'ecology_nc', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish', 'kc_point_jefferson'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'DO_mg_L'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']

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
                          NO3_uM=(lambda x: x['NO3 (uM)']),
                          Chl_mg_m3=(lambda x: x['Chl (mg m-3)']),
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

# big_basin_list = ['mb', 'wb', 'ss', 'hc']

# long_site_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson']


#short_mask_ecology = (odf['segment'].isin(big_basin_list)) & (odf['source'] == 'ecology_nc') & (~odf['name'].isin(short_exclude_sites)) & (odf['year'] >= 1998)

# short_mask_point_jefferson = (odf['segment'] == 'mb') & (odf['name'] =='KSBP01') & (odf['year'] >= 1998)

# long_mask = (odf['segment'].isin(long_site_list))

# # %%

# odf.loc[short_mask_ecology, 'short_long'] = 'short'

# odf.loc[short_mask_point_jefferson, 'short_long'] = 'short'

# odf.loc[long_mask, 'short_long'] = 'long'

# # %%

# odf = odf[odf['short_long'] != 'nan']

# # %%

# short_site_list = odf[odf['short_long'] == 'short']['name'].unique()

# %%

odf_ecology_nc = odf[odf['source'] == 'ecology_nc']

odf_ecology_nc['site'] = odf_ecology_nc['source']

odf['site'] = odf['segment']


odf = pd.concat([odf, odf_ecology_nc])


    

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

# long_deep_non_lc_nso_mask = (odf['z'] < 0.8*odf['min_segment_h']) & (odf['segment'] != 'lynch_cove_mid') & (odf['segment'] != 'near_seattle_offshore') & (odf['short_long'] == 'long')

# long_deep_lc_mask = (odf['z'] < 0.4*odf['min_segment_h']) & (odf['segment'] == 'lynch_cove_mid') & (odf['short_long'] == 'long')

# long_deep_nso_mask = (odf['z'] < 0.75*odf['min_segment_h']) & (odf['segment'] == 'near_seattle_offshore') & (odf['short_long'] == 'long') #CHANGED 5/21/2024


short_deep_mask = (odf['z'] < 0.8*odf['h']) #& (odf['short_long'] == 'short')

surf_mask = (odf['z'] >= -5)

# %%

odf.loc[surf_mask, 'surf_deep'] = 'surf'

# odf.loc[long_deep_non_lc_nso_mask, 'surf_deep'] = 'deep'

# odf.loc[long_deep_lc_mask, 'surf_deep'] = 'deep'

# odf.loc[long_deep_nso_mask, 'surf_deep'] = 'deep'

odf.loc[short_deep_mask, 'surf_deep'] = 'deep'

# %%

# odf.loc[odf['short_long'] == 'short', 'site'] = odf[odf['short_long'] == 'short']['name']

# odf.loc[odf['short_long'] == 'long', 'site'] = odf[odf['short_long'] == 'long']['segment']

##odf['site'] = odf['segment']

# %%

# temp = odf.groupby(['site','cid']).min().reset_index()

# cid_exclude = temp[(temp['site'].isin(['HCB005', 'HCB007', 'lynch_cove_mid'])) & (temp['z'] < -50)]['cid']

# odf = odf[~odf['cid'].isin(cid_exclude)]

# %%

#temp0 = odf[odf['surf_deep'] != 'nan']

# %%

odf_depth_mean = odf.groupby(['site', 'year', 'var','cid']).mean(numeric_only=True).reset_index().dropna() #####

# %%

# cid_deep = odf_depth_mean.loc[odf_depth_mean['surf_deep'] == 'deep', 'cid']

# # %%

# odf_depth_mean_deep = odf_depth_mean[odf_depth_mean['cid'].isin(cid_deep)]

# # %%

# odf_calc = odf_depth_mean_deep.pivot(index = ['site', 'year', 'month', 'yearday', 'date_ordinal','cid'], columns = ['surf_deep', 'var'], values ='val')

# odf_calc.columns = odf_calc.columns.to_flat_index().map('_'.join)

# odf_calc = odf_calc.reset_index()

# # %%

# odf_calc['surf_dens'] = gsw.density.sigma0(odf_calc['surf_SA'], odf_calc['surf_CT'])

# odf_calc['deep_dens'] = gsw.density.sigma0(odf_calc['deep_SA'], odf_calc['deep_CT'])

# # %%

# odf_calc['strat_sigma'] = odf_calc['deep_dens'] - odf_calc['surf_dens']

# # %%

# A_0 = 5.80818 #all in umol/kg

# A_1 = 3.20684

# A_2 = 4.11890

# A_3 = 4.93845

# A_4 = 1.01567

# A_5 = 1.41575

# B_0 = -7.01211e-3

# B_1 = -7.25958e-3

# B_2 = -7.93334e-3

# B_3 = -5.54491e-3

# C_0 = -1.32412e-7

# # %%

# odf_calc['T_s'] = np.log((298.15 - odf_calc['surf_CT'])/(273.15 + odf_calc['surf_CT']))

# odf_calc['C_o_*'] = np.exp(A_0 + A_1*odf_calc['T_s'] + A_2*odf_calc['T_s']**2 + A_3*odf_calc['T_s']**3 + A_4*odf_calc['T_s']**4 + A_5*odf_calc['T_s']**5 + 
#                        odf_calc['surf_SA']*(B_0 + B_1*odf_calc['T_s'] + B_2*odf_calc['T_s']**2 + B_3*odf_calc['T_s']**3) + C_0*odf_calc['surf_SA']**2)

# odf_calc['DO_sol'] =  odf_calc['C_o_*']*(odf_calc['surf_dens']/1000 + 1)*32/1000

# # %%

# ######
# odf_calc_long = pd.melt(odf_calc, id_vars = ['site', 'year', 'month', 'yearday','date_ordinal','cid'], value_vars=['strat_sigma', 'DO_sol'], var_name='var', value_name='val')

# # %%

# low_DO_season_start = 213 #aug1

# low_DO_season_end = 335 #nov30
# # %%

# summer_mask_odf_depth_mean = (odf_depth_mean['yearday'] >= low_DO_season_start) & (odf_depth_mean['yearday']<= low_DO_season_end)

# odf_depth_mean.loc[summer_mask_odf_depth_mean, 'summer_non_summer'] = 'summer'

# odf_depth_mean.loc[~summer_mask_odf_depth_mean, 'summer_non_summer'] = 'non_summer'



# summer_mask_odf_calc_long = (odf_calc_long['yearday'] >= low_DO_season_start) & (odf_calc_long['yearday']<= low_DO_season_end)

# odf_calc_long.loc[summer_mask_odf_calc_long, 'summer_non_summer'] = 'summer'

# odf_calc_long.loc[~summer_mask_odf_calc_long, 'summer_non_summer'] = 'non_summer'


# odf_depth_mean_deep_DO = odf_depth_mean[(odf_depth_mean['var'] == 'DO_mg_L') & (odf_depth_mean['surf_deep'] == 'deep')]

# %%


# odf_depth_mean_deep_DO_q50 = odf_depth_mean_deep_DO[['site', 'year', 'summer_non_summer','val']].groupby(['site', 'year', 'summer_non_summer']).quantile(0.5)

# odf_depth_mean_deep_DO_q50 = odf_depth_mean_deep_DO_q50.rename(columns={'val':'deep_DO_q50'})

# odf_depth_mean_deep_DO_q75 = odf_depth_mean_deep_DO[['site', 'year', 'summer_non_summer','val']].groupby(['site', 'year', 'summer_non_summer']).quantile(0.75)

# odf_depth_mean_deep_DO_q75 = odf_depth_mean_deep_DO_q75.rename(columns={'val':'deep_DO_q75'})

# odf_depth_mean_deep_DO_q25 = odf_depth_mean_deep_DO[['site', 'year', 'summer_non_summer','val']].groupby(['site', 'year', 'summer_non_summer']).quantile(0.25)

# odf_depth_mean_deep_DO_q25 = odf_depth_mean_deep_DO_q25.rename(columns={'val':'deep_DO_q25'})


# odf_depth_mean_deep_DO_percentiles = pd.merge(odf_depth_mean_deep_DO, odf_depth_mean_deep_DO_q75, how='left', on=['site','summer_non_summer','year'])

# odf_depth_mean_deep_DO_percentiles = pd.merge(odf_depth_mean_deep_DO_percentiles, odf_depth_mean_deep_DO_q50, how='left', on=['site','summer_non_summer','year'])

# odf_depth_mean_deep_DO_percentiles = pd.merge(odf_depth_mean_deep_DO_percentiles, odf_depth_mean_deep_DO_q25, how='left', on=['site','summer_non_summer','year'])

# %%

annual_counts = (odf
                      .dropna()
                      .groupby(['site','year', 'var']).agg({'cid' :lambda x: x.nunique()})
                      .reset_index()
                      .rename(columns={'cid':'cid_count'})
                      )

odf_use= odf.groupby(['site', 'year','var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})


odf_use.columns = odf_use.columns.to_flat_index().map('_'.join)

odf_use = odf_use.reset_index().dropna() #this drops std nan I think! which removes years with 1 cast!


odf_use = (odf_use
                  .rename(columns={'date_ordinal_mean':'date_ordinal'})
                  .dropna()
                  .assign(
                          datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                          )
                  )


odf_use = pd.merge(odf_use, annual_counts, how='left', on=['site','year','var'])


odf_use = odf_use[odf_use['cid_count'] >1] #redundant but fine (see note line 234)

odf_use['val_ci95hi'] = odf_use['val_mean'] + 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])

odf_use['val_ci95lo'] = odf_use['val_mean'] - 1.96*odf_use['val_std']/np.sqrt(odf_use['cid_count'])

# %%

alpha = 0.05

all_stats_filt = pd.DataFrame()




            
            # for var in odf_calc_use['var'].unique():
                        
            #     mask = (odf_calc_use['site'] == site) & (odf_calc_use['summer_non_summer'] == season) & (odf_calc_use['var'] == var)
                
            #     plot_df = odf_calc_use[mask]
                
            #     x = plot_df['date_ordinal']
                
            #     x_plot = plot_df['datetime']
                
            #     y = plot_df['val']
                
            #     for stat in ['linreg', 'mk_ts']:
                    
            #         plot_df = odf_calc_use[mask]
                    
            #         if stat == 'linreg':
                        
            #             plot_df['stat'] = stat
                
            #             result = stats.linregress(x, y)
                        
            #             B1 = result.slope
                        
            #             B0 = result.intercept
                        
            #             sB1 = result.stderr
                        
            #             plot_df['p'] = result.pvalue
                        
            #             slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
                        
            #             slope_datetime_s = (B0 + sB1*x.max() - (B0 + sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                
            #             plot_df['slope_datetime'] = slope_datetime #per year
                        
            #             plot_df['slope_datetime_s'] = slope_datetime_s #per year
                        
            #             plot_df['slope_datetime_unc_cent'] =  str(np.round(slope_datetime*100,1)) + '+/-' + str(np.round(slope_datetime_s*100,1))
                        
            #             plot_df_concat = plot_df[['site','stat','var', 'p', 'slope_datetime_unc_cent', 'slope_datetime', 'slope_datetime_s']].head(1)
                        
            #             plot_df_concat['deep_DO_q'] = deep_DO_q
            
            #             all_stats_filt = pd.concat([all_stats_filt, plot_df_concat])
                
            #         elif stat == 'mk_ts':
                        
            #             plot_df['stat'] = stat
                        
            #             reject_null, p_value, Z = dfun.mann_kendall(y, alpha)
                                    
            #             plot_df['p'] = p_value
                                    
            #             result = stats.theilslopes(y,x,alpha=alpha)
                
            #             B1 = result.slope
                
            #             B0 = result.intercept
                        
            #             high_sB1 = result.high_slope
                        
            #             low_sB1 = result.low_slope
    
            #             slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
                
            #             plot_df['slope_datetime'] = slope_datetime #per year
                
            #             if (high_sB1-B1) >= (B1-low_sB1):
                            
            #                 sB1 = high_sB1-B1
                            
            #             else:
                            
            #                 sB1 = B1-low_sB1
                                    
            #             slope_datetime_s = (B0 + sB1*x.max() - (B0 + sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                
            #             plot_df['slope_datetime_s'] = slope_datetime_s #per year
                        
                        
            #             slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                        
            #             slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
                        
            #             plot_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
                        
            #             plot_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
                        
                
            #             plot_df['slope_datetime_unc_cent'] =  str(np.round(slope_datetime*100,1)) + '+/-' + str(np.round(slope_datetime_s*100,1))
                        
            #             plot_df_concat = plot_df[['site','stat','var', 'p', 'slope_datetime_unc_cent', 'slope_datetime', 'slope_datetime_s', 'slope_datetime_s_hi', 'slope_datetime_s_lo']].head(1)
                        
            #             plot_df_concat['deep_DO_q'] = deep_DO_q
            
            #             all_stats_filt = pd.concat([all_stats_filt, plot_df_concat])
            
            
for site in odf_use['site'].unique():
    #for depth in ['surf', 'deep']:
        
    for var in var_list:
        
        mask = (odf_use['site'] == site) & (odf_use['var'] == var)
        
        plot_df = odf_use[mask]
        
        x = plot_df['date_ordinal']
        
        x_plot = plot_df['datetime']
        
        plot_df['val'] = plot_df['val_mean']
        
        y = plot_df['val']
        
                        
        #plot_df['stat'] = stat
        
        reject_null, p_value, Z = dfun.mann_kendall(y, alpha)
                    
        plot_df['p'] = p_value


        result = stats.theilslopes(y,x,alpha=alpha)
        
        B1 = result.slope

        B0 = result.intercept

        plot_df['B1'] = B1

        plot_df['B0'] = B0
        
        high_sB1 = result.high_slope
        
        low_sB1 = result.low_slope

        slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)

        plot_df['slope_datetime'] = slope_datetime #per year

        # if (high_sB1-B1) >= (B1-low_sB1):
            
        #     sB1 = high_sB1-B1
            
        # else:
            
        #     sB1 = B1-low_sB1
                    
        # slope_datetime_s = (B0 + sB1*x.max() - (B0 + sB1*x.min()))/(x_plot.max().year - x_plot.min().year)

        # plot_df['slope_datetime_s'] = slope_datetime_s #per year
        
        
        slope_datetime_s_hi = (B0 + high_sB1*x.max() - (B0 + high_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
        
        slope_datetime_s_lo = (B0 + low_sB1*x.max() - (B0 + low_sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
        
        plot_df['slope_datetime_s_hi'] = slope_datetime_s_hi #per year
        
        plot_df['slope_datetime_s_lo'] = slope_datetime_s_lo #per year
        

        #plot_df['slope_datetime_unc_cent'] =  str(np.round(slope_datetime*100,1)) + '+/-' + str(np.round(slope_datetime_s*100,1))

        #plot_df = plot_df.rename(columns={'var': 'var_'})
        
        #plot_df['var'] = plot_df['surf_deep'] + '_' + plot_df['var']
        
        plot_df['site'] == site
                                        
        plot_df_concat = plot_df[['site','var', 'p', 'slope_datetime', 'slope_datetime_s_hi', 'slope_datetime_s_lo', 'B0', 'B1']].head(1)
        
        #plot_df_concat['deep_DO_q'] = deep_DO_q

        all_stats_filt = pd.concat([all_stats_filt, plot_df_concat])
                                
                        
# %%

fig, ax = plt.subplots(figsize=(6, 4))
        
for site in ['ps', 'ecology_nc']:
    
    plot_df = odf_use[(odf_use['site'] == site) & (odf_use['var'] == 'DO_mg_L')] # & (odf_use_full['surf_deep'] == 'deep')]
    

        
    #colors = {'Bottom 20%':'#673AB7', 'Surface 5m':'#E91E63'}

    
    #sns.scatterplot(data=plot_df_full, x='year', y = 'val',  ax=ax[0], label = 'all')
    
    # plot_df.loc[plot_df['surf_deep'] == 'surf','Surface5m_Bottom20pct'] = 'Surface 5m'
    
    # plot_df.loc[plot_df['surf_deep'] == 'deep','Surface5m_Bottom20pct'] = 'Bottom 20%'
    
    plot_df['val'] = plot_df['val_mean']
    
    if site == 'ps':
    
        y_spot = 0.1
        
        color = 'black'
    
    elif site == 'ecology_nc':
        
        y_spot = 0.2
        
        color = '#E91E63'
    
    sns.scatterplot(data=plot_df, x='year', y = 'val',  color=color,ax=ax, alpha=0.7)
    
    for idx in plot_df.index:
        
    #     if plot_df.loc[idx,'surf_deep'] == 'surf':
            
    #         color = '#E91E63'
            
        ax.plot([plot_df.loc[idx,'year'], plot_df.loc[idx,'year']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']],  color=color, alpha =0.7, zorder = -4, linewidth=1)
    
        # elif plot_df.loc[idx,'surf_deep'] == 'deep':
            
        #     color = '#673AB7'
            
    #ax.plot([plot_df.loc[idx,'year'], plot_df.loc[idx,'year']],[plot_df.loc[idx,'val_ci95lo'], plot_df.loc[idx,'val_ci95hi']], color=color, alpha =0.7, zorder = -4, linewidth=1)
    
    #for var in ['surf_DO_mg_L','deep_DO_mg_L']:
        
    filt_df = all_stats_filt[(all_stats_filt['site'] == site) & (all_stats_filt['var'] == 'DO_mg_L')]
    
    # if 'surf' in var:
        
    #     if filt_df['p'].iloc[0] < alpha:
        
    #         color = '#E91E63'
            
    #     else:
            
    #         color = 'lightgray'

    #     x = plot_df[plot_df['surf_deep'] == 'surf']['date_ordinal']
        
    #     x_plot = plot_df[plot_df['surf_deep'] == 'surf']['year']
        
    #     y = plot_df[plot_df['surf_deep'] == 'surf']['val']
        
    #     y_spot = 0.2
        
    # else:
        
    #     if filt_df['p'].iloc[0] < alpha:
        
    #         color = '#673AB7'
            
    #     else:
            
    #         color = 'lightgray'
        
    x = plot_df['date_ordinal']
    
    x_plot = plot_df['year']
    
    y = plot_df['val']
    
    p = filt_df['p'].iloc[0]
        
    B0 = filt_df['B0'].iloc[0]
    
    B1 = filt_df['B1'].iloc[0]
    
    slope_datetime = filt_df['slope_datetime'].iloc[0]
    
    slope_datetime_s_hi = filt_df['slope_datetime_s_hi'].iloc[0]
    
    slope_datetime_s_lo = filt_df['slope_datetime_s_lo'].iloc[0]

    # slope_datetime = (B0 + B1*x.max() - (B0 + B1*x.min()))/(x_plot.max().year - x_plot.min().year)
    
    # slope_datetime_s = (B0 + sB1*x.max() - (B0 + sB1*x.min()))/(x_plot.max().year - x_plot.min().year)
    
    if p > 0.05:
        
        color = 'gray'

    
    ax.plot([x_plot.min(), x_plot.max()], [B0 + B1*x.min(), B0 + B1*x.max()], color=color, alpha =0.7, linestyle = 'dashed', linewidth=2)

    ax.text(0.98,y_spot, site + ' theilsen = ' + str(np.round(slope_datetime*100,2)) + '/cent. +' + str(np.round(slope_datetime_s_hi*100 - slope_datetime*100,2)) +'/-' + str(np.round(slope_datetime*100-slope_datetime_s_lo*100,2)), horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, color=color, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    #ax.text(0.5,y_spot, "NOTE: No statistically significant trend is observed. Some observed\nvariability is due to biases introduced by inconsistent sampling\nlocation, depth, and frequency throughout the time series.", horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, color=color, bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'), wrap=False)

ax.text(0.05, 0.95, 'Puget Sound Annual Average [DO]', horizontalalignment='left', verticalalignment='top', transform = ax.transAxes, fontweight='bold')



ax.set_ylim(0, 12)

ax.axhspan(0,2, color = 'lightgray', alpha = 0.2)

ax.set_ylabel(r'DO [mg/L]')

ax.set_xlabel('')

# ax[1].set_ylim(0,366)

# ax[1].set_ylabel('cast yearday')

# ax[1].axhspan(213,274, color = 'lightgray', alpha = 0.2) #july31/august1-september30/oct1


# ax[2].set_ylim(-300,0)

# ax[2].set_ylabel('cast deep depth [m]')

ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)

# ax[1].grid(color = 'lightgray', linestyle = '--', alpha=0.5)

# ax[2].grid(color = 'lightgray', linestyle = '--', alpha=0.5)

#ax.legend(title='', loc='lower left')

plt.tight_layout()


plt.savefig('/Users/dakotamascarenas/Desktop/pltz/allps_ecology_surfdeepDO_timeseries.png', bbox_inches='tight', dpi=500, transparent=False)


