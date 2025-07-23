"""
*** Fill in!

Getting the data 

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


# %%

Ldir = Lfun.Lstart(gridname='cas7')

# %%

poly_list = ['sog_n', 'sog_s_wo_si', 'si', 'soj','sji','mb', 'wb', 'hc', 'ss']

odf_dict = dfun.getPolyData(Ldir, poly_list)

with open('/Users/dakotamascarenas/Desktop/big_dict.pkl', 'wb') as f:
    pickle.dump(odf_dict, f)


# %%

with open('/Users/dakotamascarenas/Desktop/big_dict.pkl', 'rb') as f:
    odf_dict = pickle.load(f)


# %%

basin_list = list(odf_dict.keys())

# %%
depth_div_0 = -5
depth_div_1 = -10
depth_div_2 = -15
depth_div_3 = -30
depth_div_4 = -50
depth_div_5 = -75
depth_div_6 = -100
depth_div_7 = -150
depth_div_8 = -200


period_div_0 = 1945
period_div_1 = 1980
period_div_2 = 1995

#year_div = 2010

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                     .assign(
                         datetime=(lambda x: pd.to_datetime(x['time'])),
                         depth_range=(lambda x: pd.cut(x['z'], 
                                                       bins=[x['z'].min()-1, depth_div_8, depth_div_7, depth_div_6, depth_div_5, depth_div_4, depth_div_3, depth_div_2, depth_div_1, depth_div_0, 0],
                                                       labels= ['>200m', '150-200m', '100-150m', '75-100m', '50-75m', '30-50m', '15-30m', '10-15m', '5-10m', '<5m'])), # make less manual
                         year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                         month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                         season=(lambda x: pd.cut(x['month'],
                                                  bins=[0,3,6,9,12],
                                                  labels=['winter', 'spring', 'summer', 'fall'])),
                         DO_mg_L=(lambda x: x['DO (uM)']*32/1000),
                         period_label=(lambda x: pd.cut(x['year'], 
                                                    bins=[x['year'].min()-1, period_div_0-1, period_div_1-1, period_div_2-1, x['year'].max()],
                                                    labels= ['-1945', '1945-1980', '1980-1995','1995-'])),
                         date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal())),
                         segment=(lambda x: key)
                             )
                     )

# %%

var_list = ['SA', 'CT', 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO_mg_L', 'DO (uM)']

# %%

for key in odf_dict.keys():
    
    for var in var_list:
        
        if var not in odf_dict[key].columns:
            
            odf_dict[key][var] = np.nan
            

# %%



# plot monthly average DO sampling depth and average DO with cid based confidence intervals in each depth bin, plot cast average for each source/otype

# cast averages

cast_depth_avgs_dict = dict()

for key in odf_dict.keys():
    
    cast_depth_avgs_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype'],
                                         value_vars=var_list, var_name='var', value_name = 'val')
                                 .groupby(['cid', 'depth_range', 'var', 'source', 'otype']).mean()
                                 .reset_index()
                                 .dropna()
                                 .assign(
                                         segment=(lambda x: key),
                                         season=(lambda x: pd.cut(x['month'],
                                                                  bins=[0,3,6,9,12],
                                                                  labels=['winter', 'spring', 'summer', 'fall'])),
                                         datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                                         )
                                 )
                                 
                                 
# %%

depth_month_avgs_dict = dict()

for key in odf_dict.keys():
    
    # build in the timeframe flexibility
    
    counts = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'period_label', 'source', 'otype'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                         .dropna()
                         .set_index('datetime')
                         .groupby([pd.Grouper(freq='M'), 'depth_range', 'var']).agg({'cid' :lambda x: x.nunique()})
                         .reset_index()
                         )
    
    depth_month_avgs_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                      .set_index('datetime')
                      .groupby([pd.Grouper(freq='M'), 'depth_range', 'var']).agg(['mean', 'std'])
                      .drop(columns =['lat','lon','cid', 'year', 'month', 'date_ordinal'])
                      )
    
    depth_month_avgs_dict[key].columns = depth_month_avgs_dict[key].columns.to_flat_index().map('_'.join)
    
    depth_month_avgs_dict[key] = (depth_month_avgs_dict[key]
                      .reset_index() 
                      .assign(
                              # period_label=(lambda x: pd.cut(x['year'], 
                              #                              bins=[x['year'].min()-1, year_div-1, x['year'].max()],
                              #                              labels= ['pre', 'post'])),
                              segment=(lambda x: key),
                              year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              season=(lambda x: pd.cut(x['month'],
                                                       bins=[0,3,6,9,12],
                                                       labels=['winter', 'spring', 'summer', 'fall'])),
                              date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal()))
                              )
                      )
    
    depth_month_avgs_dict[key] = pd.merge(depth_month_avgs_dict[key], counts, how='left', on=['datetime', 'depth_range', 'var']).rename(columns={'cid':'cid_count'})


# %%

depth_month_avgs_df = pd.concat(depth_month_avgs_dict.values(), ignore_index=True)

# %%

depth_month_avgs_df['val_ci95hi'] = depth_month_avgs_df['val_mean'] + 1.96*depth_month_avgs_df['val_std']/np.sqrt(depth_month_avgs_df['cid_count'])

depth_month_avgs_df['val_ci95lo'] = depth_month_avgs_df['val_mean'] - 1.96*depth_month_avgs_df['val_std']/np.sqrt(depth_month_avgs_df['cid_count'])

# %%

for basin in basin_list:

    plot_df = depth_month_avgs_df[(depth_month_avgs_df['segment'] == basin) & (depth_month_avgs_df['cid_count'] > 1)]
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
    
    plot_df_SA = plot_df[plot_df['var'] == 'SA']
    
    plot_df_CT = plot_df[plot_df['var'] == 'CT']
    
    plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
    
    # for idx in plot_df_SA.index:
        
    #     ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    # for idx in plot_df_CT.index:
        
    #     ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
    
    # for idx in plot_df_DO.index:
        
    #     ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    sns.scatterplot(data = plot_df_SA, x='datetime', y = 'val_mean', hue='depth_range', palette= 'crest_r', marker='s', ax=ax0)
    
    sns.scatterplot(data = plot_df_CT, x='datetime', y = 'val_mean', hue='depth_range', palette= 'crest_r', marker='^', ax=ax1)
    
    sns.scatterplot(data = plot_df_DO, x='datetime', y = 'val_mean', hue='depth_range', palette= 'crest_r', ax=ax2)
    
    
    # ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
    
    # ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
    
    # ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
    
    ax2.set_xlabel('Date')
    
    ax0.set_ylabel('SA [psu]')
    
    ax1.set_ylabel('CT [deg C]')
    
    ax2.set_ylabel('DO [mg/L]')
    
    ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax0.set_ylim([26,36])
    
    ax1.set_ylim([2,16])
    
    ax2.set_ylim([0,14])
    
    ax0.set_title(basin)
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_all_depths_ci.png', dpi=500)


# %%


for basin in basin_list:
    
    for depth in depth_month_avgs_df['depth_range'].unique():

        plot_df = depth_month_avgs_df[(depth_month_avgs_df['segment'] == basin) & (depth_month_avgs_df['cid_count'] > 1) & (depth_month_avgs_df['depth_range'] == depth)]
        
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
        
        plot_df_SA = plot_df[plot_df['var'] == 'SA']
        
        plot_df_CT = plot_df[plot_df['var'] == 'CT']
        
        plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
        
        plot_df_DO = plot_df_DO[plot_df_DO['val_mean'] < 50] #SHOULD I FILTER THIS OUT BEFORE?!?!?!
        
        for idx in plot_df_SA.index:
            
            ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        for idx in plot_df_CT.index:
            
            ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
        
        for idx in plot_df_DO.index:
            
            ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        a = sns.scatterplot(data = plot_df_SA, x='datetime', y = 'val_mean', hue='depth_range', palette='crest_r', marker='s', ax=ax0, legend=False)
        
        b = sns.scatterplot(data = plot_df_CT, x='datetime', y = 'val_mean', hue='depth_range', palette='crest_r', marker='^', ax=ax1, legend=False)
        
        c = sns.scatterplot(data = plot_df_DO, x='datetime', y = 'val_mean', hue='depth_range', palette='crest_r', ax=ax2, legend=False)
        
        
        
        # ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
        
        # ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
        
        # ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
        
        ax2.set_xlabel('Date')
        
        ax0.set_ylabel('SA [psu]')
        
        ax1.set_ylabel('CT [deg C]')
        
        ax2.set_ylabel('DO [mg/L]')
        
        ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax0.set_ylim([20,36])
        
        ax1.set_ylim([2,16])
        
        ax2.set_ylim([0, 20])
        
        ax0.set_title(basin + ' ' + depth)
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_' + depth + '_ci.png', dpi=500)

# %%

for basin in basin_list:
    
    for depth in depth_month_avgs_df['depth_range'].unique():
        
        fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, figsize=(20, 15))
        
        c=0
        
        for season in ['winter', 'spring', 'summer', 'fall']:

            plot_df = depth_month_avgs_df[(depth_month_avgs_df['segment'] == basin) & (depth_month_avgs_df['cid_count'] > 1) & (depth_month_avgs_df['depth_range'] == depth) & (depth_month_avgs_df['season'] == season)]
            
            plot_df_SA = plot_df[plot_df['var'] == 'SA']
            
            plot_df_CT = plot_df[plot_df['var'] == 'CT']
            
            plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
            
            plot_df_DO = plot_df_DO[plot_df_DO['val_mean'] < 50] #SHOULD I FILTER THIS OUT BEFORE?!?!?!
            
            for idx in plot_df_SA.index:
                
                ax[0,c].vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
                
            for idx in plot_df_CT.index:
                
                ax[1,c].vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
            
            for idx in plot_df_DO.index:
                
                ax[2,c].vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
                
            sns.scatterplot(data = plot_df_SA, x='datetime', y = 'val_mean', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', marker='s', ax=ax[0,c], legend=False)
            
            sns.scatterplot(data = plot_df_CT, x='datetime', y = 'val_mean', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', marker='^', ax=ax[1,c], legend=False)
            
            sns.scatterplot(data = plot_df_DO, x='datetime', y = 'val_mean', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', ax=ax[2,c], legend=False)
            
            ax[0,c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[1,c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[2,c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[0,c].set_title(season)
            
            ax[0,c].set_ylim([20,36])
            
            ax[1,c].set_ylim([2,16])
            
            ax[2,c].set_ylim([0, 20])
            
            ax[2,c].set_xlabel('Date')
            
            ax[0,c].set_ylabel('SA [psu]')
            
            ax[1,c].set_ylabel('CT [deg C]')
            
            ax[2,c].set_ylabel('DO [mg/L]')
            
            c+=1
            
            
        plt.suptitle(basin + ' ' + depth)
            
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_' + depth + '_monthly_by_seasons_ci.png', dpi=500)


# %%

# plot z in each depth bin, colored by depth



for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15), sharex=True)
                
    plot_df_monthly_means = depth_month_avgs_df[(depth_month_avgs_df['segment'] == basin) & (~np.isnan(depth_month_avgs_df['val_mean'])) & (depth_month_avgs_df['var'] == 'DO_mg_L')] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
    
    plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L')] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                
    sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='depth_range', palette='crest_r', alpha=0.5)
    
    sns.scatterplot(data=plot_df_monthly_means, x='datetime', y='z_mean', color='black', edgecolor=None)
    
    plt.title(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('Depth [m]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth.png', dpi=500)


# %%

for basin in basin_list:
    
    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(20, 15))
    
    c=0
    
    for season in ['winter', 'spring', 'summer', 'fall']:
                
        plot_df_monthly_means = depth_month_avgs_df[(depth_month_avgs_df['segment'] == basin) & (~np.isnan(depth_month_avgs_df['val_mean'])) & (depth_month_avgs_df['var'] == 'DO_mg_L') & (depth_month_avgs_df['season'] == season)] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
        
        plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L') & (cast_depth_avgs_dict[basin]['season'] == season)] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                    
        sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', alpha=0.5, ax=ax[c])
        
        sns.scatterplot(data=plot_df_monthly_means, x='datetime', y='z_mean', color='black', edgecolor=None, ax=ax[c])
                
        ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax[c].set_title(season)
        
        ax[c].set_xlabel('Date')
        
        ax[c].set_ylabel('Depth [m]')
        
        c+=1
          
    plt.suptitle(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_monthly_by_seasons.png', dpi=500)


# %%

for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15), sharex=True)
                
    plot_df_monthly_means = depth_month_avgs_df[(depth_month_avgs_df['segment'] == basin) & (~np.isnan(depth_month_avgs_df['val_mean'])) & (depth_month_avgs_df['var'] == 'DO_mg_L')] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
    
    plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L')] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                
    sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='source', palette='Set2', hue_order=['dfo1','ecology','collias', 'nceiSalish'], alpha=0.5)
    
    sns.scatterplot(data=plot_df_monthly_means, x='datetime', y='z_mean', color='black', edgecolor=None)
    
    plt.title(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('Depth [m]')
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_by_source.png', dpi=500)


# %%

for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15), sharex=True)
                
    plot_df_monthly_means = depth_month_avgs_df[(depth_month_avgs_df['segment'] == basin) & (~np.isnan(depth_month_avgs_df['val_mean'])) & (depth_month_avgs_df['var'] == 'DO_mg_L')] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
    
    plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L')] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                
    sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='otype', palette='Set2', hue_order=['bottle','ctd'], alpha=0.5)
    
    sns.scatterplot(data=plot_df_monthly_means, x='datetime', y='z_mean', color='black', edgecolor=None)
    
    plt.title(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_by_type.png', dpi=500)


# %%

# %%

# %%


# now seasonal averages







depth_season_avgs_dict = dict()

for key in odf_dict.keys():
    
    # build in the timeframe flexibility
    
    counts = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'period_label', 'source', 'otype'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                         .dropna()
                         #.set_index('datetime')
                         .groupby(['year','season', 'depth_range', 'var']).agg({'cid' :lambda x: x.nunique()})
                         .reset_index()
                         )
    
    depth_season_avgs_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                     # .set_index('datetime')
                      .groupby(['year','season', 'depth_range', 'var']).agg(['mean', 'std'])
                      .drop(columns =['lat','lon','cid', 'month'])
                      )
    
    depth_season_avgs_dict[key].columns = depth_season_avgs_dict[key].columns.to_flat_index().map('_'.join)
        
    depth_season_avgs_dict[key] = (depth_season_avgs_dict[key]
                      .drop(columns=['date_ordinal_std'])
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .reset_index() 
                      .dropna()
                      .assign(
                              # period_label=(lambda x: pd.cut(x['year'], 
                              #                              bins=[x['year'].min()-1, year_div-1, x['year'].max()],
                              #                              labels= ['pre', 'post'])),
                              segment=(lambda x: key),
                              # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              #month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              # season=(lambda x: pd.cut(x['month'],
                              #                          bins=[0,3,6,9,12],
                              #                          labels=['winter', 'spring', 'summer', 'fall'])),
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    depth_season_avgs_dict[key] = pd.merge(depth_season_avgs_dict[key], counts, how='left', on=['year', 'season', 'depth_range', 'var']).rename(columns={'cid':'cid_count'})



depth_season_avgs_df = pd.concat(depth_season_avgs_dict.values(), ignore_index=True)


depth_season_avgs_df['val_ci95hi'] = depth_season_avgs_df['val_mean'] + 1.96*depth_season_avgs_df['val_std']/np.sqrt(depth_season_avgs_df['cid_count'])

depth_season_avgs_df['val_ci95lo'] = depth_season_avgs_df['val_mean'] - 1.96*depth_season_avgs_df['val_std']/np.sqrt(depth_season_avgs_df['cid_count'])


# plot SA/CT/DO like monthly


for basin in basin_list:

    plot_df = depth_season_avgs_df[(depth_season_avgs_df['segment'] == basin) & (depth_season_avgs_df['cid_count'] > 1)]
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
    
    plot_df_SA = plot_df[plot_df['var'] == 'SA']
    
    plot_df_CT = plot_df[plot_df['var'] == 'CT']
    
    plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
    
    # for idx in plot_df_SA.index:
        
    #     ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    # for idx in plot_df_CT.index:
        
    #     ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
    
    # for idx in plot_df_DO.index:
        
    #     ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    sns.scatterplot(data = plot_df_SA, x='datetime', y = 'val_mean', hue='depth_range', palette= 'crest_r', marker='s', ax=ax0)
    
    sns.scatterplot(data = plot_df_CT, x='datetime', y = 'val_mean', hue='depth_range', palette= 'crest_r', marker='^', ax=ax1)
    
    sns.scatterplot(data = plot_df_DO, x='datetime', y = 'val_mean', hue='depth_range', palette= 'crest_r', ax=ax2)
    
    
    # ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
    
    # ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
    
    # ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
    
    ax2.set_xlabel('Date')
    
    ax0.set_ylabel('SA [psu]')
    
    ax1.set_ylabel('CT [deg C]')
    
    ax2.set_ylabel('DO [mg/L]')
    
    ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax0.set_ylim([26,36])
    
    ax1.set_ylim([2,16])
    
    ax2.set_ylim([0,14])
    
    ax0.set_title(basin)
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_all_depths_seasonal_avg_ci.png', dpi=500)



# seasons broken out my depth

for basin in basin_list:
    
    for depth in depth_season_avgs_df['depth_range'].unique():

        plot_df = depth_season_avgs_df[(depth_season_avgs_df['segment'] == basin) & (depth_season_avgs_df['cid_count'] > 1) & (depth_season_avgs_df['depth_range'] == depth)]
        
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
        
        plot_df_SA = plot_df[plot_df['var'] == 'SA']
        
        plot_df_CT = plot_df[plot_df['var'] == 'CT']
        
        plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
        
        plot_df_DO = plot_df_DO[plot_df_DO['val_mean'] < 50] #SHOULD I FILTER THIS OUT BEFORE?!?!?!
        
        for idx in plot_df_SA.index:
            
            ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        for idx in plot_df_CT.index:
            
            ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
        
        for idx in plot_df_DO.index:
            
            ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        a = sns.scatterplot(data = plot_df_SA, x='datetime', y = 'val_mean', hue='depth_range', palette='crest_r', marker='s', ax=ax0, legend=False)
        
        b = sns.scatterplot(data = plot_df_CT, x='datetime', y = 'val_mean', hue='depth_range', palette='crest_r', marker='^', ax=ax1, legend=False)
        
        c = sns.scatterplot(data = plot_df_DO, x='datetime', y = 'val_mean', hue='depth_range', palette='crest_r', ax=ax2, legend=False)
        
        
        
        # ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
        
        # ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
        
        # ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
        
        ax2.set_xlabel('Date')
        
        ax0.set_ylabel('SA [psu]')
        
        ax1.set_ylabel('CT [deg C]')
        
        ax2.set_ylabel('DO [mg/L]')
        
        ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax0.set_ylim([20,36])
        
        ax1.set_ylim([2,16])
        
        ax2.set_ylim([0, 20])
        
        ax0.set_title(basin + ' ' + depth)
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_' + depth + '_seasonal_avgs_ci.png', dpi=500)



for basin in basin_list:
    
    for depth in depth_season_avgs_df['depth_range'].unique():
        
        fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, figsize=(20, 15))
        
        c=0
        
        for season in ['winter', 'spring', 'summer', 'fall']:

            plot_df = depth_season_avgs_df[(depth_season_avgs_df['segment'] == basin) & (depth_season_avgs_df['cid_count'] > 1) & (depth_season_avgs_df['depth_range'] == depth) & (depth_season_avgs_df['season'] == season)]
            
            plot_df_SA = plot_df[plot_df['var'] == 'SA']
            
            plot_df_CT = plot_df[plot_df['var'] == 'CT']
            
            plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
            
            plot_df_DO = plot_df_DO[plot_df_DO['val_mean'] < 50] #SHOULD I FILTER THIS OUT BEFORE?!?!?!
            
            for idx in plot_df_SA.index:
                
                ax[0,c].vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
                
            for idx in plot_df_CT.index:
                
                ax[1,c].vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
            
            for idx in plot_df_DO.index:
                
                ax[2,c].vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
                
            sns.scatterplot(data = plot_df_SA, x='datetime', y = 'val_mean', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', marker='s', ax=ax[0,c], legend=False)
            
            sns.scatterplot(data = plot_df_CT, x='datetime', y = 'val_mean', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', marker='^', ax=ax[1,c], legend=False)
            
            sns.scatterplot(data = plot_df_DO, x='datetime', y = 'val_mean', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', ax=ax[2,c], legend=False)
            
            ax[0,c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[1,c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[2,c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
            ax[0,c].set_title(season)
            
            ax[0,c].set_ylim([20,36])
            
            ax[1,c].set_ylim([2,16])
            
            ax[2,c].set_ylim([0, 20])
            
            ax[2,c].set_xlabel('Date')
            
            ax[0,c].set_ylabel('SA [psu]')
            
            ax[1,c].set_ylabel('CT [deg C]')
            
            ax[2,c].set_ylabel('DO [mg/L]')
            
            c+=1
            
            
        plt.suptitle(basin + ' ' + depth)
            
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_' + depth + '_seasonal_by_seasons_ci.png', dpi=500)





# plot z in each depth bin, colored by depth



for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15), sharex=True)
                
    plot_df_seasonal_means = depth_season_avgs_df[(depth_season_avgs_df['segment'] == basin) & (~np.isnan(depth_season_avgs_df['val_mean'])) & (depth_season_avgs_df['var'] == 'DO_mg_L')] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
    
    plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L')] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                
    sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='depth_range', palette='crest_r', alpha=0.5)
    
    sns.scatterplot(data=plot_df_seasonal_means, x='datetime', y='z_mean', color='black', edgecolor=None)
    
    plt.title(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('Depth [m]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_seasonal_avg.png', dpi=500)



for basin in basin_list:
    
    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(20, 15))
    
    c=0
    
    for season in ['winter', 'spring', 'summer', 'fall']:
                
        plot_df_seasonal_means = depth_season_avgs_df[(depth_season_avgs_df['segment'] == basin) & (~np.isnan(depth_season_avgs_df['val_mean'])) & (depth_season_avgs_df['var'] == 'DO_mg_L') & (depth_season_avgs_df['season'] == season)] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
        
        plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L') & (cast_depth_avgs_dict[basin]['season'] == season)] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                    
        sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', alpha=0.5, ax=ax[c])
        
        sns.scatterplot(data=plot_df_seasonal_means, x='datetime', y='z_mean', color='black', edgecolor=None, ax=ax[c])
                
        ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax[c].set_title(season)
        
        ax[c].set_xlabel('Date')
        
        ax[c].set_ylabel('Depth [m]')
        
        c+=1
          
    plt.suptitle(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_seasonal_by_seasons.png', dpi=500)

# %%

for basin in basin_list:
    
    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(20, 15))
    
    c=0
    
    for season in ['winter', 'spring', 'summer', 'fall']:
                
        plot_df_seasonal_means = depth_season_avgs_df[(depth_season_avgs_df['segment'] == basin) & (~np.isnan(depth_season_avgs_df['val_mean'])) & (depth_season_avgs_df['var'] == 'DO_mg_L') & (depth_season_avgs_df['season'] == season)] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
        
        plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L') & (cast_depth_avgs_dict[basin]['season'] == season)] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                    
        sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='source', palette='Set2', hue_order=['dfo1','ecology','collias', 'nceiSalish'], alpha=0.5, ax=ax[c])
        
        sns.scatterplot(data=plot_df_seasonal_means, x='datetime', y='z_mean', color='black', edgecolor=None, ax=ax[c])
                
        ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax[c].set_title(season)
        
        ax[c].set_xlabel('Date')
        
        ax[c].set_ylabel('Depth [m]')
        
        c+=1
          
    plt.suptitle(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_by_source_seasonal_by_seasons.png', dpi=500)


# %%

for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15), sharex=True)
                
    plot_df_seasonal_means = depth_season_avgs_dict[(depth_season_avgs_dict['segment'] == basin) & (~np.isnan(depth_season_avgs_dict['val_mean'])) & (depth_season_avgs_dict['var'] == 'DO_mg_L')] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
    
    plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L')] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                
    sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='source', palette='Set2', hue_order=['dfo1','ecology','collias', 'nceiSalish'], alpha=0.5)
    
    sns.scatterplot(data=plot_df_seasonal_means, x='datetime', y='z_mean', color='black', edgecolor=None)
    
    plt.title(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('Depth [m]')
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_by_source_seasonal_avg.png', dpi=500)


# %%

for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15), sharex=True)
                
    plot_df_seasonal_means = depth_season_avgs_dict[(depth_season_avgs_dict['segment'] == basin) & (~np.isnan(depth_season_avgs_dict['val_mean'])) & (depth_season_avgs_dict['var'] == 'DO_mg_L')] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
    
    plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L')] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                
    sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='otype', palette='Set2', hue_order=['bottle','ctd'], alpha=0.5)
    
    sns.scatterplot(data=plot_df_seasonal_means, x='datetime', y='z_mean', color='black', edgecolor=None)
    
    plt.title(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_by_type_seasonal_avg.png', dpi=500)

# %%

# %%

# %%



# # NOW YEARLY

# # could do grouper probably, for now use seasonal precedent

depth_year_avgs_dict = dict()

for key in odf_dict.keys():
    
    # build in the timeframe flexibility
    
    counts = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'period_label', 'source', 'otype'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                         .dropna()
                         #.set_index('datetime')
                         .groupby(['year', 'depth_range', 'var']).agg({'cid' :lambda x: x.nunique()})
                         .reset_index()
                         )
    
    depth_year_avgs_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                     # .set_index('datetime')
                      .groupby(['year', 'depth_range', 'var']).agg(['mean', 'std'])
                      .drop(columns =['lat','lon','cid', 'month'])
                      )
    
    depth_year_avgs_dict[key].columns = depth_year_avgs_dict[key].columns.to_flat_index().map('_'.join)
        
    depth_year_avgs_dict[key] = (depth_year_avgs_dict[key]
                      .drop(columns=['date_ordinal_std'])
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .reset_index() 
                      .dropna()
                      .assign(
                              # period_label=(lambda x: pd.cut(x['year'], 
                              #                              bins=[x['year'].min()-1, year_div-1, x['year'].max()],
                              #                              labels= ['pre', 'post'])),
                              segment=(lambda x: key),
                              # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              #month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              # season=(lambda x: pd.cut(x['month'],
                              #                          bins=[0,3,6,9,12],
                              #                          labels=['winter', 'spring', 'summer', 'fall'])),
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    depth_year_avgs_dict[key] = pd.merge(depth_year_avgs_dict[key], counts, how='left', on=['year', 'depth_range', 'var']).rename(columns={'cid':'cid_count'})


depth_year_avgs_df = pd.concat(depth_year_avgs_dict.values(), ignore_index=True)


depth_year_avgs_df['val_ci95hi'] = depth_year_avgs_df['val_mean'] + 1.96*depth_year_avgs_df['val_std']/np.sqrt(depth_year_avgs_df['cid_count'])

depth_year_avgs_df['val_ci95lo'] = depth_year_avgs_df['val_mean'] - 1.96*depth_year_avgs_df['val_std']/np.sqrt(depth_year_avgs_df['cid_count'])

# %%
for basin in basin_list:

    plot_df = depth_year_avgs_df[(depth_year_avgs_df['segment'] == basin) & (depth_year_avgs_df['cid_count'] > 1)]
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
    
    plot_df_SA = plot_df[plot_df['var'] == 'SA']
    
    plot_df_CT = plot_df[plot_df['var'] == 'CT']
    
    plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
    
    sns.scatterplot(data = plot_df_SA, x='datetime', y = 'val_mean', hue='depth_range', palette= 'crest_r', marker='s', ax=ax0)
    
    sns.scatterplot(data = plot_df_CT, x='datetime', y = 'val_mean', hue='depth_range', palette= 'crest_r', marker='^', ax=ax1)
    
    sns.scatterplot(data = plot_df_DO, x='datetime', y = 'val_mean', hue='depth_range', palette= 'crest_r', ax=ax2)
    
    ax2.set_xlabel('Date')
    
    ax0.set_ylabel('SA [psu]')
    
    ax1.set_ylabel('CT [deg C]')
    
    ax2.set_ylabel('DO [mg/L]')
    
    ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax0.set_ylim([26,36])
    
    ax1.set_ylim([2,16])
    
    ax2.set_ylim([0,14])
    
    ax0.set_title(basin)
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_all_depths_yearly_avg_ci.png', dpi=500)
    

for basin in basin_list:
    
    for depth in depth_year_avgs_df['depth_range'].unique():

        plot_df = depth_year_avgs_df[(depth_year_avgs_df['segment'] == basin) & (depth_year_avgs_df['cid_count'] > 1) & (depth_year_avgs_df['depth_range'] == depth)]
        
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
        
        plot_df_SA = plot_df[plot_df['var'] == 'SA']
        
        plot_df_CT = plot_df[plot_df['var'] == 'CT']
        
        plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
        
        plot_df_DO = plot_df_DO[plot_df_DO['val_mean'] < 50] #SHOULD I FILTER THIS OUT BEFORE?!?!?!
        
        for idx in plot_df_SA.index:
            
            ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        for idx in plot_df_CT.index:
            
            ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
        
        for idx in plot_df_DO.index:
            
            ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
            
        a = sns.scatterplot(data = plot_df_SA, x='datetime', y = 'val_mean', hue='depth_range', palette='crest_r', marker='s', ax=ax0, legend=False)
        
        b = sns.scatterplot(data = plot_df_CT, x='datetime', y = 'val_mean', hue='depth_range', palette='crest_r', marker='^', ax=ax1, legend=False)
        
        c = sns.scatterplot(data = plot_df_DO, x='datetime', y = 'val_mean', hue='depth_range', palette='crest_r', ax=ax2, legend=False)
        
        
        
        # ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
        
        # ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
        
        # ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
        
        ax2.set_xlabel('Date')
        
        ax0.set_ylabel('SA [psu]')
        
        ax1.set_ylabel('CT [deg C]')
        
        ax2.set_ylabel('DO [mg/L]')
        
        ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax1.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax2.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax0.set_ylim([20,36])
        
        ax1.set_ylim([2,16])
        
        ax2.set_ylim([0, 20])
        
        ax0.set_title(basin + ' ' + depth)
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_' + depth + '_seasonal_avgs_ci.png', dpi=500)
        

for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15), sharex=True)
                
    plot_df_year_means = depth_year_avgs_df[(depth_year_avgs_df['segment'] == basin) & (~np.isnan(depth_year_avgs_df['val_mean'])) & (depth_year_avgs_df['var'] == 'DO_mg_L')] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
    
    plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L')] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                
    sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='depth_range', palette='crest_r', alpha=0.5)
    
    sns.scatterplot(data=plot_df_year_means, x='datetime', y='z_mean', color='black', edgecolor=None)
    
    plt.title(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('Depth [m]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_yearly_avg.png', dpi=500)
    
    
for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15), sharex=True)
                
    plot_df_year_means = depth_year_avgs_df[(depth_year_avgs_df['segment'] == basin) & (~np.isnan(depth_year_avgs_df['val_mean'])) & (depth_year_avgs_df['var'] == 'DO_mg_L')] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
    
    plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L')] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                
    sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='source', palette='Set2', hue_order=['dfo1','ecology','collias', 'nceiSalish'], alpha=0.5)
    
    sns.scatterplot(data=plot_df_year_means, x='datetime', y='z_mean', color='black', edgecolor=None)
    
    plt.title(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('Depth [m]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_by_source_yearly_avg.png', dpi=500)
    
    
for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15), sharex=True)
                
    plot_df_year_means = depth_year_avgs_df[(depth_year_avgs_df['segment'] == basin) & (~np.isnan(depth_year_avgs_df['val_mean'])) & (depth_year_avgs_df['var'] == 'DO_mg_L')] #& (depth_month_avgs_df['depth_range'] == depth)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
    
    plot_df_cast_means = cast_depth_avgs_dict[basin][(~np.isnan(cast_depth_avgs_dict[basin]['val'])) & (cast_depth_avgs_dict[basin]['var'] == 'DO_mg_L')] # & (cast_depth_avgs_dict[basin]['depth_range'] == depth)]     
                
    sns.scatterplot(data= plot_df_cast_means, x='datetime', y='z', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', alpha=0.5)
    
    sns.scatterplot(data=plot_df_year_means, x='datetime', y='z_mean', color='black', edgecolor=None)
    
    plt.title(basin + ' avg DO sampling depths and cast avgs (by depth bins)')
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('Depth [m]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z_by_depth_by_season_yearly_avg.png', dpi=500)


# %%

# %%

# %%




# now, average casts for region and season for different time periods

avg_cast_f_dict = dict()

for basin in basin_list:
    
    avg_cast_f_dict[basin] = dict()
    
    for period in ['-1945', '1945-1980', '1980-1995','1995-']:
        
        avg_cast_f_dict[basin][period] = dict()
                
        for season in ['winter', 'spring','summer', 'fall']:
                        
            df_temp = odf_dict[basin][(odf_dict[basin]['period_label'] == period) & (~np.isnan(odf_dict[basin]['DO_mg_L'])) & (odf_dict[basin]['DO_mg_L'] < 50) & (odf_dict[basin]['season'] == season) & (odf_dict[basin]['z'].min() < -2)]
            
            if len(df_temp) > 2:
            
                df_temp = df_temp[['z','DO_mg_L']]
                
                df_temp['bin'] = df_temp['z'].apply(np.ceil)
                                  
                min_bin = math.floor(df_temp['z'].min())
                
                avgs = df_temp.groupby(['bin'], as_index=False).mean()
                
                data_array = avgs['DO_mg_L'].to_numpy()
                
                good_data = data_array[~np.isnan(data_array)]
                
                bin_array = avgs['bin'].to_numpy()
                
                bin_data = bin_array[~np.isnan(data_array)]
                                    
                avg_cast_f = interp1d(bin_data, good_data, kind='linear', fill_value='extrapolate')
                
                avg_cast_f_dict[basin][period][season] = avg_cast_f
                
# %%


# plot average casts per period with all cast data (color by year)

for basin in basin_list:
    
    for period in ['-1945', '1945-1980', '1980-1995','1995-']:
        
        fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True, figsize=(20, 15))
        
        c=0
        
        for season in ['winter', 'spring', 'summer', 'fall']:
            
            plot_df_casts = odf_dict[basin][(odf_dict[basin]['period_label'] == period) & (~np.isnan(odf_dict[basin]['DO_mg_L'])) & (odf_dict[basin]['DO_mg_L'] < 50) & (odf_dict[basin]['season'] == season)]
            
            if len(plot_df_casts) > 2:
            
                min_bin = math.floor(plot_df_casts['z'].min())
                
                edges = np.arange(min_bin,1)
                
                sns.scatterplot(data=plot_df_casts, x='DO_mg_L', y ='z', hue='year', palette='flare', alpha=0.5, ax=ax[c])
                
                ax[c].plot(avg_cast_f_dict[basin][period][season](edges), edges, '-k', linewidth=2, label='avg cast')
                        
                ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
                
                ax[c].set_title(season)
                
                ax[c].set_xlabel('DO [mg/L]')
                
                ax[c].set_ylabel('Depth [m]')
                
                c+=1
              
        plt.suptitle(basin + ' ' + period + ' cast data + avg cast (1m bins)')
        
        fig.tight_layout()
            
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + period + '_DO_avg_cast_by_season.png', dpi=500)



# %%

# for 1/9

# SEASONS - var and z DO

# YEARS - var and z DO

# color z by sampling type!

# cast average z!!!

# DO THE AVERAGE CAST PLOT THING FROM VFC thing!!! - seasons, grouped by period labels


# %%

