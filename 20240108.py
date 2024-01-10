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
                                 .groupby(['cid', 'depth_range', 'var']).mean()
                                 .reset_index()
                                 .dropna()
                                 .assign(
                                         segment=(lambda x: key),
                                         season=(lambda x: pd.cut(x['month'],
                                                                  bins=[0,3,6,9,12],
                                                                  labels=['winter', 'spring', 'summer', 'fall'])),
                                         datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x)))) ### convert to 
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




# %%

# for 1/9

# SEASONS - var and z DO

# YEARS - var and z DO

# color z by sampling type!

# cast average z!!!

# DO THE AVERAGE CAST PLOT THING FROM VFC thing!!! - seasons, grouped by period labels


# %%

month_counts_total_dict = dict()

for key in odf_dict.keys():
    
    month_counts_total_dict_temp = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                         .dropna()
                         .set_index('datetime')
                         .groupby([pd.Grouper(freq='M'), 'var']).agg({'cid' :lambda x: x.nunique()})
                         .reset_index()
                         )
    
    month_counts_total_dict[key] = month_counts_total_dict_temp
    
# %%

month_counts_depths_dict = dict()

for key in odf_dict.keys():
    
    month_counts_depths_dict_temp = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                         .dropna()
                         .set_index('datetime')
                         .groupby([pd.Grouper(freq='M'), 'depth_range', 'var']).agg({'cid' :lambda x: x.nunique()})
                         .reset_index()
                         )
    
    month_counts_depths_dict[key] = month_counts_depths_dict_temp
    
# %%

seasonal_counts_dict = dict()

for key in odf_dict.keys():
    
    seasonal_counts_dict_temp = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                         .dropna()
                         .set_index('datetime')
                         .groupby(['season', 'depth_range', 'var']).agg({'cid' :lambda x: x.nunique()})
                         .reset_index()
                         )
    
    seasonal_counts_dict[key] = seasonal_counts_dict_temp
    

# %%

yearly_counts_dict = dict()

for key in odf_dict.keys():
    
    yearly_counts_dict_temp = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_bool', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                         .dropna()
                         .set_index('datetime')
                         .groupby(['year', 'depth_bool', 'var']).agg({'cid' :lambda x: x.nunique()})
                         .reset_index()
                         )
    
    yearly_counts_dict[key] = seasonal_counts_dict_temp


# %%

# full_avgs_dict = dict()

# for key in odf_dict.keys():
    
#     full_avgs_dict[key] = (odf_dict[key]
#                       .set_index('datetime')
#                       .groupby([pd.Grouper(freq='M')]).mean()
#                       .drop(columns =['lat','lon','cid'])
#                       .assign(season=(lambda x: pd.cut(x['month'],
#                                                 bins=[0,3,6,9,12],
#                                                 labels=['winter', 'spring', 'summer', 'fall'])),
#                               # period_label=(lambda x: pd.cut(x['year'], 
#                               #                              bins=[x['year'].min()-1, year_div-1, x['year'].max()],
#                               #                              labels= ['pre', 'post'])),
#                               depth_bool=(lambda x: 'full_depth'),
#                               segment=(lambda x: key)
#                               )
#                       .reset_index()
#                       )
    
# %%


depth_avgs_dict = dict()

for key in odf_dict.keys():
    
    # build in the timeframe flexibility
    
    counts = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_bool', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                         .dropna()
                         .set_index('datetime')
                         .groupby([pd.Grouper(freq='M'), 'depth_bool', 'var']).agg({'cid' :lambda x: x.nunique()})
                         .reset_index()
                         )
    
    depth_avgs_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_bool', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                      .set_index('datetime')
                      .groupby([pd.Grouper(freq='M'), 'depth_bool', 'var']).agg(['mean', 'count', 'std'])
                      .drop(columns =['lat','lon','cid', 'year', 'month', 'date_ordinal'])
                      )
    
    depth_avgs_dict[key].columns = depth_avgs_dict[key].columns.to_flat_index().map('_'.join)
    
    depth_avgs_dict[key] = (depth_avgs_dict[key]
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
    
    depth_avgs_dict[key] = pd.merge(depth_avgs_dict[key], counts, how='left', on=['datetime', 'depth_bool', 'var'])
    
# %%

# depth_avgs_castcounts_dict = dict()

# for key in odf_dict.keys():
    
#     depth_avgs_castcounts_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_bool', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment'],
#                                     value_vars=var_list, var_name='var', value_name = 'val')
#                       .set_index('datetime')
#                       .groupby([pd.Grouper(freq='M'), 'depth_bool', 'var']).agg(['mean', 'count', 'std'])
#                       .drop(columns =['lat','lon','year', 'month', 'date_ordinal'])
#                       )
#                 # keep z_mean and cid_count IN PROGRESS >>>>>>>>
#     depth_avgs_dict[key].columns = depth_avgs_dict[key].columns.to_flat_index().map('_'.join)
    
#     depth_avgs_dict[key] = (depth_avgs_dict[key]
#                       .reset_index()
#                       .drop(columns=['z_std', 'z_count',])
#                       .assign(
#                               # period_label=(lambda x: pd.cut(x['year'], 
#                               #                              bins=[x['year'].min()-1, year_div-1, x['year'].max()],
#                               #                              labels= ['pre', 'post'])),
#                               segment=(lambda x: key),
#                               year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
#                               month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
#                               season=(lambda x: pd.cut(x['month'],
#                                                        bins=[0,3,6,9,12],
#                                                        labels=['winter', 'spring', 'summer', 'fall'])),
#                               date_ordinal=(lambda x: x['datetime'].apply(lambda x: x.toordinal()))
#                               )
#                       )
    
# %%

depth_avgs_df = pd.concat(depth_avgs_dict.values(), ignore_index=True)

# %%

depth_avgs_df['val_ci95hi'] = depth_avgs_df['val_mean'] + 1.96*depth_avgs_df['val_std']/np.sqrt(depth_avgs_df['val_count'])

depth_avgs_df['val_ci95lo'] = depth_avgs_df['val_mean'] - 1.96*depth_avgs_df['val_std']/np.sqrt(depth_avgs_df['val_count'])

# %%
shallow_avgs_df = depth_avgs_df[depth_avgs_df['depth_bool'] == 'shallow']

mid_avgs_df = depth_avgs_df[depth_avgs_df['depth_bool'] == 'mid']

deep_avgs_df  = depth_avgs_df[depth_avgs_df['depth_bool'] == 'deep']

# %%

for basin in basin_list:
    
    fig, ax = plt.subplots(nrows=3, figsize=(10,15), sharex=True)
    
    c = 0
    
    plt.suptitle(basin + ' avg sampling depths', y = 1.0)
    
    for depth_bool in ['shallow', 'mid','deep']:
                    
        plot_df = depth_avgs_dict[basin][(~np.isnan(depth_avgs_dict[basin]['val_mean'])) & (depth_avgs_dict[basin]['var'] == 'DO_mg_L') & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)] #[(~np.isnan(depth_avgs_dict[basin]['DO_mg_L']))] # & (depth_avgs_dict[basin]['depth_bool'] == depth_bool)]
                
        ax[c].scatter(plot_df['datetime'], plot_df['z_mean'], color='k', sizes=[5])
        
        if depth_bool == 'shallow':
            
            ax[c].set_title('shallow (<15m depth)')
        
        elif depth_bool == 'mid':
            
            ax[c].set_title('mid (15m-35m depth)')
            
        elif depth_bool == 'deep':
            
            ax[c].set_title('deep (>35m depth)')
        
        
        ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
        ax[c].set_ylabel('z [m]')
        
        c+=1
        
    ax[2].set_xlabel('Date')
            
    fig.tight_layout()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_avg_DO_z.png', dpi=500)
            
    
# %%

for basin in ['sog_s_wo_si', 'si', 'sog_s']:

    plot_df = deep_avgs_df[(deep_avgs_df['segment'] == basin) & (deep_avgs_df['val_count'] > 1)]
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
    
    plot_df_SA = plot_df[plot_df['var'] == 'SA']
    
    plot_df_CT = plot_df[plot_df['var'] == 'CT']
    
    plot_df_DO = plot_df[plot_df['var'] == 'DO_mg_L']
    
    for idx in plot_df_SA.index:
        
        ax0.vlines(plot_df_SA.loc[idx, 'datetime'], plot_df_SA.loc[idx, 'val_ci95lo'], plot_df_SA.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    for idx in plot_df_CT.index:
        
        ax1.vlines(plot_df_CT.loc[idx, 'datetime'], plot_df_CT.loc[idx, 'val_ci95lo'], plot_df_CT.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)    
    
    for idx in plot_df_DO.index:
        
        ax2.vlines(plot_df_DO.loc[idx, 'datetime'], plot_df_DO.loc[idx, 'val_ci95lo'], plot_df_DO.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
    
    ax0.scatter(plot_df_SA['datetime'], plot_df_SA['val_mean'], color='k', marker='s', sizes=[7])
    
    ax1.scatter(plot_df_CT['datetime'], plot_df_CT['val_mean'], color='k', marker='^', sizes=[7])
    
    ax2.scatter(plot_df_DO['datetime'], plot_df_DO['val_mean'], color='k', sizes=[7])
    
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
    
    ax0.set_title(basin + ' >35m deep')
    
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_SA_CT_DO_deep_ci.png', dpi=500)
    

# %%

seasonal_avgs_dict = dict()

for key in odf_dict.keys():
    
    depth_avgs_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_bool', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                      .set_index('datetime')
                      .groupby([pd.Grouper(freq='M'), 'depth_bool', 'var']).agg(['mean', 'count', 'std'])
                      .drop(columns =['lat','lon','cid', 'z','year', 'month', 'date_ordinal'])
                      )
    
    depth_avgs_dict[key].columns = depth_avgs_dict[key].columns.to_flat_index().map('_'.join)
    
    depth_avgs_dict[key] = (depth_avgs_dict[key]
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
        
        