"""
Work on 1/16/2024
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

# poly_list = ['sog_n', 'sog_s_wo_si', 'si', 'soj','sji','mb', 'wb', 'hc_wo_lc', 'lc', 'ss']

# odf_dict = dfun.getPolyData(Ldir, poly_list)

# with open('/Users/dakotamascarenas/Desktop/big_dict.pkl', 'wb') as f:
#     pickle.dump(odf_dict, f)
    
    # THIS IS CURRENTLY SET TO INCLUDE ECOLOGY BOTTLE DO!!!


# %%

with open('/Users/dakotamascarenas/Desktop/big_dict.pkl', 'rb') as f:
    odf_dict = pickle.load(f)


# %%

basin_list = list(odf_dict.keys())

# %%
# depth_div_0 = -5
# depth_div_1 = -10
# depth_div_2 = -15
# depth_div_3 = -30
# depth_div_4 = -50
# depth_div_5 = -75
# depth_div_6 = -100
# depth_div_7 = -150
# depth_div_8 = -200


period_div_0 = 1945
period_div_1 = 1980
period_div_2 = 1995

#year_div = 2010

for key in odf_dict.keys():
    
    odf_dict[key] = (odf_dict[key]
                     .assign(
                         datetime=(lambda x: pd.to_datetime(x['time'])),
                         # depth_range=(lambda x: pd.cut(x['z'], 
                         #                               bins=[x['z'].min()-1, depth_div_8, depth_div_7, depth_div_6, depth_div_5, depth_div_4, depth_div_3, depth_div_2, depth_div_1, depth_div_0, 0],
                         #                               labels= ['>200m', '150-200m', '100-150m', '75-100m', '50-75m', '30-50m', '15-30m', '10-15m', '5-10m', '<5m'])), # make less manual
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



# remove depth ranges for this calc...

cast_avgs_dict = dict()

for key in odf_dict.keys():
    
    cast_avgs_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'period_label'],
                                         value_vars=var_list, var_name='var', value_name = 'val')
                                 #.set_index('datetime')
                                 .groupby(['cid', 'var']).agg({'val': 'mean', 'z':'mean', 'date_ordinal':'mean', 'month':'first', 'source':'first', 'otype':'first', 'period_label':'first'})
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
                                 
                                 
                                 
# %%


# plot 1 - average depth of bottle casts over time with all bottle cast depths overlaid


for basin in basin_list:
    
    fig, ax = plt.subplots(figsize=(10,15))
    
    plt.rc('font', size=14)
    
    plot_df_cast_means = cast_avgs_dict[basin][(cast_avgs_dict[basin]['otype'] == 'bottle') & (cast_avgs_dict[basin]['var'] == 'DO_mg_L')]
    
    plot_df = odf_dict[basin][(odf_dict[basin]['otype'] == 'bottle') & (~np.isnan(odf_dict[basin]['DO_mg_L'])) & (odf_dict[basin]['DO_mg_L'] >=0) & (odf_dict[basin]['DO_mg_L'] <50)]
    
    sns.scatterplot(data = plot_df, x='datetime', y = 'z', hue='DO_mg_L', alpha=0.5, ax=ax)
    
    #sns.scatterplot(data = plot_df_cast_means, x='datetime', y = 'z', color='k', ax=ax)
    
    ax.set_xlabel('Date')
    
    ax.set_ylabel('DO [mg/L]')
    
    ax.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
    
    ax.set_title(basin + ' bottle DO sampling depths')

    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_bottle_DO_sampling_depths.png', dpi=500)



# %%

for key in odf_dict.keys():

    odf_dict[key] = (odf_dict[key]
                     .assign(
                        # datetime=(lambda x: pd.to_datetime(x['time'])),
                          depth_range=(lambda x: pd.cut(x['z'], 
                                                        bins=[x['z'].min()-1, -165, -135, -105, -80, -65, -55, -47.5, -42.5, -37.5, -32.5, -27.5, -22.5, -17.5, -12.5, -10.5, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, 0],
                                                        labels= ['>165m','135m-165m','105m-135m', '80m-105m', '65m-80m','55m-65m','47.5m-55m','42.5m-47.5m','37.5m-42.5m','32.5m-37.5m','27.5m-32.5m','22.5m-27.5m', '17.5m-22.5m', '12.5m-17.5m', '10.5m-12.5m', '9.5m-10.5m','8.5m-9.5m', '7.5m-8.5m', '6.5m-7.5m', '5.5m-6.5m', '4.5m-5.5m', '3.5m-4.5m', '2.5m-3.5m', '1.5m-2.5m', '<1.5m']))
                          )
                     )# make less manual

# %%


# calculate the mean in each depth range...

depth_period_avgs_dict = dict()

for key in odf_dict.keys():
    
        
    counts = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'period_label', 'source', 'otype'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                         .dropna()
                         #.set_index('datetime')
                         .groupby(['period_label','season', 'depth_range', 'var', 'otype']).agg({'cid' :lambda x: x.nunique()})
                         .reset_index()
                         )
    
    depth_period_avgs_dict[key] = (pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'depth_range', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'period_label'],
                                    value_vars=var_list, var_name='var', value_name = 'val')
                      #.set_index('datetime')
                      .drop(columns=['segment', 'source'])
                      .groupby(['period_label','season', 'depth_range', 'var', 'otype']).agg(['mean', 'std'])
                      .drop(columns =['lat','lon','cid', 'year', 'month'])
                      )
    
    depth_period_avgs_dict[key].columns = depth_period_avgs_dict[key].columns.to_flat_index().map('_'.join)
    
    depth_period_avgs_dict[key] = (depth_period_avgs_dict[key]
                      .drop(columns=['date_ordinal_std'])
                      .rename(columns={'date_ordinal_mean':'date_ordinal'})
                      .reset_index() 
                      .dropna()
                      .assign(
                              segment=(lambda x: key),
                              # year=(lambda x: pd.DatetimeIndex(x['datetime']).year),
                              # month=(lambda x: pd.DatetimeIndex(x['datetime']).month),
                              # season=(lambda x: pd.cut(x['month'],
                              #                          bins=[0,3,6,9,12],
                              #                          labels=['winter', 'spring', 'summer', 'fall'])),
                              datetime=(lambda x: x['date_ordinal'].apply(lambda x: pd.Timestamp.fromordinal(int(x))))
                              )
                      )
    
    depth_period_avgs_dict[key] = pd.merge(depth_period_avgs_dict[key], counts, how='left', on=['period_label', 'season', 'depth_range', 'var', 'otype']).rename(columns={'cid':'cid_count'})
    
    
    # get season/period number of counts
    
    # SA_df = odf_dict[basin][(odf_dict[basin]['var'] == 'SA') & (odf_dict[basin]['season'] == season)]

# %%

depth_period_avgs_df = pd.concat(depth_period_avgs_dict.values(), ignore_index=True)


depth_period_avgs_df['val_ci95hi'] = depth_period_avgs_df['val_mean'] + 1.96*depth_period_avgs_df['val_std']/np.sqrt(depth_period_avgs_df['cid_count'])

depth_period_avgs_df['val_ci95lo'] = depth_period_avgs_df['val_mean'] - 1.96*depth_period_avgs_df['val_std']/np.sqrt(depth_period_avgs_df['cid_count'])


# %%

for basin in basin_list:
    
    for var in ['SA', 'CT', 'DO_mg_L']:
        
        if var =='SA':
            
            marker = 's'
        
        elif var == 'CT':
            
            marker = '^'
            
        else:
            
            marker = 'o'
    
        fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20, 20), squeeze=True, sharey=True, sharex=True)
        
        ax = ax.flatten()
        
        plt.rc('font', size=14)
    
        plot_df = odf_dict[basin][(odf_dict[basin]['otype'] == 'bottle') & (~np.isnan(odf_dict[basin]['DO_mg_L'])) & (odf_dict[basin]['DO_mg_L'] >=0) & (odf_dict[basin]['DO_mg_L'] <50)]
        
        plot_df_avgs = depth_period_avgs_df[(depth_period_avgs_df['segment'] == basin) & (depth_period_avgs_df['otype'] == 'bottle')] # & (depth_period_avgs_df['DO_mg_L'] >=0) & (depth_period_avgs_df['DO_mg_L'] <50)]
    
        c=0
    
        for period in ['-1945', '1945-1980', '1980-1995','1995-']:
            
            for season in ['winter', 'spring', 'summer', 'fall']:
                
                plot_df_use = plot_df[(plot_df['period_label'] == period) & (plot_df['season'] == season)]
                                      
                plot_df_avgs_use = plot_df_avgs[(plot_df_avgs['period_label'] == period) & (plot_df_avgs['season'] == season) & (plot_df_avgs['var'] == var)]
                
                if not plot_df_use.empty:
                                    
                    sns.scatterplot(data = plot_df_use, x=var, y ='z', hue='season', hue_order = ['fall', 'spring', 'summer', 'winter'], palette='husl', marker=marker, ax=ax[c], alpha=0.5, legend=False)
                
                    sns.scatterplot(data = plot_df_avgs_use, x='val_mean', y = 'z_mean', color = 'k', marker=marker, ax=ax[c])
                    
                    ax[c].fill_betweenx(plot_df_avgs_use['z_mean'], plot_df_avgs_use['val_ci95lo'], plot_df_avgs_use['val_ci95hi'], zorder=-4, color='gray', alpha=0.7)
                    
                    # for idx in plot_df_avgs_use.index:
                        
                    #     ax[c].hlines(plot_df_avgs_use.loc[idx, 'z_mean'], plot_df_avgs_use.loc[idx, 'val_ci95lo'], plot_df_avgs_use.loc[idx, 'val_ci95hi'], color='gray', linewidth=1, alpha=0.5)
        
                    #also CI!!!
                    
                    # ax[c].set_xlabel('Date')
            
                    # ax[c].set_ylabel('DO [mg/L]')
            
                    ax[c].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
                    ax[c].set_title(period + ' ' + season)
                        
                    if basin == 'lc':
                        
                        ax[c].set_ylim([-50,0])
                
                c+=1
                
        ax[12].set_xlabel(var)
        
        ax[13].set_xlabel(var)
        
        ax[14].set_xlabel(var)
        
        ax[15].set_xlabel(var)
        
        ax[0].set_ylabel('z [m]')
        
        ax[4].set_ylabel('z [m]')
        
        ax[8].set_ylabel('z [m]')

        ax[12].set_ylabel('z [m]')

        fig.suptitle(basin + ' ' + var +' sampling depths')
        
        fig.tight_layout()
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + basin + '_' + var + '_bottle_sampling_depths.png', dpi=500)
    
