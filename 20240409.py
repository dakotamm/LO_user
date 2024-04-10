#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:11:45 2024

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


# %%

Ldir = Lfun.Lstart(gridname='cas7')

# %%

poly_list = ['wb']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['collias', 'ecology', 'kc', 'kc_taylor', 'kc_whidbey', 'nceiSalish'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))

# %%

basin_list = list(odf_dict.keys())

var_list = ['SA', 'CT', 'NO3 (uM)', 'Chl (mg m-3)', 'NO2 (uM)', 'SiO4 (uM)', 'NH4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO_mg_L', 'DO (uM)']

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
            
    odf_dict[key] = pd.melt(odf_dict[key], id_vars=['cid', 'lon', 'lat', 'time', 'datetime', 'z', 'year', 'month', 'season', 'date_ordinal', 'segment', 'source', 'otype', 'decade'],
                                         value_vars=var_list, var_name='var', value_name = 'val')
    
# %%

odf = pd.concat(odf_dict.values(), ignore_index=True)

odf = odf[odf['var'].isin(['DO_mg_L', 'CT', 'SA'])]

#odf = odf[(odf['val'] >= 0) & (odf['val'] <50)]
    
# %%

#lc_exclude = odf[(odf['segment'] == 'lynch_cove_shallow') & (odf['z'] < -45)]

# %%

#odf = odf[~odf['cid'].isin(lc_exclude['cid'].unique())]


# %%

odf['source_type'] = odf['source'] + '_' + odf['otype']

# %%

# toward per decade average profiles, smoothed (bigger bins) - on same plot


odf = (odf
            .assign(
               # datetime=(lambda x: pd.to_datetime(x['time'])),
                 depth_range=(lambda x: pd.cut(x['z'], 
                                               bins=[-700, -355, -275, -205, -165, -135, -105, -80, -55, -37.5, -27.5, -17.5, -7.5, 0],
                                               labels= ['>355m', '275m-355m', '205m-275m', '165-205m','135m-165m','105m-135m', '80m-105m', '65m-80m','55m-80m','27.5m-37.5m', '17.5m-27.5m', '7.5m-17.5m', '<7.5m']))
                 # decade=(lambda x: pd.cut(x['year'],
                 #                          bins=[1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030],
                 #                          labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020']))
                 
                 
                 )
            )# make less manual

# %%

# annual averaging

annual_std_temp = (odf#drop(columns=['segment', 'source'])
                  .groupby(['year','season', 'segment', 'depth_range', 'var', 'otype']).agg({'val':['mean', 'std']}) #, 'z':['mean'], 'date_ordinal':['mean']})
                  #.drop(columns =['lat','lon','cid', 'year', 'month'])
                  )

annual_std_temp.columns = annual_std_temp.columns.to_flat_index().map('_'.join)

annual_std_temp = annual_std_temp.reset_index()


# %%

odf_temp = pd.merge(odf, annual_std_temp, how='left', on=['year', 'season', 'segment','depth_range', 'var', 'otype'])

# %%

odf_annual = odf_temp[(odf_temp['val'] >= odf_temp['val_mean'] - 2*odf_temp['val_std']) & (odf_temp['val'] <= odf_temp['val_mean'] + 2*odf_temp['val_std'])]

odf_annual = odf_annual.drop(columns = ['val_mean', 'val_std'])

# %%

annual_counts = (odf_annual
                     .dropna()
                     #.set_index('datetime')
                     .groupby(['year','season', 'segment', 'depth_range', 'var', 'otype']).agg({'cid' :lambda x: x.nunique()})
                     .reset_index()
                     .rename(columns={'cid':'cid_count'})
                     )

# %%


annual_avgs_df = (odf_annual#drop(columns=['segment', 'source'])
                  .groupby(['year','season', 'segment', 'depth_range', 'var', 'otype']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
                  #.drop(columns =['lat','lon','cid', 'year', 'month'])
                  )


annual_avgs_df.columns = annual_avgs_df.columns.to_flat_index().map('_'.join)


# %%

annual_avgs_df = (annual_avgs_df
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

annual_avgs_df = pd.merge(annual_avgs_df, annual_counts, how='left', on=['year', 'season', 'segment','depth_range', 'var', 'otype'])

# %%

annual_avgs_df = annual_avgs_df[annual_avgs_df['cid_count'] >1]

annual_avgs_df['val_ci95hi'] = annual_avgs_df['val_mean'] + 1.96*annual_avgs_df['val_std']/np.sqrt(annual_avgs_df['cid_count'])

annual_avgs_df['val_ci95lo'] = annual_avgs_df['val_mean'] - 1.96*annual_avgs_df['val_std']/np.sqrt(annual_avgs_df['cid_count'])


# %%

# %%

year_range = np.arange(2016,2019)

mosaic = []


for var in ['SA', 'CT', 'DO_mg_L']:
    
    new_list = ['map']
    
    for year in year_range:

        new_list.append(str(year) + '_' + var)
        
    mosaic.append(new_list)
    
season = 'summer'

year_min = year_range.min()

year_max = year_range.max()    



fig, ax = plt.subplot_mosaic(mosaic, layout="constrained", figsize = (10,10))

plot_df_map = odf[(odf['segment'] == 'wb') & (odf['season'] == season) & (odf['year'].isin(year_range))].groupby('cid').first().reset_index()

sns.scatterplot(data=plot_df_map, x='lon', y='lat', hue='year', palette='Set2', ax = ax['map'], s = 100, alpha=0.5)

#sns.scatterplot(data=plot_df_mean, x='lon', y='lat', hue='decade', palette='Set2', marker='s', sizes=20)

ax['map'].autoscale(enable=False)

pfun.add_coast(ax['map'])

pfun.dar(ax['map'])

ax['map'].set_xlim(-123.2, -122.1)

ax['map'].set_ylim(47,48.5)

ax['map'].set_title(season + ' sampling locations')

ax['map'].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol = 2)



for var in ['SA', 'CT', 'DO_mg_L']:
    
    plot_df_big = annual_avgs_df[(annual_avgs_df['season'] == season) & (annual_avgs_df['segment'] == 'wb') & (annual_avgs_df['var'] == var)]
    
    if var =='SA':
        
        marker = 's'
        
        xmin = 22
        
        xmax = 32
        
        color = 'blue'
        
        xlabel = 'SA [psu]'
    
    elif var == 'CT':
        
        marker = '^'
        
        xmin = 7
        
        xmax = 17
        
        color = 'red'
        
        xlabel = ['CT [deg C]']
        
    else:
        
        marker = 'o'
        
        xmin = 0
        
        xmax = 12
        
        color = 'black'
        
        xlabel = ['DO [mg/L]']
        
    for year in year_range:
        
        ax_name = str(year) + '_' + var
                
        for year_ in year_range:
        
            if (year_ == year_min) and (year == year_min):
                
                ax[ax_name].fill_betweenx(plot_df_big[plot_df_big['year'] == year_]['z_mean'], plot_df_big[plot_df_big['year'] == year_]['val_ci95lo'], plot_df_big[plot_df_big['year'] == year_]['val_ci95hi'],
                         zorder=-4, alpha=0.5, color='lightgray', label = 'all decades')
                
                ax[ax_name].legend(loc ='lower right')
                
            else:
                
                ax[ax_name].fill_betweenx(plot_df_big[plot_df_big['year'] == year_]['z_mean'], plot_df_big[plot_df_big['year'] == year_]['val_ci95lo'], plot_df_big[plot_df_big['year'] == year_]['val_ci95hi'],
                         zorder=-4, alpha=0.5, color='lightgray')
                        
        plot_df = plot_df_big[plot_df_big['year'] == year]
        
        if not plot_df.empty:
            
            sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', color = color, ax=ax[ax_name], orient='y', legend=False)

            ax[ax_name].fill_betweenx(plot_df['z_mean'], plot_df['val_ci95lo'], plot_df['val_ci95hi'],
                             zorder=-4, alpha=0.5, color=color, label = str(year))
            
            ax[ax_name].legend(loc ='lower right')
            
        ax[ax_name].set_xlim(xmin, xmax)
        
        ax[ax_name].set_ylim(-200, 0)
        
        if year == year_min:

            ax[ax_name].set_ylabel('z [m]')
            
        else:
             
            ax[ax_name].set(ylabel=None)
            
            ax[ax_name].set(yticklabels=[])
        
        ax[ax_name].set_xlabel(xlabel)
        
        
        if var == 'DO_mg_L':
                
            ax[ax_name].axvspan(0,2, color = 'lightgray', alpha = 0.2)

        ax[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)

        #axd[ax_name].text(0.05 , 0.95, basin.replace('_', ' ').title() + ' >', transform=axd[ax_name].transAxes, verticalalignment = 'bottom')
        
           


#fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/wb_DO_SA_CT_' + season +'_' + str(year_min) + '-' + str(year_max) + '.png', dpi=500, transparent=False)
        
        
    
    
    
    

