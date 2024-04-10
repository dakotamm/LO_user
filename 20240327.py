#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:53:56 2024

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

poly_list = ['admiralty_sill', 'budd_inlet', 'carr_inlet_mid', 'dana_passage', 'hat_island', 'hazel_point', 'hood_canal_mouth', 'lynch_cove_shallow', 'near_alki', 'near_edmonds', 'port_susan_mid', 'saratoga_passage_north', 'ps', 'hc_wo_lc']

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['ecology', 'nceiSalish', 'collias', 'kc', 'kc_taylor', 'kc_whidbey'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2024))

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

lc_exclude = odf[(odf['segment'] == 'lynch_cove_shallow') & (odf['z'] < -45)]

# %%

odf = odf[~odf['cid'].isin(lc_exclude['cid'].unique())]
    
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

odf[(odf['source'] == 'ecology') & (odf['var'] == 'DO_mg_L') & (odf['otype'] == 'bottle')] = np.nan

# %%

decade_std_temp = (odf#drop(columns=['segment', 'source'])
                  .groupby(['decade','season', 'segment', 'depth_range', 'var']).agg({'val':['mean', 'std']}) #, 'z':['mean'], 'date_ordinal':['mean']})
                  #.drop(columns =['lat','lon','cid', 'year', 'month'])
                  )

decade_std_temp.columns = decade_std_temp.columns.to_flat_index().map('_'.join)

decade_std_temp = decade_std_temp.reset_index()


# %%

odf_temp = pd.merge(odf, decade_std_temp, how='left', on=['decade', 'season', 'segment','depth_range', 'var'])

# %%

odf_decade = odf_temp[(odf_temp['val'] >= odf_temp['val_mean'] - 2*odf_temp['val_std']) & (odf_temp['val'] <= odf_temp['val_mean'] + 2*odf_temp['val_std'])]

odf_decade = odf_decade.drop(columns = ['val_mean', 'val_std'])

# %%

decade_counts = (odf_decade
                     .dropna()
                     #.set_index('datetime')
                     .groupby(['decade','season', 'segment', 'depth_range', 'var']).agg({'cid' :lambda x: x.nunique()})
                     .reset_index()
                     .rename(columns={'cid':'cid_count'})
                     )

# %%


decade_avgs_df = (odf_decade#drop(columns=['segment', 'source'])
                  .groupby(['decade','season', 'segment', 'depth_range', 'var']).agg({'val':['mean', 'std'], 'z':['mean'], 'date_ordinal':['mean']})
                  #.drop(columns =['lat','lon','cid', 'year', 'month'])
                  )


decade_avgs_df.columns = decade_avgs_df.columns.to_flat_index().map('_'.join)


# %%

decade_avgs_df = (decade_avgs_df
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

decade_avgs_df = pd.merge(decade_avgs_df, decade_counts, how='left', on=['decade', 'season', 'segment','depth_range', 'var'])

# %%

decade_avgs_df = decade_avgs_df[decade_avgs_df['cid_count'] >1]

decade_avgs_df['val_ci95hi'] = decade_avgs_df['val_mean'] + 1.96*decade_avgs_df['val_std']/np.sqrt(decade_avgs_df['cid_count'])

decade_avgs_df['val_ci95lo'] = decade_avgs_df['val_mean'] - 1.96*decade_avgs_df['val_std']/np.sqrt(decade_avgs_df['cid_count'])


# %%

annual_skagit_df = pd.read_csv('/Users/dakotamascarenas/Desktop/skagit_annual.txt',sep='\t',header=(34), skiprows=(35,35))


# %%

annual_skagit_df = annual_skagit_df.assign(
                    decade=(lambda x: pd.cut(x['year_nu'],
                         bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                         labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True))
)

# %%

skagit_decadal_means = (annual_skagit_df#drop(columns=['segment', 'source'])
                  .groupby(['decade']).agg({'mean_va':['mean', 'std']}))
                  
skagit_decadal_means.columns = skagit_decadal_means.columns.to_flat_index().map('_'.join)

skagit_decadal_means = skagit_decadal_means.reset_index()

# %%

annual_skagit_df = pd.merge(annual_skagit_df, skagit_decadal_means, how='left', on=['decade'])

# %%

skagit_decade_counts = (annual_skagit_df
                     .dropna()
                     #.set_index('datetime')
                     .groupby(['decade']).agg({'year_nu' :lambda x: x.nunique()})
                     .reset_index()
                     .rename(columns={'year_nu':'decade_counts'})
                     )

# %%

annual_skagit_df = pd.merge(annual_skagit_df, skagit_decade_counts, how='left', on=['decade'])


# %%

annual_skagit_df = annual_skagit_df[annual_skagit_df['decade_counts'] >1]

annual_skagit_df['val_ci95hi'] = annual_skagit_df['mean_va_mean'] + 1.96*annual_skagit_df['mean_va_std']/np.sqrt(annual_skagit_df['decade_counts'])

annual_skagit_df['val_ci95lo'] = annual_skagit_df['mean_va_mean'] - 1.96*annual_skagit_df['mean_va_std']/np.sqrt(annual_skagit_df['decade_counts'])


# %%

# %%

# %%

monthly_skagit_df = pd.read_csv('/Users/dakotamascarenas/Desktop/skagit_monthly.txt',sep='\t',header=(35), skiprows=(36,36))


# %%

monthly_skagit_df = monthly_skagit_df.assign(
                    decade=(lambda x: pd.cut(x['year_nu'],
                         bins=[1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009, 2019, 2029],
                         labels=['1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020'], right=True)))

# %%

monthly_skagit_df['day'] = 1

monthly_skagit_df['datetime'] = pd.to_datetime(dict(year=monthly_skagit_df['year_nu'], month=monthly_skagit_df['month_nu'], day=monthly_skagit_df['day']))

# %%

skagit_decadal_means_m = (monthly_skagit_df#drop(columns=['segment', 'source'])
                  .groupby(['decade']).agg({'mean_va':['mean', 'std']}))
                  
skagit_decadal_means_m.columns = skagit_decadal_means_m.columns.to_flat_index().map('_'.join)

skagit_decadal_means_m = skagit_decadal_means_m.reset_index()

# %%

monthly_skagit_df = pd.merge(monthly_skagit_df, skagit_decadal_means_m, how='left', on=['decade'])

# %%

skagit_decade_counts_m = (monthly_skagit_df
                     .dropna()
                     #.set_index('datetime')
                     .groupby(['decade']).agg({'year_nu' :lambda x: x.nunique()})
                     .reset_index()
                     .rename(columns={'year_nu':'decade_counts'})
                     )

# %%

monthly_skagit_df = pd.merge(monthly_skagit_df, skagit_decade_counts_m, how='left', on=['decade'])


# %%

monthly_skagit_df = monthly_skagit_df[monthly_skagit_df['decade_counts'] >1]

monthly_skagit_df['val_ci95hi'] = monthly_skagit_df['mean_va_mean'] + 1.96*monthly_skagit_df['mean_va_std']/np.sqrt(monthly_skagit_df['decade_counts'])

monthly_skagit_df['val_ci95lo'] = monthly_skagit_df['mean_va_mean'] - 1.96*monthly_skagit_df['mean_va_std']/np.sqrt(monthly_skagit_df['decade_counts'])

# %%

annual_skagit_df['month'] = 1

annual_skagit_df['day'] = 1

annual_skagit_df['datetime'] = pd.to_datetime(dict(year=annual_skagit_df['year_nu'], month=annual_skagit_df['month'], day=annual_skagit_df['day']))


# %%

mosaic = [['skagit_annual', 'skagit_annual', 'skagit_annual', 'skagit_annual','skagit_annual', 'skagit_annual', 'skagit_annual', 'skagit_annual']]

for basin in basin_list:
    
    new_list = []
    
    for decade in ['1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010']:

        new_list.append(basin + '_' + decade)
        
    mosaic.append(new_list)
    
# %%


fig, axd = plt.subplot_mosaic(mosaic, layout="constrained", figsize = (20,30)) 


#sns.lineplot(data=monthly_skagit_df, x='datetime', y='mean_va', ax = axd['skagit_annual'], color='lightgray', label = 'Monthly')

sns.lineplot(data=annual_skagit_df, x='datetime', y='mean_va', ax = axd['skagit_annual'], color='gray', label = 'Yearly')


sns.lineplot(data=monthly_skagit_df, x='datetime', y='mean_va_mean', ax = axd['skagit_annual'], color='black', label = 'Decadal')


axd['skagit_annual'].legend(loc = 'upper right')



#axd['skagit_annual'].fill_between(monthly_skagit_df['datetime'], monthly_skagit_df['val_ci95lo'], monthly_skagit_df['val_ci95hi'], color='lightgray', alpha=0.5)


axd['skagit_annual'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)

axd['skagit_annual'].set_xlim(datetime.date(1940, 1, 1), datetime.date(2019, 12,31))

axd['skagit_annual'].set_ylabel('Discharge [cfs]')

axd['skagit_annual'].set(xlabel=None)

axd['skagit_annual'].text(0.01 , 0.85, 'Skagit River Mean Discharge', transform=axd['skagit_annual'].transAxes, bbox = {'facecolor': 'white', 'alpha':0.5,'boxstyle': "round,pad=0.3"})




for basin in basin_list:

    plot_df_big = decade_avgs_df[(decade_avgs_df['segment'] == basin) & (decade_avgs_df['season'] == 'fall') & (decade_avgs_df['var'] == 'DO_mg_L')]


    for decade in ['1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010']:
        
        #if basin == 'hat_island':
            
        color0 = 'black'
        
        ax_name = basin + '_' + decade
        
        if decade == '1940':
            
            axd[ax_name].text(0.05 , 0.95, basin.replace('_', ' ').title() + ' ->', transform=axd[ax_name].transAxes, bbox = {'facecolor': 'white', 'alpha':0.5,'boxstyle': "round,pad=0.3"}, verticalalignment = 'top')

           
                        
        
        for decade_ in ['1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010']:
        
            if (decade_ == '1940') and (decade == '1940'):
                
                axd[ax_name].fill_betweenx(plot_df_big[plot_df_big['decade'] == decade_]['z_mean'], plot_df_big[plot_df_big['decade'] == decade_]['val_ci95lo'], plot_df_big[plot_df_big['decade'] == decade_]['val_ci95hi'],
                         zorder=-4, alpha=0.5, color='lightgray', label = 'all decades')
                
                axd[ax_name].legend(loc ='lower right')
                
            else:
                
                axd[ax_name].fill_betweenx(plot_df_big[plot_df_big['decade'] == decade_]['z_mean'], plot_df_big[plot_df_big['decade'] == decade_]['val_ci95lo'], plot_df_big[plot_df_big['decade'] == decade_]['val_ci95hi'],
                         zorder=-4, alpha=0.5, color='lightgray')
                        
        plot_df = decade_avgs_df[(decade_avgs_df['segment'] == basin) & (decade_avgs_df['decade'] == decade) & (decade_avgs_df['season'] == 'fall') & (decade_avgs_df['var'] == 'DO_mg_L')]
        

        if not plot_df.empty:
            
            sns.lineplot(data = plot_df, x='val_mean', y ='z_mean', color = color0, ax=axd[ax_name], orient='y', legend=False)

            axd[ax_name].fill_betweenx(plot_df['z_mean'], plot_df['val_ci95lo'], plot_df['val_ci95hi'],
                             zorder=-4, alpha=0.5, color=color0, label = decade + 's')
            
            axd[ax_name].legend(loc ='lower right')
            
        axd[ax_name].set_xlim(0, 14)
        
        axd[ax_name].set_ylim(-200, 0)
        
        if decade == '1940':

            axd[ax_name].set_ylabel('z [m]')
            
        else:
             
            axd[ax_name].set(ylabel=None)
            
            axd[ax_name].set(yticklabels=[])
        
        if basin == 'hc_wo_lc':
            
            axd[ax_name].set_xlabel('DO [mg/L]')

        else:
             
            axd[ax_name].set(xlabel=None)
            
            axd[ax_name].set(xticklabels=[]) 

        
        #axd[ax_name].set_title(decade)
        
        axd[ax_name].axvspan(0,2, color = 'lightgray', alpha = 0.2)

        axd[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)

        #axd[ax_name].text(0.05 , 0.95, basin.replace('_', ' ').title() + ' >', transform=axd[ax_name].transAxes, verticalalignment = 'bottom')
        
           


#fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/prelim_skagit_decadal_all_w_KC.png', dpi=500, transparent=False)

