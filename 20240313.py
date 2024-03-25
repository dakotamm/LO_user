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

odf_dict = dfun.getPolyData(Ldir, poly_list, source_list=['ecology', 'nceiSalish', 'collias'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2023))

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


fig, axd = plt.subplot_mosaic([['map', 'skagit_annual', 'skagit_annual'], ['map', 'hat_island', 'near_alki']], layout="constrained", figsize = (16,8))

plt.rc('font', size=14)

lat_lon_hat = odf[(odf['segment'] == 'hat_island') & (odf['var'] == 'DO_mg_L')].groupby('cid').first().reset_index()

lat_lon_alki = odf[(odf['segment'] == 'near_alki') & (odf['var'] == 'DO_mg_L')].groupby('cid').first().reset_index()


sns.scatterplot(data=lat_lon_hat, x='lon', y='lat', color = 'blue', ax = axd['map'])

sns.scatterplot(data=lat_lon_alki, x='lon', y='lat', color = 'red', ax = axd['map'])

pfun.add_coast(axd['map'])

pfun.dar(axd['map'])

axd['map'].set_xlim(-123.2, -122.1)

axd['map'].set_ylim(47,48.7)

sns.lineplot(data=annual_skagit_df, x='year_nu', y='mean_va', ax = axd['skagit_annual'])

sns.lineplot(data=annual_skagit_df, x='year_nu', y='mean_va_mean', ax = axd['skagit_annual'], color='black', linewidth=3)



axd['skagit_annual'].fill_between(annual_skagit_df['year_nu'], annual_skagit_df['val_ci95lo'], annual_skagit_df['val_ci95hi'], color='lightgray', alpha=0.5)


axd['skagit_annual'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)





plot_df = decade_avgs_df[(decade_avgs_df['segment'] == 'hat_island') & (decade_avgs_df['season'] == 'fall') & (decade_avgs_df['var'] == 'DO_mg_L')]


#plot_df_ali = decade_avgs_df[(decade_avgs_df['segment'] == 'near_alki') & (decade_avgs_df['season'] == 'fall') & (decade_avgs_df['var'] == 'DO_mg_L')]

decade0 = '1950'

color0 = 'lightblue'

decade1 = '2010'

color1 = 'blue'


#if not plot_df.empty:
        
sns.lineplot(data = plot_df[plot_df['decade'] == decade0], x='val_mean', y ='z_mean', color = color0, ax=axd['hat_island'], orient='y', legend=False)

sns.lineplot(data = plot_df[plot_df['decade'] == decade1], x='val_mean', y ='z_mean', color = color1, ax=axd['hat_island'], orient='y', legend=False)


                
axd['hat_island'].fill_betweenx(plot_df[plot_df['decade'] == decade0]['z_mean'], plot_df[plot_df['decade'] == decade0]['val_ci95lo'], plot_df[plot_df['decade'] == decade0]['val_ci95hi'],
                 zorder=-4, alpha=0.5, color=color0)

axd['hat_island'].fill_betweenx(plot_df[plot_df['decade'] == decade1]['z_mean'], plot_df[plot_df['decade'] == decade1]['val_ci95lo'], plot_df[plot_df['decade'] == decade1]['val_ci95hi'],
                 zorder=-4, alpha=0.5, color=color1)

        
axd['hat_island'].axvspan(0,2, color = 'lightgray', alpha = 0.2)

axd['hat_island'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
axd['hat_island'].set_xlim(0, 14)
    

axd['hat_island'].set_xlabel('DO [mg/L]')

axd['hat_island'].set_ylabel('z [m]')




plot_df = decade_avgs_df[(decade_avgs_df['segment'] == 'near_alki') & (decade_avgs_df['season'] == 'fall') & (decade_avgs_df['var'] == 'DO_mg_L')]

decade0 = '1950'

color0 = 'pink' #'#8ad6cc'

decade1 = '2010'

color1 = 'red' #'#f97171'

sns.lineplot(data = plot_df[plot_df['decade'] == decade0], x='val_mean', y ='z_mean', color = color0, ax=axd['near_alki'], orient='y', legend=False)

sns.lineplot(data = plot_df[plot_df['decade'] == decade1], x='val_mean', y ='z_mean', color = color1, ax=axd['near_alki'], orient='y', legend=False)


                
axd['near_alki'].fill_betweenx(plot_df[plot_df['decade'] == decade0]['z_mean'], plot_df[plot_df['decade'] == decade0]['val_ci95lo'], plot_df[plot_df['decade'] == decade0]['val_ci95hi'],
                 zorder=-4, alpha=0.5, color=color0)

axd['near_alki'].fill_betweenx(plot_df[plot_df['decade'] == decade1]['z_mean'], plot_df[plot_df['decade'] == decade1]['val_ci95lo'], plot_df[plot_df['decade'] == decade1]['val_ci95hi'],
                 zorder=-4, alpha=0.5, color=color1)

        
axd['near_alki'].axvspan(0,2, color = 'lightgray', alpha = 0.2)

axd['near_alki'].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
axd['near_alki'].set_xlim(0, 14)
    

axd['near_alki'].set_xlabel('DO [mg/L]')

axd['near_alki'].set_ylabel('z [m]')

fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/prelim_skagit_decadal_hat_alki.png', dpi=500, transparent=False)

