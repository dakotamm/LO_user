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

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/prelim_skagit_decadal_all.png', dpi=500, transparent=False)


# %%

from matplotlib.patches import Rectangle
import cmocean


ds = xr.open_dataset('/Users/dakotamascarenas/LO_data/grids/cas7/grid.nc')
z = -ds.h.values
mask_rho = np.transpose(ds.mask_rho.values)
lon = ds.lon_rho.values
lat = ds.lat_rho.values
X = lon[0,:] # grid cell X values
Y = lat[:,0] # grid cell Y values
plon, plat = pfun.get_plon_plat(lon,lat)
# make a version of z with nans where masked
zm = z.copy()
zm[np.transpose(mask_rho) == 0] = np.nan
zm[np.transpose(mask_rho) != 0] = -1

j1 = 570
j2 = 1170
i1 = 220
i2 = 652



# other direction

# mosaic = [['map', 'dana_passage_SA', 'hazel_point_SA', 'near_edmonds_SA', 'hat_island_SA'],
          
#           ['map', 'dana_passage_CT', 'hazel_point_CT', 'near_edmonds_CT', 'hat_island_CT'],
          
#           ['map', 'dana_passage_DO_mg_L', 'hazel_point_DO_mg_L', 'near_edmonds_DO_mg_L', 'hat_island_DO_mg_L']]

mosaic = [['map', 'hat_island_SA', 'hat_island_CT', 'hat_island_DO_mg_L'],
          ['map', 'near_edmonds_SA', 'near_edmonds_CT', 'near_edmonds_DO_mg_L'],
          ['map', 'hazel_point_SA', 'hazel_point_CT', 'hazel_point_DO_mg_L'],
          ['map', 'dana_passage_SA', 'dana_passage_CT', 'dana_passage_DO_mg_L']
          ]


fig, axd = plt.subplot_mosaic(mosaic, layout="constrained", figsize = (15,8)) 



for season in ['fall']:
    
    lat_lon_df = odf[(odf['segment'].isin(['dana_passage', 'hazel_point', 'near_edmonds', 'hat_island'])) & (odf['var'] == 'DO_mg_L') & (odf['season'] ==  season)].groupby('segment').first().reset_index()
            
    axd['map'].add_patch(Rectangle((X[i1], Y[j1]), -121.4-X[i1],Y[j2]-Y[j1], facecolor='white'))
    axd['map'].pcolormesh(plon, plat, zm, linewidth=0.5, vmin=-6, vmax=0, cmap = 'gray') #cmap=plt.get_cmap(cmocean.cm.ice))
    # format
    #axd['map'].axes.xaxis.set_visible(False)
    #axd['map'].axes.yaxis.set_visible(False)
    axd['map'].set_xlim(X[i1],-121.4)#X[i2]) # Salish Sea
    axd['map'].set_ylim(Y[j1],Y[j2]) # Salish Sea
    
    sns.scatterplot(data=lat_lon_df, x='lon', y='lat', color = 'white', edgecolor='black', ax = axd['map'], s=200, legend =False)

    
    for basin in ['dana_passage', 'hazel_point', 'near_edmonds', 'hat_island']:
        
        if basin == 'near_edmonds':
            
            axd['map'].text(lat_lon_df.loc[lat_lon_df['segment'] == basin,'lon'].values[0] - 0.05, lat_lon_df.loc[lat_lon_df['segment'] == basin, 'lat'].values[0] + 0.035, 'Triple Junction',
                            bbox = {'facecolor': 'white', 'alpha':0.8, 'boxstyle': "round,pad=0.3"}, verticalalignment = 'bottom', horizontalalignment = 'right')
            
        elif basin == 'hat_island':
            
            axd['map'].text(lat_lon_df.loc[lat_lon_df['segment'] == basin,'lon'].values[0] - 0.05, lat_lon_df.loc[lat_lon_df['segment'] == basin, 'lat'].values[0] + 0.05, basin.replace('_', ' ').title(),
                            bbox = {'facecolor': 'white', 'alpha':0.8,'boxstyle': "round,pad=0.3"}, verticalalignment = 'bottom', horizontalalignment = 'right')
        else: 
            
            axd['map'].text(lat_lon_df.loc[lat_lon_df['segment'] == basin,'lon'].values[0], lat_lon_df.loc[lat_lon_df['segment'] == basin, 'lat'].values[0] - 0.07, basin.replace('_', ' ').title(),
                            bbox = {'facecolor': 'white', 'alpha':0.8,'boxstyle': "round,pad=0.3"}, verticalalignment = 'top', horizontalalignment = 'center')    
    
        
        # if basin == 'near_edmonds':
            
        #     axd['map'].text(lat_lon_df['lon'], lat_lon_df['lat'], 'Triple Junction')
    
            
        # else:s
            
        #     axd['map'].text(lat_lon_df['lon'], lat_lon_df['lat'], basin.replace('_', ' ').title())
    
        for var in ['SA', 'CT', 'DO_mg_L']:
            
            ax_name = basin + '_' + var
        
            if var =='SA':
                
                marker = 's'
                
                xmin = 22
                
                xmax = 32
                
                label = 'Salinity [PSU]'
                        
            elif var == 'CT':
                
                marker = '^'
                
                xmin = 7
                
                xmax = 17
                
                label = 'Temperature [deg C]'
                
            else:
                
                marker = 'o'
                
                xmin = 0
                
                xmax = 12
                
                label = 'DO [mg/L]'
                
            plot_df = decade_avgs_df[(decade_avgs_df['segment'] == basin) & (decade_avgs_df['var'] == var) & (decade_avgs_df['season'] == season)]
            
            decade0 = '1950'
            
            color0 =  'gray' #'#8ad6cc'
            
            decade1 = '2010'
            
            color1 = 'black' #'#f97171'
    
            if not plot_df.empty:
                    
                sns.lineplot(data = plot_df[plot_df['decade'] == decade0], x='val_mean', y ='z_mean', color = color0, ax=axd[ax_name], orient='y', legend=False)
                
                sns.lineplot(data = plot_df[plot_df['decade'] == decade1], x='val_mean', y ='z_mean', color = color1, ax=axd[ax_name], orient='y', legend=False)

                                
                axd[ax_name].fill_betweenx(plot_df[plot_df['decade'] == decade0]['z_mean'], plot_df[plot_df['decade'] == decade0]['val_ci95lo'], plot_df[plot_df['decade'] == decade0]['val_ci95hi'],
                                 zorder=-4, alpha=0.5, color=color0, label = decade0 + 's')
                
                axd[ax_name].fill_betweenx(plot_df[plot_df['decade'] == decade1]['z_mean'], plot_df[plot_df['decade'] == decade1]['val_ci95lo'], plot_df[plot_df['decade'] == decade1]['val_ci95hi'],
                                 zorder=-4, alpha=0.5, color=color1, label=decade1 + 's')
                
                #axd[ax_name].legend()
                
                if var == 'DO_mg_L':
                    
                    axd[ax_name].axvspan(0,2, color = 'lightgray', alpha = 0.2)
                    
                if var == 'SA':
                     
                    if basin == 'near_edmonds':
                    
                     
                        axd[ax_name].text(0.05, 0.05, 'Triple Junction ->', transform=axd[ax_name].transAxes, bbox = {'facecolor': 'white', 'alpha':0.5,'boxstyle': "round,pad=0.3"}, verticalalignment = 'bottom')
                        
                    else:
                        
                        axd[ax_name].text(0.05 , 0.05, basin.replace('_', ' ').title() + ' ->', transform=axd[ax_name].transAxes, bbox = {'facecolor': 'white', 'alpha':0.5,'boxstyle': "round,pad=0.3"}, verticalalignment = 'bottom')

                    
        
                axd[ax_name].grid(color = 'lightgray', linestyle = '--', alpha=0.5)
                        
                axd[ax_name].set_xlim(xmin, xmax)
                
                axd[ax_name].set_ylim(-200, 0)
                    
                
                axd[ax_name].set_xlabel(label)
                
                
                
                axd[ax_name].set_ylabel('z [m]')
                
                if not var == 'SA':
                    
                    axd[ax_name].set(ylabel=None)
                    
                    axd[ax_name].set(yticklabels=[])  
                    
                if not basin == 'dana_passage':
                    
                    axd[ax_name].set(xlabel=None)
                    
                    axd[ax_name].set(xticklabels=[])  

axd['hat_island_SA'].legend()



pfun.add_coast(axd['map'])

pfun.dar(axd['map'])

axd['map'].set_xlim(-123.5, -122)

axd['map'].set_ylim(46.9,48.5)

axd['map'].set(xlabel=None)
 
axd['map'].set(ylabel=None)

axd['map'].tick_params(axis='x', labelrotation=45)

#plt.subplots_adjust(top = 0.5, bottom = 0.5)



#fig.suptitle(basin +' average casts')

#fig.tight_layout()
    
plt.savefig('/Users/dakotamascarenas/Desktop/pltz/map_average_casts_decade_season_CI_1950_2010_FALL.png', dpi=500, transparent=False)

