"""
IDK YET

Test on mac in ipython:
run plot_time_his -gtx cas6_v0_live -test False

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
from lo_tools import plotting_functions as pfun
import pickle

import numpy as np

import VFC_functions as vfun

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import cm

from datetime import datetime, date

import os

# %%

SMALL_SIZE =12
MEDIUM_SIZE = 16
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%

threshold_val = 2

threshold_depth = -40

seg_str = ['basins']

#years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2020]


years = np.arange(1930,2023)

# %%

#years = [2017, 2018, 2019]

vol_df = pd.DataFrame()

wtd_avg_df = pd.DataFrame()

avg_df = pd.DataFrame()

for year in years:
    
    for segs in seg_str:
    
        fn_cid = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/cid_dict.pkl'
        
        fn_vol_df = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/vol_df.p'
        
        fn_wtd_avg = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/wtd_avg_df_NEW.p'
        
        fn_avg = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/avg_df_NEW.p'
        
        if os.path.isfile(fn_cid):
            
            vol_df_temp = pd.read_pickle(fn_vol_df)
        
            vol_df_temp['year'] = year
        
            vol_df_temp['day'] = 1
            
            wtd_avg_df_temp = pd.read_pickle(fn_wtd_avg)
        
            wtd_avg_df_temp['year'] = year
        
            wtd_avg_df_temp['day'] = 1
            
            avg_df_temp = pd.read_pickle(fn_avg)
        
            avg_df_temp['year'] = year
        
            avg_df_temp['day'] = 1
                        
            with open(fn_cid, 'rb') as f: 
                cid_dict = pickle.load(f)
            
            vol_df_temp['num_casts'] = np.nan
            
            wtd_avg_df_temp['num_casts'] = np.nan
            
            avg_df_temp['num_casts'] = np.nan
            
            cid_df = pd.DataFrame.from_dict(cid_dict)
            
            for col in cid_df.columns:
                
                for m in cid_df.index:
                
                    vol_df_temp.loc[(vol_df_temp['month'] == m) & (vol_df_temp['segment'] == col), 'num_casts'] = len(cid_df.loc[m,col])
                    
                    wtd_avg_df_temp.loc[(wtd_avg_df_temp['month'] == m) & (wtd_avg_df_temp['segment'] == col), 'num_casts'] = len(cid_df.loc[m,col])
                    
                    avg_df_temp.loc[(avg_df_temp['month'] == m) & (avg_df_temp['segment'] == col), 'num_casts'] = len(cid_df.loc[m,col])

                    
            vol_df_temp.loc[vol_df_temp['num_casts'] == 0, 'num_casts'] = np.nan

            vol_df = pd.concat([vol_df, vol_df_temp], ignore_index=True)
            
            wtd_avg_df_temp.loc[wtd_avg_df_temp['num_casts'] == 0, 'num_casts'] = np.nan

            wtd_avg_df = pd.concat([wtd_avg_df, wtd_avg_df_temp], ignore_index=True)
            
            avg_df_temp.loc[avg_df_temp['num_casts'] == 0, 'num_casts'] = np.nan

            avg_df = pd.concat([avg_df, avg_df_temp], ignore_index=True)
            
if seg_str[0] == 'basins':
    
    vol_df = vol_df[(vol_df['segment'] != 'Strait of Georgia') & (vol_df['segment'] != 'Strait of Juan de Fuca')]

    wtd_avg_df = wtd_avg_df[(wtd_avg_df['segment'] != 'Strait of Georgia') & (wtd_avg_df['segment'] != 'Strait of Juan de Fuca')]
    
    avg_df = avg_df[(avg_df['segment'] != 'Strait of Georgia') & (avg_df['segment'] != 'Strait of Juan de Fuca')]
    
    
vol_df['date'] = pd.to_datetime(dict(year=vol_df.year, month=vol_df.month, day=vol_df.day))

vol_df['date_ordinal'] = vol_df['date'].apply(lambda date: date.toordinal())

wtd_avg_df['date'] = pd.to_datetime(dict(year=wtd_avg_df.year, month=wtd_avg_df.month, day=wtd_avg_df.day))

wtd_avg_df['date_ordinal'] = wtd_avg_df['date'].apply(lambda date: date.toordinal())

avg_df['date'] = pd.to_datetime(dict(year=avg_df.year, month=avg_df.month, day=avg_df.day))

avg_df['date_ordinal'] = avg_df['date'].apply(lambda date: date.toordinal())


# %%

vol_df['season'] = np.nan

wtd_avg_df['season'] = np.nan

avg_df['season'] = np.nan


vol_df.loc[vol_df['month'].isin([1,2,3]), 'season'] = 'winter'

vol_df.loc[vol_df['month'].isin([4,5,6]), 'season'] = 'spring'

vol_df.loc[vol_df['month'].isin([7,8,9]), 'season'] = 'summer'

vol_df.loc[vol_df['month'].isin([10,11,12]), 'season'] = 'fall'


wtd_avg_df.loc[wtd_avg_df['month'].isin([1,2,3]), 'season'] = 'winter'

wtd_avg_df.loc[wtd_avg_df['month'].isin([4,5,6]), 'season'] = 'spring'

wtd_avg_df.loc[wtd_avg_df['month'].isin([7,8,9]), 'season'] = 'summer'

wtd_avg_df.loc[wtd_avg_df['month'].isin([10,11,12]), 'season'] = 'fall'


avg_df.loc[avg_df['month'].isin([1,2,3]), 'season'] = 'winter'

avg_df.loc[avg_df['month'].isin([4,5,6]), 'season'] = 'spring'

avg_df.loc[avg_df['month'].isin([7,8,9]), 'season'] = 'summer'

avg_df.loc[avg_df['month'].isin([10,11,12]), 'season'] = 'fall'


# %%

wtd_avg_df_filt = wtd_avg_df[wtd_avg_df['DO_wtd_avg_below_mg_L'] <20]

avg_df_filt = avg_df[avg_df['DO_avg_below_mg_L'] <20]

vol_df_filt = vol_df[vol_df['data_type'] == 'OBS']

# %%

vol_df_temp_avg= vol_df_filt[['segment','year','season','vol_km3','date_ordinal']].groupby(by=['segment','year','season']).mean().reset_index()

wtd_avg_df_temp_avg= wtd_avg_df_filt[['segment','year','season','DO_wtd_avg_below_mg_L','date_ordinal']].groupby(by=['segment','year','season']).mean().reset_index()

avg_df_temp_avg= avg_df_filt[['segment','year','season','DO_avg_below_mg_L','date_ordinal']].groupby(by=['segment','year','season']).mean().reset_index()


vol_df_temp_sum= vol_df_filt[['segment','year','season','num_casts']].groupby(by=['segment','year','season']).sum().reset_index()

wtd_avg_df_temp_sum= wtd_avg_df_filt[['segment','year','season','num_casts']].groupby(by=['segment','year','season']).sum().reset_index()

avg_df_temp_sum= avg_df_filt[['segment','year','season','num_casts']].groupby(by=['segment','year','season']).sum().reset_index()


vol_df_seasonal = pd.merge(vol_df_temp_avg, vol_df_temp_sum, how='left', on=['segment','year','season'])

wtd_avg_df_seasonal = pd.merge(wtd_avg_df_temp_avg, wtd_avg_df_temp_sum, how='left', on=['segment','year','season'])

avg_df_seasonal = pd.merge(avg_df_temp_avg, avg_df_temp_sum, how='left', on=['segment','year','season'])


# %%

if seg_str[0] == 'sound_straits':

    fig, ax = plt.subplots(1,1,figsize=(18,18))
    
    
    ax0 = sns.relplot(data = avg_df_filt, x ='date_ordinal', y = 'DO_avg_below_mg_L', row = 'segment', col = 'season', col_order=['winter','spring','summer','fall'], hue = 'segment', palette = 'Set2', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], size='num_casts',sizes=(50,500), height = 6, style = 'season', style_order=['winter','spring','summer','fall'])
    
   
    labels = [date(1930,1,1), date(1940,1,1),  
                  date(1950,1,1),
                  date(1960,1,1),
                  date(1970,1,1),  
                  date(1980,1,1),   
                  date(1990,1,1),  
                  date(2000,1,1),  
                  date(2010,1,1),
                  date(2020,1,1), date(2030,1,1)] #,date(2022,1,1)]
    
    new_labels = [date.toordinal(item) for item in labels]
    
    
    for ax00 in ax0.axes.flat:
        
        ax00.set_xticks(new_labels)
    
  
    
    
        ax00.set_xticklabels(['','1940', '1950','1960',
                        '1970','1980','1990','2000','2010','2020',''], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
        ax00.set_facecolor("white")
    

    
        ax00.set(xlabel = 'Date')

    
        ax00.set(ylabel = 'Sub-40m Avg DO [$mg/L$]')

    
        ax00.grid(alpha=0.3)
    
     
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_40m_'+ seg_str[0]+ '_DO_avg_seasons_relplot.png', transparent=False, dpi=500)




    fig, ax = plt.subplots(1,1,figsize=(18,18))
    
    
    ax0 = sns.relplot(data = wtd_avg_df_filt, x ='date_ordinal', y = 'DO_wtd_avg_below_mg_L', row = 'segment', col = 'season', col_order=['winter','spring','summer','fall'], hue = 'segment', palette = 'Set2', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], size='num_casts',sizes=(50,500), height = 6, style = 'season', style_order=['winter','spring','summer','fall'])
    
    
    labels = [date(1930,1,1), date(1940,1,1),  
                  date(1950,1,1),
                  date(1960,1,1),
                  date(1970,1,1),  
                  date(1980,1,1),   
                  date(1990,1,1),  
                  date(2000,1,1),  
                  date(2010,1,1),
                  date(2020,1,1), date(2030,1,1)] #,date(2022,1,1)]
    
    new_labels = [date.toordinal(item) for item in labels]
    
    
    for ax00 in ax0.axes.flat:
        
        ax00.set_xticks(new_labels)
    
    
    
    
        ax00.set_xticklabels(['','1940', '1950','1960',
                        '1970','1980','1990','2000','2010','2020',''], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
        ax00.set_facecolor("white")
    
    
    
        ax00.set(xlabel = 'Date')
    
    
        ax00.set(ylabel = 'Sub-40m Wtd Avg DO[$mg/L$]')
    
    
        ax00.grid(alpha=0.3)
    
     
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_40m_'+ seg_str[0]+ '_DO_wtd_avg_seasons_relplot.png', transparent=False, dpi=500)
    

    
    fig, ax = plt.subplots(1,1,figsize=(18,18))
    
    
    ax0 = sns.relplot(data = vol_df_filt, x ='date_ordinal', y = 'vol_km3', row = 'segment', col = 'season', col_order=['winter','spring','summer','fall'], hue = 'segment', palette = 'Set2', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], size='num_casts',sizes=(50,500), height = 6, style = 'season', style_order=['winter','spring','summer','fall'])
    
    
    labels = [date(1930,1,1), date(1940,1,1),  
                  date(1950,1,1),
                  date(1960,1,1),
                  date(1970,1,1),  
                  date(1980,1,1),   
                  date(1990,1,1),  
                  date(2000,1,1),  
                  date(2010,1,1),
                  date(2020,1,1), date(2030,1,1)] #,date(2022,1,1)]
    
    new_labels = [date.toordinal(item) for item in labels]
    
    
    for ax00 in ax0.axes.flat:
        
        ax00.set_xticks(new_labels)
    
    
    
    
        ax00.set_xticklabels(['','1940', '1950','1960',
                        '1970','1980','1990','2000','2010','2020',''], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
        ax00.set_facecolor("white")
    
    
    
        ax00.set(xlabel = 'Date')
    
    
        ax00.set(ylabel = 'Hypoxic Volume [$km^3$]')
    
    
        ax00.grid(alpha=0.3)
    
     
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_'+ seg_str[0]+ '_hyp_vol_seasons_relplot.png', transparent=False, dpi=500)
    
# %%

if seg_str[0] == 'basins':

    fig, ax = plt.subplots(1,1,figsize=(18,18))
    
    
    ax0 = sns.relplot(data = avg_df_filt, x ='date_ordinal', y = 'DO_avg_below_mg_L', row = 'segment', col = 'season', col_order=['winter','spring','summer','fall'], hue = 'segment', palette = 'Set2_r',  hue_order = ['Admiralty Inlet','Whidbey Basin','South Sound','Hood Canal','Main Basin'], size='num_casts',sizes=(50,500), height = 7, style = 'season', style_order=['winter','spring','summer','fall'])
    
   
    
    labels = [date(1995,1,1), 
                  date(2000,1,1),  date(2005,1,1), 
                  date(2010,1,1), date(2015,1,1), 
                  date(2020,1,1), date(2025,1,1)] 
    
    new_labels = [date.toordinal(item) for item in labels]
    
    
    for ax00 in ax0.axes.flat:
        
        ax00.set_xticks(new_labels)
    
  
    
    
        ax00.set_xticklabels(['','2000','2005','2010','2015','2020',''], rotation=0,
                fontdict={'horizontalalignment':'center'})
    
    
        ax00.set_facecolor("white")
    

    
        ax00.set(xlabel = 'Date')

    
        ax00.set(ylabel = 'Sub-40m Avg DO [$mg/L$]')

    
        ax00.grid(alpha=0.3)
    
     
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_40m_'+ seg_str[0]+ '_DO_avg_seasons_relplot.png', transparent=False, dpi=500)




    fig, ax = plt.subplots(1,1,figsize=(18,18))
    
    
    ax0 = sns.relplot(data = wtd_avg_df_filt, x ='date_ordinal', y = 'DO_wtd_avg_below_mg_L', row = 'segment', col = 'season', col_order=['winter','spring','summer','fall'], hue = 'segment', palette = 'Set2_r',  hue_order = ['Admiralty Inlet','Whidbey Basin','South Sound','Hood Canal','Main Basin'], size='num_casts',sizes=(50,500), height = 6, style = 'season', style_order=['winter','spring','summer','fall'])
    
    
    labels = [date(1995,1,1), 
                  date(2000,1,1),  date(2005,1,1), 
                  date(2010,1,1), date(2015,1,1), 
                  date(2020,1,1), date(2025,1,1)] 
    
    new_labels = [date.toordinal(item) for item in labels]
    
    
    for ax00 in ax0.axes.flat:
        
        ax00.set_xticks(new_labels)
    
  
    
    
        ax00.set_xticklabels(['','2000','2005','2010','2015','2020',''], rotation=0,
                fontdict={'horizontalalignment':'center'})
    
    
        ax00.set_facecolor("white")
    
    
    
        ax00.set(xlabel = 'Date')
    
    
        ax00.set(ylabel = 'Sub-40m Wtd Avg DO[$mg/L$]')
    
    
        ax00.grid(alpha=0.3)
    
     
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_40m_'+ seg_str[0]+ '_DO_wtd_avg_seasons_relplot.png', transparent=False, dpi=500)
    

    
    fig, ax = plt.subplots(1,1,figsize=(18,18))
    
    
    ax0 = sns.relplot(data = vol_df_filt[vol_df_filt['segment'] != 'Tacoma Narrows'], x ='date_ordinal', y = 'vol_km3', row = 'segment', col = 'season', col_order=['winter','spring','summer','fall'], hue = 'segment', palette = 'Set2_r',  hue_order = ['Admiralty Inlet','Whidbey Basin','South Sound','Hood Canal','Main Basin'], size='num_casts',sizes=(50,500), height = 6, style = 'season', style_order=['winter','spring','summer','fall'])
    
    
    labels = [date(1995,1,1), 
                  date(2000,1,1),  date(2005,1,1), 
                  date(2010,1,1), date(2015,1,1), 
                  date(2020,1,1), date(2025,1,1)] 
    
    new_labels = [date.toordinal(item) for item in labels]
    
    
    for ax00 in ax0.axes.flat:
        
        ax00.set_xticks(new_labels)
    
  
    
    
        ax00.set_xticklabels(['','2000','2005','2010','2015','2020',''], rotation=0,
                fontdict={'horizontalalignment':'center'})
    
    
        ax00.set_facecolor("white")
    
    
    
        ax00.set(xlabel = 'Date')
    
    
        ax00.set(ylabel = 'Hypoxic Volume [$km^3$]')
    
    
        ax00.grid(alpha=0.3)
    
     
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_'+ seg_str[0]+ '_hyp_vol_seasons_relplot.png', transparent=False, dpi=500)
    