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

seg_str = ['sound_straits']

#years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2020]


years = np.arange(1930,2023)

# %%

#years = [2017, 2018, 2019]

vol_df = pd.DataFrame()

wtd_avg_df = pd.DataFrame()

avg_df = pd.DataFrame()

for year in years:
    
    for segs in seg_str:
    
        fn_cid = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/cid_dict_NEW.pkl'
        
        fn_vol_df = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/vol_df_NEW.p'
        
        fn_wtd_avg = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/wtd_avg_df_NEW_wTS.p'
        
        fn_avg = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/avg_df_NEW_wTS.p'
        
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

# vol_df_wide = pd.DataFrame()

# years0 = [2017]

# for year in years0:
        
#     for segs in seg_str:
        
#         fn_cid = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/cid_dict.pkl'
        
#         fn_vol_df_wide = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/vol_df_wide.p'
        
#         if os.path.isfile(fn_cid):
        
#             vol_df_wide_temp = pd.read_pickle(fn_vol_df_wide)
            
#             vol_df_wide_temp['year'] = year
            
#             vol_df_wide_temp['day'] = 1
            
#             with open(fn_cid, 'rb') as f: 
#                 cid_dict = pickle.load(f)
            
#             vol_df_wide_temp['num_casts'] = np.nan
            
#             cid_df = pd.DataFrame.from_dict(cid_dict)
            
#             for col in cid_df.columns:
                
#                 for m in cid_df.index:
                
#                     vol_df_wide_temp.loc[(vol_df_wide_temp['month'] == m) & (vol_df_wide_temp['segment'] == col), 'num_casts'] = len(cid_df.loc[m,col])
            
#             vol_df_wide_temp.loc[vol_df_wide_temp['num_casts'] == 0, 'num_casts'] = np.nan
            
#             vol_df_wide = pd.concat([vol_df_wide, vol_df_wide_temp], ignore_index=True)


# vol_df_wide['date'] = pd.to_datetime(dict(year=vol_df_wide.year, month=vol_df_wide.month, day=vol_df_wide.day))

# vol_df_wide['date_ordinal'] = vol_df_wide['date'].apply(lambda date: date.toordinal())


# vol_df_wide['E_LO_His_LO_casts'] = np.sqrt(vol_df_wide['SE_LO_his_LO_casts'])

# %%

# vdfw_temp = vol_df_wide[['segment', 'month', 'E_LO_His_LO_casts']]

# vol_df = pd.merge(vol_df, vdfw_temp, how='left', on=['segment', 'month'])

# %%

# vol_df_wide['upper'] = vol_df_wide['OBS'] + vol_df_wide['E_LO_His_LO_casts']

# vol_df_wide['lower'] = vol_df_wide['OBS'] - vol_df_wide['E_LO_His_LO_casts']

# vol_df_wide.loc[vol_df_wide['lower'] < 0, 'lower'] = 0

# %%

wtd_avg_df_filt = wtd_avg_df[wtd_avg_df['DO_wtd_avg_below_mg_L'] <20]

avg_df_filt = avg_df[avg_df['DO_avg_below_mg_L'] <20]



# %%

if seg_str[0] == 'sound_straits':

    fig, ax = plt.subplots(3,1,figsize=(18,27))
    
    ax0 = sns.scatterplot(data = vol_df[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound')], x = 'date_ordinal', y = 'vol_km3', ax =ax[0], size='num_casts',sizes=(50,500))# , style = 'data_type') size='num_casts',sizes=(30,300),

    ax1 = sns.scatterplot(data = avg_df_filt[(avg_df_filt['data_type'] == 'OBS') & (avg_df_filt['segment'] == 'Puget Sound')], x = 'date_ordinal', y = 'DO_avg_below_mg_L', ax =ax[0], size='num_casts',sizes=(50,500))
    
    labels = [date(1995,1,1), 
                  date(2000,1,1),  date(2005,1,1), 
                  date(2010,1,1), date(2015,1,1), 
                  date(2020,1,1), date(2025,1,1)] 
                  #date(2020,1,1),  date(2021,1,1)] #,date(2022,1,1)]
    
    new_labels = [date.toordinal(item) for item in labels]
    
    ax0.set_xticks(new_labels)
    
    ax1.set_xticks(new_labels)
        
    
    
    ax0.set_xticklabels(['','2000','2005','2010','2015','2020',''], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax0.set_facecolor("white")
    

    
    ax0.set(xlabel = 'Date')
    
    ax1.set_xticklabels(['','2000','2005','2010','2015','2020',''], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax1.set_facecolor("white")
    

    
    ax1.set(xlabel = 'Date')
  
    
    ax[0].set(ylabel = 'Hypoxic Volume [$km^3$]')
    
    ax[1].set(ylabel = 'Sub-40m Avg DO [$mg/L$]')
    
    
    ax[0].legend(title=False, loc='upper left')#, ncol =3)
    
    ax[1].legend(title=False, loc='upper left')#, ncol =3)
    
    #plt.legend(loc='upper left')
    
    
    ax[0].grid(alpha=0.3)
    
    ax[1].grid(alpha=0.3)
    
    ax[2].grid(alpha=0.3)
    
    
    #plt.legend() #title = 'Basin [Order of Increasing Volume]')

    fig.tight_layout()

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_sub_avg'+ seg_str[0]+ '_PS.png', transparent=False, dpi=500)

# %%



if seg_str[0] == 'sound_straits':

    fig, ax = plt.subplots(3,1,figsize=(18,27))
    
    
    ax0 = sns.scatterplot(data = vol_df[vol_df['data_type'] == 'OBS'], x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'Set2', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], ax =ax[0], size='num_casts',sizes=(50,500))# , style = 'data_type') size='num_casts',sizes=(30,300),
    
    
    ax0.set(xlim=(vol_df['date_ordinal'].min()-1, vol_df['date_ordinal'].max()+1),ylim=(0,150))
    
    ax1 = sns.scatterplot(data = wtd_avg_df_filt[wtd_avg_df_filt['data_type'] == 'OBS'], x = 'date_ordinal', y = 'DO_wtd_avg_below_mg_L', hue = 'segment', palette = 'Set2', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], ax=ax[1], size='num_casts',sizes=(50,500))# , style = 'data_type')
    
    
    ax2 = sns.scatterplot(data = avg_df_filt[avg_df_filt['data_type'] == 'OBS'], x = 'date_ordinal', y = 'DO_avg_below_mg_L', hue = 'segment', palette = 'Set2', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], ax = ax[2], size='num_casts',sizes=(50,500))# , style = 'data_type')
    
    labels = [date(1910,1,1), date(1915,1,1), date(1920,1,1), date(1925,1,1,), date(1930,1,1), date(1935,1,1), date(1940,1,1), date(1945,1,1),  
                  date(1950,1,1), date(1955,1,1), 
                  date(1960,1,1), date(1965,1,1), 
                  date(1970,1,1),  date(1975,1,1), 
                  date(1980,1,1),  date(1985,1,1), 
                  date(1990,1,1),  date(1995,1,1), 
                  date(2000,1,1),  date(2005,1,1), 
                  date(2010,1,1), date(2015,1,1), 
                  date(2020,1,1), date(2025,1,1)] #,date(2022,1,1)]
    
    new_labels = [date.toordinal(item) for item in labels]
    
    ax0.set_xticks(new_labels)
    
    ax1.set_xticks(new_labels)
    
    ax2.set_xticks(new_labels)
    
    
    
    ax0.set_xticklabels(['','','','','1930','1935','1940', '1945','1950','1955','1960','1965',
                        '1970','1975','1980','1985','1990','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax0.set_facecolor("white")
    

    
    ax0.set(xlabel = 'Date')
    
    ax1.set_xticklabels(['','','','','1930','1935','1940', '1945','1950','1955','1960','1965',
                        '1970','1975','1980','1985','1990','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax1.set_facecolor("white")
    

    
    ax1.set(xlabel = 'Date')
    
    
    ax2.set_xticklabels(['','','','', '1930','1935','1940', '1945','1950','1955','1960','1965',
                        '1970','1975','1980','1985','1990','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax2.set_facecolor("white")
    

    
    ax2.set(xlabel = 'Date')
    
    
    ax[0].set(ylabel = 'Hypoxic Volume [$km^3$]')
    
    ax[1].set(ylabel = 'Sub-40m Wtd Avg DO[$mg/L$]')
    
    ax[2].set(ylabel = 'Sub-40m Avg DO [$mg/L$]')
    
    
    ax[0].legend(title=False, loc='upper left')#, ncol =3) plt.legend(bbox_to_anchor=(1.02, 0.55), loc='upper left', borderaxespad=0)

    ax[1].legend(title=False, loc='upper left')#, ncol =3)
    
    ax[2].legend(title=False, loc='upper left')#, ncol =3)
    
    
    
    
    ax[0].grid(alpha=0.3)
    
    ax[1].grid(alpha=0.3)
    
    ax[2].grid(alpha=0.3)
    
    plt.legend() #title = 'Basin [Order of Increasing Volume]')
     
    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_'+ seg_str[0]+ '_vol_wtd_avg_avg_w_casts.png', transparent=False, dpi=500)

# %%

if seg_str[0] == 'basins':

    fig, ax = plt.subplots(3,1,figsize=(18,27))
        
    
    ax0 = sns.scatterplot(data = vol_df[vol_df['data_type'] == 'OBS'], x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'Set2_r',  hue_order = ['Admiralty Inlet','Whidbey Basin','South Sound','Hood Canal','Main Basin'], ax =ax[0], size='num_casts',sizes=(50,500))# , style = 'data_type')size='num_casts',sizes=(30,300),
    
    
    #ax0.set(xlim=(vol_df['date_ordinal'].min()-1, vol_df['date_ordinal'].max()+1),ylim=(0,150))
    
    ax1 = sns.scatterplot(data = wtd_avg_df_filt[wtd_avg_df_filt['data_type'] == 'OBS'], x = 'date_ordinal', y = 'DO_wtd_avg_below_mg_L', hue = 'segment', palette = 'Set2_r', hue_order = ['Admiralty Inlet','Whidbey Basin','South Sound','Hood Canal','Main Basin'], ax=ax[1], size='num_casts',sizes=(50,500))# , style = 'data_type')
    
    
    ax2 = sns.scatterplot(data = avg_df_filt[avg_df_filt['data_type'] == 'OBS'], x = 'date_ordinal', y = 'DO_avg_below_mg_L', hue = 'segment', palette = 'Set2_r', hue_order = ['Admiralty Inlet','Whidbey Basin','South Sound','Hood Canal','Main Basin'], ax = ax[2], size='num_casts',sizes=(50,500))# , style = 'data_type')
    
    
    labels = [date(1990,1,1,), date(1995,1,1), 
                  date(2000,1,1),  date(2005,1,1), 
                  date(2010,1,1), date(2015,1,1), 
                  date(2020,1,1), date(2025,1,1)] 
                  #date(2020,1,1),  date(2021,1,1)] #,date(2022,1,1)]
    
    new_labels = [date.toordinal(item) for item in labels]
    
    ax0.set_xticks(new_labels)
    
    ax1.set_xticks(new_labels)
    
    ax2.set_xticks(new_labels)
    
    
    
    ax0.set_xticklabels(['','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax0.set_facecolor("white")
    

    
    ax0.set(xlabel = 'Date')
    
    ax1.set_xticklabels(['','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax1.set_facecolor("white")
    

    
    ax1.set(xlabel = 'Date')
    
    
    ax2.set_xticklabels(['','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax2.set_facecolor("white")
    

    
    ax2.set(xlabel = 'Date')
    
    ax[0].set(ylabel = 'Hypoxic Volume [$km^3$]')
    
    ax[1].set(ylabel = 'Sub-40m Wtd Avg DO [$mg/L$]')
    
    ax[2].set(ylabel = 'Sub-40m Avg DO [$mg/L$]')
    
    
    ax[0].legend(title=False, loc='upper left')#, ncol =3)
    
    ax[1].legend(title=False, loc='upper left')#, ncol =3)
    
    ax[2].legend(title=False, loc='upper left')#, ncol =3)
    
    plt.legend(loc='upper left')
    
    
    ax[0].grid(alpha=0.3)
    
    ax[1].grid(alpha=0.3)
    
    ax[2].grid(alpha=0.3)
    
    
    plt.legend() #title = 'Basin [Order of Increasing Volume]')

    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_'+ seg_str[0]+ '_vol_wtd_avg_avg_w_casts.png', transparent=False, dpi=500)

# %%

#ADHOC JUST WHIDBEY & HOOD CANAL


spots = ['Whidbey Basin', 'Hood Canal']


if seg_str[0] == 'basins':

    fig, ax = plt.subplots(3,1,figsize=(18,27))
        
    
    ax0 = sns.scatterplot(data = vol_df[(vol_df['data_type'] == 'OBS') & (vol_df['segment'].isin(spots))], x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'Set2_r', hue_order =['Whidbey Basin', 'Hood Canal'], ax =ax[0], size='num_casts',sizes=(50,500))# , style = 'data_type')size='num_casts',sizes=(30,300),
    
    
    #ax0.set(xlim=(vol_df['date_ordinal'].min()-1, vol_df['date_ordinal'].max()+1),ylim=(0,150))
    
    ax1 = sns.scatterplot(data = wtd_avg_df_filt[(wtd_avg_df_filt['data_type'] == 'OBS') & (wtd_avg_df_filt['segment'].isin(spots))], x = 'date_ordinal', y = 'DO_wtd_avg_below_mg_L', hue = 'segment', hue_order =['Whidbey Basin', 'Hood Canal'],palette = 'Set2_r',  ax=ax[1], size='num_casts',sizes=(50,500))# , style = 'data_type')
    
    
    ax2 = sns.scatterplot(data = avg_df_filt[(avg_df_filt['data_type'] == 'OBS') & (avg_df_filt['segment'].isin(spots))], x = 'date_ordinal', y = 'DO_avg_below_mg_L', hue = 'segment', palette = 'Set2_r', hue_order =['Whidbey Basin', 'Hood Canal'],ax = ax[2], size='num_casts',sizes=(50,500))# , style = 'data_type')
    
    
    labels = [date(1990,1,1), date(1995,1,1), 
                  date(2000,1,1),  date(2005,1,1), 
                  date(2010,1,1), date(2015,1,1), 
                  date(2020,1,1), date(2025,1,1)] 
                  #date(2020,1,1),  date(2021,1,1)] #,date(2022,1,1)]
    
    new_labels = [date.toordinal(item) for item in labels]
    
    ax0.set_xticks(new_labels)
    
    ax1.set_xticks(new_labels)
    
    ax2.set_xticks(new_labels)
    
    
    
    ax0.set_xticklabels(['','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax0.set_facecolor("white")
    

    
    ax0.set(xlabel = 'Date')
    
    ax1.set_xticklabels(['','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax1.set_facecolor("white")
    

    
    ax1.set(xlabel = 'Date')
    
    
    ax2.set_xticklabels(['','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
        fontdict={'horizontalalignment':'center'})
    
    
    ax2.set_facecolor("white")
    

    
    ax2.set(xlabel = 'Date')
    
    ax[0].set(ylabel = 'Hypoxic Volume [$km^3$]')
    
    ax[1].set(ylabel = 'Sub-40m Wtd Avg DO [$mg/L$]')
    
    ax[2].set(ylabel = 'Sub-40m Avg DO [$mg/L$]')
    
    
    ax[0].legend(title=False, loc='upper left')#, ncol =3)
    
    ax[1].legend(title=False, loc='upper left')#, ncol =3)
    
    ax[2].legend(title=False, loc='upper left')#, ncol =3)
    
    #sns.move_legend(ax2, "upper left")
    
    
    
    ax[0].grid(alpha=0.3)
    
    ax[1].grid(alpha=0.3)
    
    ax[2].grid(alpha=0.3)
    
    
    #plt.legend() #title = 'Basin [Order of Increasing Volume]')

    fig.tight_layout()
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_'+ seg_str[0]+ '_vol_wtd_avg_avg_WBHConly_w_casts.png', transparent=False, dpi=500)