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

seg_str = ['basins']

#years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2020]


years = np.arange(1930,2023)

# %%

#years = [2017, 2018, 2019]

vol_df = pd.DataFrame()

wtd_avg_df = pd.DataFrame()

for year in years:
    
    for segs in seg_str:
    
        fn_cid = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/cid_dict.pkl'
        
        fn_vol_df = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/vol_df.p'
        
        fn_wtd_avg = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/wtd_avg_dict.pkl'
        
        if os.path.isfile(fn_cid):
            
            vol_df_temp = pd.read_pickle(fn_vol_df)
        
            vol_df_temp['year'] = year
        
            vol_df_temp['day'] = 1
            
            wtd_avg_df_temp = pd.read_pickle(fn_wtd_avg)
        
            wtd_avg_df_temp['year'] = year
        
            wtd_avg_df_temp['day'] = 1
                        
            with open(fn_cid, 'rb') as f: 
                cid_dict = pickle.load(f)
            
            vol_df_temp['num_casts'] = np.nan
            
            wtd_avg_df_temp['num_casts'] = np.nan
            
            cid_df = pd.DataFrame.from_dict(cid_dict)
            
            for col in cid_df.columns:
                
                for m in cid_df.index:
                
                    vol_df_temp.loc[(vol_df_temp['month'] == m) & (vol_df_temp['segment'] == col), 'num_casts'] = len(cid_df.loc[m,col])
                    
                    wtd_avg_df_temp.loc[(wtd_avg_df_temp['month'] == m) & (wtd_avg_df_temp['segment'] == col), 'num_casts'] = len(cid_df.loc[m,col])
                    
            vol_df_temp.loc[vol_df_temp['num_casts'] == 0, 'num_casts'] = np.nan

            vol_df = pd.concat([vol_df, vol_df_temp], ignore_index=True)
            
            wtd_avg_df_temp.loc[wtd_avg_df_temp['num_casts'] == 0, 'num_casts'] = np.nan

            wtd_avg_df = pd.concat([wtd_avg_df, wtd_avg_df_temp], ignore_index=True)
        
        
        
# %%

vol_df['date'] = pd.to_datetime(dict(year=vol_df.year, month=vol_df.month, day=vol_df.day))

vol_df['date_ordinal'] = vol_df['date'].apply(lambda date: date.toordinal())

wtd_avg_df['date'] = pd.to_datetime(dict(year=wtd_avg_df.year, month=wtd_avg_df.month, day=wtd_avg_df.day))

wtd_avg_df['date_ordinal'] = wtd_avg_df['date'].apply(lambda date: date.toordinal())


# %%

vol_df_wide = pd.DataFrame()

years0 = [2017]

for year in years0:
        
    for segs in seg_str:
        
        fn_cid = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/cid_dict.pkl'
        
        fn_vol_df_wide = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/vol_df_wide.p'
        
        if os.path.isfile(fn_cid):
        
            vol_df_wide_temp = pd.read_pickle(fn_vol_df_wide)
            
            vol_df_wide_temp['year'] = year
            
            vol_df_wide_temp['day'] = 1
            
            with open(fn_cid, 'rb') as f: 
                cid_dict = pickle.load(f)
            
            vol_df_wide_temp['num_casts'] = np.nan
            
            cid_df = pd.DataFrame.from_dict(cid_dict)
            
            for col in cid_df.columns:
                
                for m in cid_df.index:
                
                    vol_df_wide_temp.loc[(vol_df_wide_temp['month'] == m) & (vol_df_wide_temp['segment'] == col), 'num_casts'] = len(cid_df.loc[m,col])
            
            vol_df_wide_temp.loc[vol_df_wide_temp['num_casts'] == 0, 'num_casts'] = np.nan
            
            vol_df_wide = pd.concat([vol_df_wide, vol_df_wide_temp], ignore_index=True)
        
# %%

vol_df_wide['date'] = pd.to_datetime(dict(year=vol_df_wide.year, month=vol_df_wide.month, day=vol_df_wide.day))

vol_df_wide['date_ordinal'] = vol_df_wide['date'].apply(lambda date: date.toordinal())

# %%

vol_df_wide['E_LO_His_LO_casts'] = np.sqrt(vol_df_wide['SE_LO_his_LO_casts'])

# %%

vdfw_temp = vol_df_wide[['segment', 'month', 'E_LO_His_LO_casts']]

vol_df = pd.merge(vol_df, vdfw_temp, how='left', on=['segment', 'month'])

# %%

vol_df_wide['upper'] = vol_df_wide['OBS'] + vol_df_wide['E_LO_His_LO_casts']

vol_df_wide['lower'] = vol_df_wide['OBS'] - vol_df_wide['E_LO_His_LO_casts']

vol_df_wide.loc[vol_df_wide['lower'] < 0, 'lower'] = 0




# %%

# fig, ax = plt.subplots(1,1,figsize=(18,6))




# #plot = sns.pointplot(ax=ax, data = vol_df_plot[vol_df_plot['data_type'] == 'OBS'], x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'rocket', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound']) #, style = 'data_type')


# #ax.set_xticklabels([])

# plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Juan de Fuca'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Juan de Fuca'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Juan de Fuca'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle='None')


# plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Georgia'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Georgia'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Georgia'),'E_LO_His_LO_casts']),capsize =3,  c='gray', alpha=0.5, linestyle ='None')


# plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

# # plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Admiralty Inlet'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

# # plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Main Basin'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')


# ax = sns.scatterplot(data = vol_df[vol_df['data_type'] == 'OBS'], x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'rocket_r', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], size='num_casts',sizes=(30,300), alpha =0.7)# , style = 'data_type')


# ax.set(xlim=(vol_df['date_ordinal'].min()-1, vol_df['date_ordinal'].max()+1),ylim=(0,150))




# labels = [date(1930,1,1), date(1935,1,1), date(1940,1,1), date(1945,1,1),  
#               date(1950,1,1), date(1955,1,1), 
#               date(1960,1,1), date(1965,1,1), 
#               date(1970,1,1),  date(1975,1,1), 
#               date(1980,1,1),  date(1985,1,1), 
#               date(1990,1,1),  date(1995,1,1), 
#               date(2000,1,1),  date(2005,1,1), 
#               date(2010,1,1), date(2015,1,1), 
#               date(2020,1,1), date(2025,1,1)] # , date(2019,1,1), 
#               #date(2020,1,1),  date(2021,1,1)] #,date(2022,1,1)]

# new_labels = [date.toordinal(item) for item in labels]

# ax.set_xticks(new_labels)


# ax.set_xticklabels(['1930','1935','1940', '1945','1950','1955','1960','1965',
#                     '1970','1975','1980','1985','1990','1995','2000','2005','2010','2015','2020','2025'], rotation=0,
#     fontdict={'horizontalalignment':'center'})

# ax.set_facecolor("white")

# plt.legend() #title = 'Basin [Order of Increasing Volume]')


# ax.set(xlabel = 'Date', ylabel = 'Hypoxic Volume [$km^3$]')

# plt.legend(title=False, loc='upper left')#, ncol =3)




# plt.grid(alpha=0.3)

# fig.tight_layout()

# plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_'+ seg_str[0]+ '.png', transparent=False, dpi=500)


# %%

fig, ax = plt.subplots(1,1,figsize=(18,8))



plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Admiralty Inlet'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Admiralty Inlet'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Admiralty Inlet'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Main Basin'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Main Basin'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Main Basin'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Hood Canal'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Hood Canal'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Hood Canal'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Whidbey Basin'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Whidbey Basin'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Whidbey Basin'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Tacoma Narrows'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Tacoma Narrows'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Tacoma Narrows'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'South Sound'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'South Sound'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'South Sound'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')



ax = sns.scatterplot(data = vol_df[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] != 'Strait of Georgia') & (vol_df['segment'] != 'Strait of Juan de Fuca')], x = 'date_ordinal', y = 'vol_km3', hue = 'segment', size='num_casts', sizes=(30,300))# , style = 'data_type')


ax2 = plt.twinx()

sns.scatterplot(data = wtd_avg_df[(wtd_avg_df['data_type'] == 'OBS') & (wtd_avg_df['segment'] != 'Strait of Georgia') & (wtd_avg_df['segment'] != 'Strait of Juan de Fuca')], x = 'date_ordinal', y = 'DO_wtd_avg_mg_L', hue = 'segment', size='num_casts', sizes=(30,300), ax = ax2)# , style = 'data_type')


ax.set(xlim=(date.toordinal(date(1999,1,1)), date.toordinal(date(2019,12,31))),ylim=(0,4))






labels = [date(1999,1,1),  date(2000,1,1), 
              date(2001,1,1),  date(2002,1,1), 
              date(2003,1,1),  date(2004,1,1), 
              date(2005,1,1),  date(2006,1,1), 
              date(2007,1,1), date(2008,1,1), 
              date(2009,1,1), 
              date(2010,1,1),  date(2011,1,1), 
              date(2012,1,1),  date(2013,1,1), 
              date(2014,1,1),  date(2015,1,1), 
              date(2016,1,1), date(2017,1,1), 
              date(2018,1,1), date(2019,1,1)]


new_labels = [date.toordinal(item) for item in labels]

ax.set_xticks(new_labels)


ax.set_xticklabels(['1999','2000','2001','2002','2003','2004','2005','2006','2007',
                    '2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'], rotation=0,
    fontdict={'horizontalalignment':'center'})

plt.legend() #title = 'Basin [Order of Increasing Volume]')


ax.set(xlabel = 'Date', ylabel = 'Hypoxic Volume [$km^3$]')

ax2.set(ylabel = 'DO Wtd Avg Sub-40m [mg/L]')

plt.legend(title=False, loc='upper left') #, ncol =2)




plt.grid(alpha=0.3)

fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_wtd_avg_below_40m_'+ seg_str[0]+ '_PS.png', transparent=False, dpi=500)


# %%

fig, ax = plt.subplots(1,1,figsize=(18,8))



# plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Admiralty Inlet'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Admiralty Inlet'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Admiralty Inlet'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

# plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Main Basin'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Main Basin'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Main Basin'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Hood Canal'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Hood Canal'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Hood Canal'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

# plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Whidbey Basin'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Whidbey Basin'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Whidbey Basin'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

# plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Tacoma Narrows'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Tacoma Narrows'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Tacoma Narrows'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

# plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'South Sound'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'South Sound'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'South Sound'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')



ax = sns.scatterplot(data = vol_df[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Hood Canal')], x = 'date_ordinal', y = 'vol_km3', hue = 'segment', size='num_casts', sizes=(30,300))# , style = 'data_type')


ax2 = plt.twinx()

sns.scatterplot(data = wtd_avg_df[(wtd_avg_df['data_type'] == 'OBS') & (wtd_avg_df['segment'] == 'Hood Canal')], x = 'date_ordinal', y = 'DO_wtd_avg_mg_L', hue = 'segment', size='num_casts', sizes=(30,300), ax = ax2)# , style = 'data_type')


ax.set(xlim=(date.toordinal(date(1999,1,1)), date.toordinal(date(2019,12,31))),ylim=(0,4))






labels = [date(1999,1,1),  date(2000,1,1), 
              date(2001,1,1),  date(2002,1,1), 
              date(2003,1,1),  date(2004,1,1), 
              date(2005,1,1),  date(2006,1,1), 
              date(2007,1,1), date(2008,1,1), 
              date(2009,1,1), 
              date(2010,1,1),  date(2011,1,1), 
              date(2012,1,1),  date(2013,1,1), 
              date(2014,1,1),  date(2015,1,1), 
              date(2016,1,1), date(2017,1,1), 
              date(2018,1,1), date(2019,1,1)]


new_labels = [date.toordinal(item) for item in labels]

ax.set_xticks(new_labels)


ax.set_xticklabels(['1999','2000','2001','2002','2003','2004','2005','2006','2007',
                    '2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'], rotation=0,
    fontdict={'horizontalalignment':'center'})

plt.legend() #title = 'Basin [Order of Increasing Volume]')


ax.set(xlabel = 'Date', ylabel = 'Hypoxic Volume [$km^3$]')

ax2.set(ylabel = 'DO Wtd Avg Sub-40m [mg/L]')

plt.legend(title=False, loc='upper left') #, ncol =2)




plt.grid(alpha=0.3)

fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_wtd_avg_below_40m_'+ seg_str[0]+ '_HC.png', transparent=False, dpi=500)




# %%

# vol_df_plot = vol_df[(vol_df['segment'] == 'Puget Sound') & (vol_df['data_type'] == 'OBS')]

# fig, ax = plt.subplots(1,1,figsize=(14,6))

# #plt.rcParams['text.usetex'] = True



# #plot = sns.pointplot(ax=ax, data = vol_df_plot[vol_df_plot['data_type'] == 'OBS'], x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'rocket', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound']) #, style = 'data_type')


# #ax.set_xticklabels([])

# # plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Juan de Fuca'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Juan de Fuca'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Juan de Fuca'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle='None')


# # plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Georgia'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Georgia'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Strait of Georgia'),'E_LO_His_LO_casts']),capsize =3,  c='gray', alpha=0.5, linestyle ='None')


# plt.errorbar(np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'), 'date_ordinal']), np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'),'vol_km3']),yerr=np.array(vol_df.loc[(vol_df['data_type'] == 'OBS') & (vol_df['segment'] == 'Puget Sound'),'E_LO_His_LO_casts']),capsize =3, c ='gray', alpha=0.5, linestyle ='None')

# ax = sns.lineplot(data = vol_df_plot, x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'rocket_r', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], linewidth=3)# , style = 'data_type')


# ax.set(xlim=(date.toordinal(date(2008,1,1)), date.toordinal(date(2019,12,31))),ylim=(0,8))






# labels = [date(2008,1,1),  date(2009,1,1), 
#               date(2010,1,1),  date(2011,1,1), 
#               date(2012,1,1),  date(2013,1,1), 
#               date(2014,1,1),  date(2015,1,1), 
#               date(2016,1,1), date(2017,1,1), 
#               date(2018,1,1), date(2019,1,1)]


# new_labels = [date.toordinal(item) for item in labels]

# ax.set_xticks(new_labels)


# ax.set_xticklabels([
#                     '2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019'], rotation=0,
#     fontdict={'horizontalalignment':'center'})

# ax.set_facecolor("white")

# #plt.legend() #title = 'Basin [Order of Increasing Volume]')


# ax.set(xlabel = 'Date', ylabel = 'Hypoxic Volume [km^3]')

# plt.legend([],[], frameon=False)




# plt.grid(alpha=0.2)

# fig.tight_layout()

# plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_PS.png', transparent=True, dpi=500)


# # %%

# vol_df_plot = vol_df[(vol_df['year'] == 2017) & (vol_df['data_type'] != 'OBS')]

# vol_df_plot.loc[vol_df_plot['data_type'] =='LO His', 'data_type'] = 'Model Output'

# vol_df_plot.loc[vol_df_plot['data_type'] =='LO Casts', 'data_type'] = 'Model Volume-From-Casts'


# # %%

# fig, ax = plt.subplots(1,1,figsize=(13,5))



# ax.fill_between(x=np.array(vol_df_plot.loc[(vol_df_plot['segment'] == 'Strait of Juan de Fuca') & (vol_df_plot['data_type'] =='Model Output'), 'date_ordinal']),
#                 y1=np.array(vol_df_plot.loc[(vol_df_plot['segment'] == 'Strait of Juan de Fuca') & (vol_df_plot['data_type'] =='Model Output'),'vol_km3']),
#                 y2=np.array(vol_df_plot.loc[(vol_df_plot['segment'] == 'Strait of Juan de Fuca') & (vol_df_plot['data_type'] =='Model Volume-From-Casts'), 'vol_km3']), color='pink', alpha=0.3)

# ax.fill_between(x=np.array(vol_df_plot.loc[(vol_df_plot['segment'] == 'Strait of Georgia') & (vol_df_plot['data_type'] =='Model Output'), 'date_ordinal']),
#                 y1=np.array(vol_df_plot.loc[(vol_df_plot['segment'] == 'Strait of Georgia') & (vol_df_plot['data_type'] =='Model Output'),'vol_km3']),
#                 y2=np.array(vol_df_plot.loc[(vol_df_plot['segment'] == 'Strait of Georgia') & (vol_df_plot['data_type'] =='Model Volume-From-Casts'), 'vol_km3']), color='orange', alpha=0.3)


# ax.fill_between(x=np.array(vol_df_plot.loc[(vol_df_plot['segment'] == 'Puget Sound') & (vol_df_plot['data_type'] =='Model Output'), 'date_ordinal']),
#                 y1=np.array(vol_df_plot.loc[(vol_df_plot['segment'] == 'Puget Sound') & (vol_df_plot['data_type'] =='Model Output'),'vol_km3']),
#                 y2=np.array(vol_df_plot.loc[(vol_df_plot['segment'] == 'Puget Sound') & (vol_df_plot['data_type'] =='Model Volume-From-Casts'), 'vol_km3']), color='purple', alpha=0.3)






# ax = sns.lineplot(data = vol_df_plot, x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'rocket_r', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'] , style = 'data_type')

# ax.set(xlim=(date.toordinal(date(2017,1,1)), date.toordinal(date(2017,12,31))))






# labels = [date(2017,1,1),  date(2017,4,1), 
#               date(2017,7,1),  date(2017,10,1), 
#               date(2017,12,31)]


# new_labels = [date.toordinal(item) for item in labels]

# ax.set_xticks(new_labels)


# ax.set_xticklabels(['','April','July','October',''], rotation=0,
#     fontdict={'horizontalalignment':'center'})

# # ax.set_facecolor("white")

# #plt.legend() #title = 'Basin [Order of Increasing Volume]')


# ax.set(xlabel = '2017', ylabel = 'Hypoxic Volume [$km^3$]')

# h, l = ax.get_legend_handles_labels()

# legend = plt.legend(h[5:7], l[5:7], loc='upper left', ncol =1, )




# plt.grid(alpha=0.2)

# fig.tight_layout()

# plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol_errors.png', transparent=True, dpi=500)