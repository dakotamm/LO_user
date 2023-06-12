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

# %%

threshold_val = 2

seg_str = ['sound_straits']

years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

vol_df = pd.DataFrame()

for year in years:
    
    for segs in seg_str:
    
        fn = '/Users/dakotamascarenas/LO_output/extract/vfc/DO_' + str(threshold_val) + 'mgL_' + segs + '_months_' + str(year) +'/vol_df.p'
        
        vol_df_temp = pd.read_pickle(fn)
        
        vol_df_temp['year'] = year
        
        vol_df_temp['day'] = 1
        
        vol_df = pd.concat([vol_df, vol_df_temp], ignore_index=True)
        
        
# %%

vol_df['date'] = pd.to_datetime(dict(year=vol_df.year, month=vol_df.month, day=vol_df.day))

vol_df['date_ordinal'] = vol_df['date'].apply(lambda date: date.toordinal())

# vol_df.loc[vol_df['vol_km3'] > 100, 'vol_km3'] = np.nan

# %%

from datetime import date

fig, ax = plt.subplots(1,1,figsize=(12,8))

sns.lineplot(data = vol_df, x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'rocket', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], style = 'data_type')#, size='segment', size_order = ['Tacoma Narrows', 'South Sound', 'Admiralty Inlet', 'Hood Canal', 'Whidbey Basin', 'Main Basin', 'Strait of Juan de Fuca', 'Strait of Georgia'], sizes=(3, 1))

ax.set_title(str(years[0]) +'-'+ str(years[-1])+ ' Sub-' +str(threshold_val) + ' mg/L [DO]')

ax.set(xlim=(vol_df['date_ordinal'].min()-1, vol_df['date_ordinal'].max()+1))

new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]

ax.set(xticklabels=new_labels)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment= 'right')

plt.legend() #title = 'Basin [Order of Increasing Volume]')

plt.grid()

fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol.png')