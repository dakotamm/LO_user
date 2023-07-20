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

from datetime import datetime

# %%

threshold_val = 2

seg_str = ['sound_straits']

years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

#years = [2017, 2018, 2019]

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

#vol_df = vol_df[vol_df['segment']!= 'Strait of Georgia']

#vol_df.loc[vol_df['vol_km3'] > 100, 'vol_km3'] = np.nan

# %%

from datetime import date

fig, ax = plt.subplots(1,1,figsize=(16,8))

ax = sns.lineplot(data = vol_df, x = 'date_ordinal', y = 'vol_km3', hue = 'segment', palette = 'rocket', hue_order = ['Strait of Georgia', 'Strait of Juan de Fuca', 'Puget Sound'], style = 'data_type')#, size='segment', size_order = ['Tacoma Narrows', 'South Sound', 'Admiralty Inlet', 'Hood Canal', 'Whidbey Basin', 'Main Basin', 'Strait of Juan de Fuca', 'Strait of Georgia'], sizes=(3, 1))

ax.set_title(str(years[0]) +'-'+ str(years[-1])+ ' Sub-' +str(threshold_val) + ' mg/L [DO]')

ax.set(xlim=(vol_df['date_ordinal'].min()-1, vol_df['date_ordinal'].max()+1))

# ax.set_xticks(range(46))




# ax.set_xticks([])
# ax.set_xticks([], minor=True)










# ax.vlines([datetime.datetime(2018,1,1),datetime.datetime(2019,1,1)],0,15, alpha=.5)
# ax.set_xticks([datetime.datetime(2017,1,1),datetime.datetime(2017,7,1),datetime.datetime(2018,1,1),
#     datetime.datetime(2018,7,1),datetime.datetime(2019,1,1),datetime.datetime(2019,7,1),datetime.datetime(2019,12,31)])
# ax.set_xticklabels(['','2017','','2018','','2019',''], rotation=0,
#     fontdict={'horizontalalignment':'center'})

new_labels = [date.fromordinal(int(item)) for item in ax.get_xticks()]

# ax.set_xticks([])
# ax.set_xticks([], minor=True)

# ax.vlines([date(2001,1,1),date(2022,1,1)],0,45, alpha=.5)

# labels = [date(2000,1,1), date(2000,7,1), date(2001,1,1), date(2001,7,1),
#               date(2002,1,1), date(2002,7,1), date(2003,1,1), date(2003,7,1),
#               date(2004,1,1), date(2004,7,1), date(2005,1,1), date(2005,7,1),
#               date(2006,1,1), date(2006,7,1), date(2007,1,1), date(2007,7,1),
#               date(2008,1,1), date(2008,7,1), date(2009,1,1), date(2009,7,1),
#               date(2010,1,1), date(2010,7,1), date(2011,1,1), date(2011,7,1),
#               date(2012,1,1), date(2012,7,1), date(2013,1,1), date(2013,7,1),
#               date(2014,1,1), date(2014,7,1), date(2015,1,1), date(2015,7,1), 
#               date(2016,1,1), date(2016,7,1), date(2017,1,1), date(2017,7,1),
#               date(2018,1,1), date(2018,7,1), date(2019,1,1), date(2019,7,1),
#               date(2020,1,1), date(2020,7,1), date(2021,1,1), date(2021,7,1),
#               date(2022,1,1), date(2022,7,1)]

labels = [date(2000,1,1), date(2001,1,1), date(2002,1,1), date(2003,1,1),
              date(2004,1,1), date(2005,1,1), 
              date(2006,1,1), date(2007,1,1), 
              date(2008,1,1),  date(2009,1,1), 
              date(2010,1,1),  date(2011,1,1), 
              date(2012,1,1),  date(2013,1,1), 
              date(2014,1,1),  date(2015,1,1), 
              date(2016,1,1), date(2017,1,1), 
              date(2018,1,1), date(2019,1,1), 
              date(2020,1,1),  date(2021,1,1), 
              date(2022,1,1), ]

new_labels = [date.toordinal(item) for item in labels]

ax.set_xticks(new_labels)


# ax.set_xticks([date(2000,1,1), date(2000,7,1), date(2001,1,1), date(2001,7,1),
#              date(2002,1,1), date(2002,7,1), date(2003,1,1), date(2003,7,1),
#              date(2004,1,1), date(2004,7,1), date(2005,1,1), date(2005,7,1),
#              date(2006,1,1), date(2006,7,1), date(2007,1,1), date(2007,7,1),
#              date(2008,1,1), date(2008,7,1), date(2009,1,1), date(2009,7,1),
#              date(2010,1,1), date(2010,7,1), date(2011,1,1), date(2011,7,1),
#              date(2012,1,1), date(2012,7,1), date(2013,1,1), date(2013,7,1),
#              date(2014,1,1), date(2014,7,1), date(2015,1,1), date(2015,7,1), 
#              date(2016,1,1), date(2016,7,1), date(2017,1,1), date(2017,7,1),
#              date(2018,1,1), date(2018,7,1), date(2019,1,1), date(2019,7,1),
#              date(2020,1,1), date(2020,7,1), date(2021,1,1), date(2021,7,1),
#              date(2022,1,1), date(2022,7,1)])

# ax.set_xticklabels(['2000', '', '2001','','2002','','2003','','2004','','2005','','2006','','2007','',
#                     '2008','','2009','','2010','','2011','','2012','','2013','','2014','','2015','','2016','','2017','','2018','','2019','',
#                     '2020','','2021','', '2022',''], rotation=0,
#     fontdict={'horizontalalignment':'center'})

ax.set_xticklabels(['2000','2001','2002','2003','2004','2005','2006','2007',
                    '2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019',
                    '2020','2021', '2022'], rotation=0,
    fontdict={'horizontalalignment':'center'})

#ax.set(xticklabels=new_labels)


# ax.set_xticklabels(['','2017','','2018','','2019',''], rotation=0,
#     fontdict={'horizontalalignment':'center'})

#ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment= 'right')

plt.legend() #title = 'Basin [Order of Increasing Volume]')

plt.grid()

fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(years[0]) +'-'+ str(years[-1]) + '_sub_2mg_vol.png')