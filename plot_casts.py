"""

run plot_casts -gtx cas6_v0_live -test True

"""

import pandas as pd

from lo_tools import Lfun, zfun
from lo_tools import extract_argfun as exfun
from lo_tools import plotting_functions as pfun
from lo_tools import forcing_argfun as ffun
import pickle

from time import time as Time

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import VFC_functions_temp2 as vfun

from pathlib import PosixPath


# %%

#Ldir = ffun.intro() # this handles the argument passing

Ldir = exfun.intro() # this handles the argument passing


# %%

year_list = np.arange(1930,2023)

source_list = ['dfo1', 'ecology', 'nceiSalish']

otype_list = ['ctd','bottle']

month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']

month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

var_list = ['DO (uM)', 'CT', 'SA', 'NO3 (uM)', 'NH4 (uM)', 'TA (uM)', 'DIC (uM)']

dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)

# %%

if Ldir['testing']:
    
    source_list = ['ecology']
    
    otype_list = ['ctd']
    
    year_list = ['2017', '2018']

# %%

segments = 'regions'

jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

if Ldir['testing']:

    seg_list = ['Main Basin']

# %%

casts_df = pd.load_pickle('/Users/dakotamascarenas/Desktop/casts_df.p')


# %%

casts_df_use = casts_df[(casts_df['segment'].isin(seg_list)) & (casts_df['var'].isin(var_list))]

sns.set(font_scale=1)

for var in var_list:

    plt.figure()
    
    sns.relplot(data=casts_df_use[casts_df_use['var'] == var], x="time", y="z", style="source", col="segment", row="var", hue='month', edgecolor=None)
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + var.replace('(', '').replace(')', '').replace(' ', '_') + '_all_casts.png', bbox_inches='tight', dpi=500)




# %%

sns.set_style("whitegrid")


for year in year_list:
    
    for var in var_list:
        
        ncol = len(seg_list)
    
        fig, ax = plt.subplots(1, ncol, figsize=(40,5),sharex=True,sharey=True)
        
        c=0
        
        for seg_name in seg_list:
        
            sns.scatterplot(data=casts_df_use[(casts_df_use['var'] == var) & (casts_df_use['year']==year) & (casts_df_use['segment'] == seg_name)], x="month", y="z", style="source", hue='type', ax = ax[c])
            
            ax[c].set_title(seg_name)
            
            ax[c].set_xlim([0, 13])
            
            ax[c].set_xlabel('month')
            
            ax[c].set_ylabel('min_z')
            
            c+=1
        
        fig.suptitle(var+ ' ' + str(year))
        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + var.replace('(', '').replace(')', '').replace(' ', '_') + '_' + str(year)+'_casts.png', bbox_inches='tight', dpi=500)



# %%




nrow = len(var_list)

ncol = 2

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    for year in year_list:
        
        for mon_num, mon_str in zip(month_num, month_str):
            
            casts_df_use = casts_df[(casts_df['segment'] == seg_name) & (casts_df['year']==year) & casts_df['month'] ==int(mon_num)]
    
            fig, ax = plt.subplots(nrow, ncol, figsize=(10,40))
            
            c = 0
            
            for var in var_list:
                
                ax[c,1] = sns.scatterplot(casts_df_use[casts_df_use['var']== var], x = 
    
    