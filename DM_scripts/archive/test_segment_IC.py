"""
This tests new VFC spatial handling using VFC.
Last Modified: 10/4/2023

Test on mac in ipython:
run test_segment_IC -g cas6 -d 2012.01.01 -r backfill -s new -f ocn00 -test True

"""

import pandas as pd

from lo_tools import Lfun, zrfun
from lo_tools import extract_argfun as exfun
from lo_tools import forcing_argfun2 as ffun
from lo_tools import plotting_functions as pfun


import pickle

from time import time as Time


import numpy as np

import VFC_functions as vfun

import matplotlib.pyplot as plt




# %%


#Ldir = exfun.intro() # this handles the argument passing
Ldir = ffun.intro() # this handles all the argument passing


tt1 = Time()
# %%

    
dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

G, S, T = zrfun.get_basic_info(fn_his)
land_mask = G['mask_rho']
Lon = G['lon_rho'][0,:]
Lat = G['lat_rho'][:,0]
# plon,plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])
# z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
# dz = np.diff(z_w_grid,axis=0)
# dv = dz*G['DX']*G['DY']
h = G['h']

var_list = ['CT', 'SA', 'DO (uM)', 'Chl (mg m-3)', 'NO3 (uM)', 'NO2 (uM)', 'NH4 (uM)', 'TA (uM)', 'DIC (uM)']


# %%

ctd_or_bio = 'both'

month_window = 1

info_df = pd.DataFrame()
df = pd.DataFrame()

info_df, df, jjj_dict, iii_dict, seg_list = vfun.get_casts_IC(Ldir, ctd_or_bio, month_window)

# %%

avg_cast_f_dict = {}

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    info_df_use = info_df[(info_df['segment'] == seg_name)]
            
    df_use = df[(df['segment'] == seg_name)]
    
    avg_cast_f_dict[seg_name] = {}

    
    for var in var_list:
        
        if var in df_use:
            
            if not df_use[~np.isnan(df_use[var])].empty:
        
                avg_cast_f_dict[seg_name][var] = vfun.createAvgCast(var, info_df_use, df_use, jjj, iii, h, Ldir)

    
# %%


for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    h_min = -h[jjj,iii].max()

        
    info_df_use = info_df[(info_df['segment'] == seg_name)]
            
    df_use = df[(df['segment'] == seg_name)]
        
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df_use['cid'].unique())))
        
    fig = plt.figure(figsize=(20,10))
    
    c=1
        
    for var in var_list:
        
        ax0 = fig.add_subplot(2,7, c)
        
        if var in avg_cast_f_dict[seg_name].keys():
        
            avg_cast_f = avg_cast_f_dict[seg_name][var]
                                
            n=0
            
            for cid in df_use['cid'].unique():
                
                df_plot = df_use[df_use['cid'] == cid]
                
                if ~np.isnan(df_plot[var].iloc[0]):
                                
                    df_plot.plot(x=var, y='z', style= '.', ax=ax0, color = colors[n], markersize=5, label=int(cid))
        
                n+=1
                                
            ax0.plot(avg_cast_f(np.linspace(h_min,0)), np.linspace(h_min,0), '-k', label='avg cast')
        
        if c == 1 or c == 8:
            
            ax0.set_ylabel('z [m]')
            
        if c == 5:
            
            c+=3
            
        else:
            
            c+=1
        
        ax0.set_xlabel(var)

        #ax0.set_ylabel('z [m]')
        
        ax0.legend().remove()
                    
        ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
        
    
    ax1 = plt.subplot2grid(shape=(2, 7), loc=(0,5),rowspan=2, colspan=2)
    
    ax1.axis([Lon[iii.min() -10], Lon[iii.max() + 10], Lat[jjj.min() -10], Lat[jjj.max() + 10]])
    ax1.set_xlabel('Longitude [deg]')
    ax1.set_ylabel('Latitude [deg]')
    
    pfun.add_coast(ax1)
    pfun.dar(ax1)
            
    n = 0
    
    for cid in df_use['cid'].unique():
        
        df_plot = df_use[df_use['cid'] == cid]
 
        if df_plot['type'].unique() =='bottle':
                                    
                ax1.scatter(df_plot.iloc[0]['lon'], df_plot.iloc[0]['lat'], edgecolor='k', facecolor=colors[n], marker='>', label = int(cid))
        
        else:
            
                ax1.scatter(df_plot.iloc[0]['lon'], df_plot.iloc[0]['lat'], edgecolor='k', facecolor=colors[n], marker='<', label = int(cid))
        
        n+=1
        

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax1.grid(False)

    plt.suptitle(seg_name + ' ' + str(month_window) + ' Month Window')
        
    fig.tight_layout()
                    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/avg_casts_' + seg_name + '_' + str(month_window) + 'monthwindow.png', bbox_inches='tight')