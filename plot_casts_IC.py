"""

run plot_casts_VFC -gtx cas6_v0_live -test True

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
from lo_tools import plotting_functions as pfun
import pickle

from time import time as Time

import matplotlib.pyplot as plt

import matplotlib.colors as cm


import numpy as np

import VFC_functions as vfun

# %%

Ldir = exfun.intro() # this handles the argument passing


# %%

with open('/Users/dakotamascarenas/Desktop/' + str(Ldir['year']) + '_avg_cast_f_dict.pkl', 'rb') as f: # (str(save_dir) + '/' + '2017_data_dict.pkl'), 'wb') as f: 
    avg_cast_f_dict = pickle.load(f)
    
# %%

dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)

if Ldir['testing']:
    
    month_num = ['09']
    
    month_str = ['Sep']

    # month_num =  ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    # month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 2 #mg/L DO

var_list = ['DO_mg_L', 'S_g_kg', 'T_deg_C', 'NO3_uM', 'NH4_uM', 'TA_uM', 'DIC_uM']

segments = 'regions' #custom (specify string list and string build list), basins, whole domain, sound and strait

info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))

# %%


jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

# %%

info_df, df = vfun.getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var_list)

# %%

info_df['month'] = info_df['time'].dt.month

df['month'] = df['time'].dt.month

# %%

if Ldir['testing']:

    seg_list = ['Main Basin']
    
# %%

# fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)

# land_mask_plot = land_mask.copy()

# land_mask_plot[land_mask ==0] = np.nan


for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    h_min = -h[jjj,iii].max()


    for (mon_num, mon_str) in zip(month_num, month_str):
        
        info_df_use = info_df[(info_df['month'] == int(mon_num)) & (info_df['segment'] == seg_name)]
                
        df_use = df[(df['month'] == int(mon_num)) & (df['segment'] == seg_name)]
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(df_use['cid'].unique())))
        
        fig = plt.figure(figsize=(20,10))
        
        c=1
        
        for var in var_list:
            
            ax0 = fig.add_subplot(2,5, c)
            
            if var in avg_cast_f_dict[seg_name][int(mon_num)].keys():
            
                avg_cast_f = avg_cast_f_dict[seg_name][int(mon_num)][var]
                                    
                n=0
                
                for cid in df_use['cid'].unique():
                    
                    df_plot = df_use[df_use['cid'] == cid]
                    
                    if ~np.isnan(df_plot[var].iloc[0]):
                                    
                        df_plot.plot(x=var, y='z', style= '.', ax=ax0, color = colors[n], markersize=5, label=int(cid))
            
                    n+=1
                                    
                ax0.plot(avg_cast_f(np.linspace(h_min,0)), np.linspace(h_min,0), '-k', label='avg cast')
                
            if c == 3:
                
                c+=2
                
            c+=1
            
            ax0.set_xlabel(var)

            ax0.set_ylabel('z [m]')
            
            ax0.legend().remove()
                        
            ax0.grid(color = 'lightgray', linestyle = '--', alpha=0.5)
            
        
        ax1 = plt.subplot2grid(shape=(2, 5), loc=(0,4),rowspan=2)
        
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

        plt.suptitle(seg_name + '_' + str(Ldir['year']) + '_' + mon_str)
        
        fig.tight_layout()
                        
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/avg_casts_' + seg_name + '_' + str(Ldir['year']) + '_' + mon_str + '.png', bbox_inches='tight')
        
        
# %%

