"""
IDK YET

Test on mac in ipython:
run make_thick_plot_TEST -gtx cas6_v0_live -year 2021 -test True

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


Ldir = exfun.intro() # this handles the argument passing


# %%

    
dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

    

if Ldir['testing']:
    
    month_num = ['02']
    
    month_str = ['Feb']

    # month_num =  ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    # month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 2 #mg/L DO

var = 'DO_mg_L'

segments = 'sound_straits' #custom (specify string list and string build list), basins, whole domain, sound and strait

# seg_build_list = optional
    
G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)


info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))

# %%


jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

if Ldir['testing']:

    seg_list = ['Strait of Georgia']
    
# %%

info_df, df = vfun.getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var)

# %%

info_df['month'] = info_df['time'].dt.month

df['month'] = df['time'].dt.month


# %%

save_dir = (Ldir['LOo'] / 'extract' / 'vfc' / ('DO_' + str(threshold_val) +'mgL_' + segments + '_months_' + (str(Ldir['year']))) )

file_dir = str(save_dir)

    
with open((file_dir + '/' + 'sub_thick_obs.pkl'), 'rb') as f: 
    sub_thick_obs = pickle.load(f)  
    
with open((file_dir + '/' + 'cid_dict.pkl'), 'rb') as f: 
    cid_dict = pickle.load(f)  

    
# %%

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    iii = iii_dict[seg_name]
    
    
    min_lat = Lat[min(jjj) - 10]
    max_lat = Lat[max(jjj) + 10]
    
    min_lon = Lon[min(iii) - 10]
    max_lon = Lon[max(iii) + 10]    
    
    
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        
        sub_thick_obs_plot = np.empty(np.shape(land_mask))
        
        sub_thick_obs_plot.fill(np.nan)
        
        sub_thick_obs_plot[jjj,iii] = sub_thick_obs[seg_name][int(mon_num)]
        
        
        
        info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
                
        df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
        
        #cmap0 = cm.get_cmap('viridis', lut= len(info_df_use.index))

        
        
        pfun.start_plot(fs=14, figsize=(15,15))
        fig0, axes0 = plt.subplots(nrows=1, ncols=1, squeeze=False)
        
        c0 = axes0[0,0].pcolormesh(plon,plat, sub_thick_obs_plot, cmap='Greens', alpha = 0.8, vmin = 0, vmax = 300)
        
        for cid in info_df_use.index:
                    
            axes0[0,0].plot(info_df_use.loc[cid, 'lon'], info_df_use.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
            
        
        axes0[0,0].set_xlim([min_lon,max_lon])
        axes0[0,0].set_ylim([min_lat,max_lat])
        axes0[0,0].tick_params(labelrotation=45)
        axes0[0,0].set_title('Obs ' + mon_str + ' ' + str(Ldir['year']) + ' ' + seg_name + ' Sub-' + str(threshold_val) + ' mg/L DO')
        
        fig0.colorbar(c0,ax=axes0[0,0], label = 'Subthreshold Thickness [m]')

        
        pfun.add_coast(axes0[0,0])
        pfun.dar(axes0[0,0])

                

        fig0.tight_layout()
        plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+seg_name + '_sub_thick_'+str(threshold_val)+'_mg_L_DO_casts_' + str(Ldir['year']) + '_00' + mon_num+'.png', bbox_inches='tight')


