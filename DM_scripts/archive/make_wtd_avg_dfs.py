"""
Blah

Test on mac in ipython:
run make_vol_dfs -gtx cas6_v0_live -year 2017 -test False

"""

import pandas as pd
import numpy as np

from lo_tools import extract_argfun as exfun
import pickle

import VFC_functions as vfun

# %%


Ldir = exfun.intro() # this handles the argument passing

# %%

dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
    

if Ldir['testing']:

    month_num =  ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 2 #mg/L DO

var = 'DO_mg_L'

segments = 'basins' #custom (specify string list and string build list), basins, whole domain, sound and strait

# seg_build_list = optional
    
# G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h= vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)

info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))

save_dir = (Ldir['LOo'] / 'extract' / 'vfc' / ('DO_' + str(threshold_val) +'mgL_' + segments + '_months_' + (str(Ldir['year']))) )

# %%


jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

if Ldir['testing']:

    seg_list = ['Whidbey Basin']
    
    
    
# %%

file_dir = str(save_dir)

        
if info_fn.exists() & fn.exists():
        
    with open((file_dir + '/' + 'sub_wtd_avg_obs.pkl'), 'rb') as f: 
        sub_wtd_avg_obs = pickle.load(f)  

# %%

wtd_avg_df = pd.DataFrame()


wtd_avg_df['segment'] = []

wtd_avg_df['month'] = []

wtd_avg_df['data_type'] = []

wtd_avg_df['DO_wtd_avg_mg_L'] = []



for seg_name in seg_list:
    
    
    for (mon_num, mon_str) in zip(month_num,month_str):
        
        df_temp = pd.DataFrame()
                
        df_temp['segment'] = [seg_name]
        
        df_temp['month'] = [int(mon_num)]
        
        dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
        fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
        
        if info_fn.exists() & fn.exists():
            
            df_temp0 = pd.DataFrame()
            
            df_temp0['segment'] = [seg_name]
            
            df_temp0['month'] = [int(mon_num)]
        
            df_temp0['data_type'] = ['OBS']
            
            df_temp0['DO_wtd_avg_mg_L'] = [sub_wtd_avg_obs[seg_name][int(mon_num)]] #convert to km^3
            
            wtd_avg_df = pd.concat([wtd_avg_df, df_temp0], ignore_index=True)
        
        
wtd_avg_df.to_pickle((file_dir + '/' + 'wtd_avg_df.p'))       
        


