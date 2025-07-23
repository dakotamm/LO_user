"""
IDK YET

Test on mac in ipython:
run get_sub_avg -gtx cas6_v0_live -year 2000 -test True

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
import pickle

import numpy as np

import VFC_functions as vfun




# %%


Ldir = exfun.intro() # this handles the argument passing


# %%

dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

if not fn_his.exists():
    
    dt = pd.Timestamp('2017-01-01 01:30:00')
    fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

    

if Ldir['testing']:
    
    month_num = ['04']
    
    month_str = ['Apr']

    # month_num =  ['01', '02'] #,'03','04','05','06','07','08','09','10','11','12']
     
    # month_str = ['Jan','Feb'] #,'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 2 #mg/L DO

threshold_depth = -40 #m

var = 'DO_mg_L'

segments = 'basins' #custom (specify string list and string build list), basins, whole domain, sound and strait

# seg_build_list = optional
    
G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)


info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))

save_dir = (Ldir['LOo'] / 'extract' / 'vfc' / ('DO_' + str(threshold_val) +'mgL_' + segments + '_months_' + (str(Ldir['year']))) )

Lfun.make_dir(save_dir, clean=False)

# %%

jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

if Ldir['testing']:

    seg_list = ['Hood Canal']
    
# %%

if info_fn.exists() & fn.exists():
    
    info_df, df = vfun.getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var)
    
    
    info_df['month'] = info_df['time'].dt.month
    
    df['month'] = df['time'].dt.month
    
# %%

sub_avg = {}

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    sub_avg[seg_name] = {}
    
    
    
    for mon_num, mon_str in zip(month_num, month_str): #CHANGE TO SEASONS - AVG OUT CASTS???
        
        if info_fn.exists() & fn.exists():
        
            info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
            
            df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
            
            
            sub_avg[seg_name][int(mon_num)] = vfun.getOBSAvgBelow(info_df_use, df_use, var, threshold_depth)
            
            print('avg done')
            
            sub_avg[seg_name][int(mon_num)] = vfun.getOBSAvgBelow(info_df_use, df_use, var, threshold_depth)
            
            
        print(seg_name + ' ' + mon_str + ' ' + str(Ldir['year']))
            
# %%

if Ldir['testing'] == False:

    if info_fn.exists() & fn.exists():
    
        with open((str(save_dir) + '/' + 'sub_avg.pkl'), 'wb') as f: 
            pickle.dump(sub_avg, f)
            
    
    

