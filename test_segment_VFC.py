"""
This tests new VFC spatial handling using VFC.
Last Modified: 10/4/2023

Test on mac in ipython:
run test_segment_VFC -gtx cas6_v0_live -year 2017 -test True

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
import pickle

from time import time as Time


import numpy as np

import VFC_functions_temp3 as vfun




# %%


Ldir = exfun.intro() # this handles the argument passing

tt1 = Time()
# %%

dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

if not fn_his.exists():
    
    dt = pd.Timestamp('2017-01-01 01:30:00')
    fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

    

if Ldir['testing']:
    
    month_num = ['09']
    
    month_str = ['Sep']

    # month_num =  ['01', '02'] #,'03','04','05','06','07','08','09','10','11','12']
     
    # month_str = ['Jan','Feb'] #,'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# threshold_val = 2 #mg/L DO

# threshold_pct = 0.2 #m GOTTA BE PERCENT

var_list = ['DO_mg_L', 'S_g_kg', 'T_deg_C', 'NO3_uM', 'NH4_uM', 'TA_uM', 'DIC_uM', 'DO_uM']

segments = 'regions' #custom (specify string list and string build list), basins, whole domain, sound and strait

# seg_build_list = optional
    
G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)


info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))

#save_dir = (Ldir['LOo'] / 'extract' / 'vfc' / ('DO_' + str(threshold_val) +'mgL_' + segments + '_months_' + (str(Ldir['year']))) )

#Lfun.make_dir(save_dir, clean=False)


# %%

jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

if Ldir['testing']:

    seg_list = ['Main Basin']
    
# %%


if info_fn.exists() & fn.exists():
    
    info_df, df = vfun.getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var_list)
    
    
    info_df['month'] = info_df['time'].dt.month
    
    df['month'] = df['time'].dt.month
    
    
# %%

data_dict = {}

avg_cast_f_dict = {}


for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    data_dict[seg_name] = {}
    
    avg_cast_f_dict[seg_name] = {}


    for mon_num, mon_str in zip(month_num, month_str):
        
        tt0 = Time()
        
        data_dict[seg_name][int(mon_num)] = {}
        
        avg_cast_f_dict[seg_name][int(mon_num)] = {}
        
        if info_fn.exists() & fn.exists():
        
            info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
            
            df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
                        
        
        for var in var_list:
            
            if var in df_use:
                
                if not df_use[~np.isnan(df_use[var])].empty:
                    
                    # condition where cast has only one value???
                    
                    avg_cast_f_dict[seg_name][int(mon_num)][var] = vfun.createAvgCast(var, info_df_use, df_use, jjj, iii, h, Ldir, mon_str, seg_name)
    
                    data_dict[seg_name][int(mon_num)][var] = vfun.fillSegments(var, z_rho_grid, info_df_use, df_use, jjj, iii, avg_cast_f_dict[seg_name][int(mon_num)][var])

                else:
                    
                    data_dict[seg_name][int(mon_num)][var] = np.full(np.shape(z_rho_grid[:,jjj,iii]), np.nan) # hacky but fine

                    

# %%

# for ref later

if ~Ldir['testing']:
            
    with open('/Users/dakotamascarenas/Desktop/' + str(Ldir['year']) +'_data_dict.pkl', 'wb') as f: # (str(save_dir) + '/' + '2017_data_dict.pkl'), 'wb') as f: 
        pickle.dump(data_dict, f)

    with open('/Users/dakotamascarenas/Desktop/' + str(Ldir['year']) +'_avg_cast_f_dict.pkl', 'wb') as f: # (str(save_dir) + '/' + '2017_data_dict.pkl'), 'wb') as f: 
        pickle.dump(avg_cast_f_dict, f)
        
# %%

print(str(Ldir['year']) + ' completed after %d sec' % (int(Time()-tt1)))

# %%

data_dict_full = {}


for mon_num, mon_str in zip(month_num, month_str):
    
    data_dict_full[int(mon_num)] = {}

    for var in var_list:
        
        if var in df:
        
            data_dict_full[int(mon_num)][var] = np.full(np.shape(z_rho_grid),np.nan)
                
            for seg_name in seg_list:
                
                jjj = jjj_dict[seg_name]
                
                iii = iii_dict[seg_name]
                
                data_dict_full[int(mon_num)][var][:,jjj,iii] = data_dict[seg_name][int(mon_num)][var]
            
    
# %% 

if ~Ldir['testing']:
            

    with open('/Users/dakotamascarenas/Desktop/' + str(Ldir['year']) +'_data_dict_full.pkl', 'wb') as f:
        pickle.dump(data_dict_full, f) 
            