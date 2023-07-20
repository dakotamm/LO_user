"""
Blah

Test on mac in ipython:
run create_plot_df -gtx cas6_v0_live -year 2017 -test False

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

if fn_his.exists():


    with open((file_dir + '/' +'sub_vol_LO_his.pkl'), 'rb') as f: 
        sub_vol_LO_his = pickle.load(f) 
    
    with open((file_dir + '/' + 'sub_vol_LO_casts.pkl'), 'rb') as f: 
        sub_vol_LO_casts = pickle.load(f)
        
if info_fn.exists() & fn.exists():
        
    with open((file_dir + '/' + 'sub_vol_obs.pkl'), 'rb') as f: 
        sub_vol_obs = pickle.load(f)
        
    with open((file_dir + '/' + 'sub_avg_obs.pkl'), 'rb') as f: 
        sub_avg_obs = pickle.load(f)
        
    with open((file_dir + '/' + 'sub_wtd_avg_obs.pkl'), 'rb') as f: 
        sub_wtd_avg_obs = pickle.load(f)

# %%

vol_df = pd.DataFrame()


vol_df['segment'] = []

vol_df['month'] = []

vol_df['data_type'] = []

vol_df['vol_km3'] = []



for seg_name in seg_list:
    
    
    for (mon_num, mon_str) in zip(month_num,month_str):
        
        df_temp = pd.DataFrame()
                
        df_temp['segment'] = [seg_name]
        
        df_temp['month'] = [int(mon_num)]
        
        dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
        fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
        
        if fn_his.exists():
            
            df_temp0 = pd.DataFrame()
            
            df_temp0['segment'] = [seg_name]
            
            df_temp0['month'] = [int(mon_num)]
        
            df_temp0['data_type'] = ['LO His']
            
            df_temp0['vol_km3'] = [sub_vol_LO_his[seg_name][int(mon_num)]*1e-9] #convert to km^3
            
            vol_df = pd.concat([vol_df, df_temp0], ignore_index=True)
            
            
        if Ldir['year'] == 2017:
            
            df_temp1 = pd.DataFrame()
            
            df_temp1['segment'] = [seg_name]
            
            df_temp1['month'] = [int(mon_num)]
        
            df_temp1['data_type'] = 'LO Casts'
            
            df_temp1['vol_km3'] = [sub_vol_LO_casts[seg_name][int(mon_num)]*1e-9]
            
            vol_df = pd.concat([vol_df, df_temp1], ignore_index=True)
            
            
        if info_fn.exists() & fn.exists():    
        
        
            df_temp['data_type'] = ['OBS']
            
            df_temp['vol_km3'] = [sub_vol_obs[seg_name][int(mon_num)]*1e-9]
            
            vol_df = pd.concat([vol_df, df_temp], ignore_index=True)
        
        
vol_df.to_pickle((file_dir + '/' + 'vol_df.p'))       
        
#%%

# if fn_his.exists():

#     LO_his_mins = vol_df[vol_df['data_type'] == 'LO His'].groupby(['segment', 'data_type'])['vol_km3'].min()
    
#     LO_his_maxs = vol_df[vol_df['data_type'] == 'LO His'].groupby(['segment', 'data_type'])['vol_km3'].max()

#     LO_his_ranges = LO_his_maxs - LO_his_mins
    
#     LO_his_ranges = LO_his_ranges.to_frame().reset_index()
    

# if info_fn.exists() & fn.exists():

#     obs_mins = vol_df[vol_df['data_type'] == 'OBS'].groupby(['segment', 'data_type'])['vol_km3'].min()
    
#     obs_maxs = vol_df[vol_df['data_type'] == 'OBS'].groupby(['segment', 'data_type'])['vol_km3'].max()
    
#     obs_ranges = obs_maxs - obs_mins
    
#     obs_ranges = obs_ranges.to_frame().reset_index()


# vol_df_wide = vol_df.pivot(index=['month', 'segment'], columns = 'data_type', values='vol_km3').reset_index()


# if fn_his.exists():
    
#     vol_df_wide = pd.merge(vol_df_wide, LO_his_ranges, how='left', on='segment')


#     vol_df_wide = vol_df_wide.rename(columns = {'vol_km3':'LO_his_ranges'})
    
# if info_fn.exists() & fn.exists():

#     vol_df_wide = pd.merge(vol_df_wide, obs_ranges, how='left', on='segment')
    
#     vol_df_wide = vol_df_wide.rename(columns = {'vol_km3':'obs_ranges'})


# if fn_his.exists():
    
#     if info_fn.exists() & fn.exists():

#       #  if Ldir['year'] == 2017:
#         vol_df_wide = vol_df_wide[['month','segment','LO Casts','LO His', 'OBS', 'LO_his_ranges', 'obs_ranges']]
            
#         # else:
    
#         #     vol_df_wide = vol_df_wide[['month','segment', 'LO His', 'OBS', 'LO_his_ranges', 'obs_ranges']]
    
# else:
    
#     if info_fn.exists() & fn.exists():
        
#         vol_df_wide = vol_df_wide[['month','segment', 'OBS', 'obs_ranges']]
    
# # %%

# if fn_his.exists():
    
#     # if Ldir['year'] == 2017:


#     vol_df_wide['SE_LO_his_LO_casts'] = np.square(vol_df_wide['LO His'] - vol_df_wide['LO Casts'])
    
#     temp1 = vol_df_wide.groupby(['segment'])['SE_LO_his_LO_casts'].mean().to_frame().reset_index()
    
#     temp1 = temp1.rename(columns = {'SE_LO_his_LO_casts':'MSE_LO_his_LO_casts'})
    
#     temp1['RMSE_LO_his_LO_casts'] = np.sqrt(temp1['MSE_LO_his_LO_casts'])
    
#     vol_df_wide = pd.merge(vol_df_wide, temp1, how='left', on='segment')
    
#     vol_df_wide['norm_RMSE_LO_his_LO_casts'] = vol_df_wide['RMSE_LO_his_LO_casts'] / vol_df_wide['LO_his_ranges']

        
    
        


#     vol_df_wide['SE_OBS_LO_his'] = np.square(vol_df_wide['OBS'] - vol_df_wide['LO His'])



#     temp2 = vol_df_wide.groupby(['segment'])['SE_OBS_LO_his'].mean().to_frame().reset_index()

#     temp2 = temp2.rename(columns = {'SE_OBS_LO_his':'MSE_OBS_LO_his'})



#     temp2['RMSE_OBS_LO_his'] = np.sqrt(temp2['MSE_OBS_LO_his'])


#     vol_df_wide = pd.merge(vol_df_wide, temp2, how='left', on='segment')


#     vol_df_wide['norm_RMSE_OBS_LO_his'] = vol_df_wide['RMSE_OBS_LO_his'] / vol_df_wide['obs_ranges']


# vol_df_wide.to_pickle((file_dir + '/' + 'vol_df_wide.p'))

# %%

avg_df = pd.DataFrame()


avg_df['segment'] = []

avg_df['month'] = []

avg_df['data_type'] = []

avg_df['DO_avg_below_mg_L'] = []

for seg_name in seg_list:
    
    
    for (mon_num, mon_str) in zip(month_num,month_str):
        
        df_temp = pd.DataFrame()
                
        df_temp['segment'] = [seg_name]
        
        df_temp['month'] = [int(mon_num)]
        
        # dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
        # fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
        
        # if fn_his.exists():
            
        #     df_temp0 = pd.DataFrame()
            
        #     df_temp0['segment'] = [seg_name]
            
        #     df_temp0['month'] = [int(mon_num)]
        
        #     df_temp0['data_type'] = ['LO His']
            
        #     df_temp0['vol_km3'] = [sub_vol_LO_his[seg_name][int(mon_num)]*1e-9] #convert to km^3
            
        #     vol_df = pd.concat([vol_df, df_temp0], ignore_index=True)
            
            
        # if Ldir['year'] == 2017:
            
        #     df_temp1 = pd.DataFrame()
            
        #     df_temp1['segment'] = [seg_name]
            
        #     df_temp1['month'] = [int(mon_num)]
        
        #     df_temp1['data_type'] = 'LO Casts'
            
        #     df_temp1['vol_km3'] = [sub_vol_LO_casts[seg_name][int(mon_num)]*1e-9]
            
        #     vol_df = pd.concat([vol_df, df_temp1], ignore_index=True)
            
            
        if info_fn.exists() & fn.exists():    
        
        
            df_temp['data_type'] = ['OBS']
            
            df_temp['DO_avg_below_mg_L'] = [sub_avg_obs[seg_name][int(mon_num)]]
            
            avg_df = pd.concat([avg_df, df_temp], ignore_index=True)
        
        
vol_df.to_pickle((file_dir + '/' + 'avg_df.p'))

# %%

wtd_avg_df = pd.DataFrame()


wtd_avg_df['segment'] = []

wtd_avg_df['month'] = []

wtd_avg_df['data_type'] = []

wtd_avg_df['DO_wtd_avg_below_mg_L'] = []

for seg_name in seg_list:
    
    
    for (mon_num, mon_str) in zip(month_num,month_str):
        
        df_temp = pd.DataFrame()
                
        df_temp['segment'] = [seg_name]
        
        df_temp['month'] = [int(mon_num)]
        
        # dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
        # fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
        
        # if fn_his.exists():
            
        #     df_temp0 = pd.DataFrame()
            
        #     df_temp0['segment'] = [seg_name]
            
        #     df_temp0['month'] = [int(mon_num)]
        
        #     df_temp0['data_type'] = ['LO His']
            
        #     df_temp0['vol_km3'] = [sub_vol_LO_his[seg_name][int(mon_num)]*1e-9] #convert to km^3
            
        #     vol_df = pd.concat([vol_df, df_temp0], ignore_index=True)
            
            
        # if Ldir['year'] == 2017:
            
        #     df_temp1 = pd.DataFrame()
            
        #     df_temp1['segment'] = [seg_name]
            
        #     df_temp1['month'] = [int(mon_num)]
        
        #     df_temp1['data_type'] = 'LO Casts'
            
        #     df_temp1['vol_km3'] = [sub_vol_LO_casts[seg_name][int(mon_num)]*1e-9]
            
        #     vol_df = pd.concat([vol_df, df_temp1], ignore_index=True)
            
            
        if info_fn.exists() & fn.exists():    
        
        
            df_temp['data_type'] = ['OBS']
            
            df_temp['DO_wtd_avg_below_mg_L'] = [sub_wtd_avg_obs[seg_name][int(mon_num)]]
            
            wtd_avg_df = pd.concat([wtd_avg_df, df_temp], ignore_index=True)
        
        
vol_df.to_pickle((file_dir + '/' + 'wtd_avg_df.p'))


