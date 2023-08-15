"""
IDK YET

Test on mac in ipython:
run create_dicts -gtx cas6_v0_live -year 2000 -test True

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
import pickle

from time import time as Time


import numpy as np

import VFC_functions as vfun




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
    
    month_num = ['04']
    
    month_str = ['Apr']

    # month_num =  ['01', '02'] #,'03','04','05','06','07','08','09','10','11','12']
     
    # month_str = ['Jan','Feb'] #,'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 2 #mg/L DO

threshold_pct = 0.2 #m GOTTA BE PERCENT

var = 'S_g_kg'

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

    seg_list = ['Strait of Georgia']
    
# %%

oneoff = True

if oneoff == True:
    
    seg_list = ['Hood Canal', 'Whidbey Basin', 'Main Basin','Admiralty Inlet', 'South Sound']
    
# %%

if info_fn.exists() & fn.exists():
    
    info_df, df = vfun.getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var)
    
    
    info_df['month'] = info_df['time'].dt.month
    
    df['month'] = df['time'].dt.month

# %%

sub_thick_LO_his = {}

sub_thick_LO_casts = {}

sub_thick_obs = {}

sub_vol_LO_his = {}

sub_vol_LO_casts = {}

sub_vol_obs = {}

ii_casts = {}

jj_casts = {}

cid_dict = {}

surf_casts_array = {}

sub_casts_array_obs = {}

sub_casts_array_LO_casts = {}

sub_avg_obs = {}

sub_wtd_avg_obs = {}

# %%

for seg_name in seg_list:
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name]
    
    
    sub_thick_LO_his[seg_name] = {}

    sub_thick_LO_casts[seg_name] = {}

    sub_thick_obs[seg_name] = {}

    sub_vol_LO_his[seg_name] = {}

    sub_vol_LO_casts[seg_name] = {}
    
    sub_vol_obs[seg_name] = {}

    ii_casts[seg_name] = {}

    jj_casts[seg_name] = {}
    
    cid_dict[seg_name] = {}
    
    surf_casts_array[seg_name] = {}
    
    sub_casts_array_obs[seg_name] = {}
    
    sub_casts_array_LO_casts[seg_name] = {}
    
    sub_avg_obs[seg_name] = {}

    sub_wtd_avg_obs[seg_name] = {}
    
    # if oneoff == True:
    
    #     dt = pd.Timestamp('2017-01-01 01:30:00')
    #     fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
    
    #     G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)
    
    for mon_num, mon_str in zip(month_num, month_str):
        
        tt0 = Time()
        
        if info_fn.exists() & fn.exists():
            
        
            info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
            
            df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
        
            surf_casts_array[seg_name][int(mon_num)] = vfun.assignSurfaceToCasts(info_df_use, jjj, iii)
            
        else:
            
            print('no obs data for ' + str(Ldir['year']))
        
        
        dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
        
        fn_his = vfun.get_his_fn_from_dt(Ldir, dt) #note change from cfun
        
        
        
        if not fn_his.exists():
            
            dt = pd.Timestamp('2017-01-01 01:30:00')
            fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
        
            G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)
            
            print('no LO data for ' + mon_str + ' ' + str(Ldir['year']))
        
        
        else:
            
            G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)
            
            # sub_vol_LO_his[seg_name][int(mon_num)], sub_thick_LO_his[seg_name][int(mon_num)] = vfun.getLOHisSubVolThick(dv, dz, fn_his, jjj, iii, var, threshold_val)
        
            # if Ldir['year'] == 2017:
                
            # if info_fn.exists() & fn.exists():
                
                # vfun.extractLOCasts(Ldir, info_df_use, fn_his)
                
                # sub_vol_LO_casts[seg_name][int(mon_num)], sub_thick_LO_casts[seg_name][int(mon_num)], sub_casts_array_LO_casts[seg_name][int(mon_num)] = vfun.getLOCastsSubVolThick(Ldir, info_df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])
        
        
        if info_fn.exists() & fn.exists():
            
            if var == 'DO_mg_L':
        
                sub_vol_obs[seg_name][int(mon_num)], sub_thick_obs[seg_name][int(mon_num)], sub_casts_array_obs[seg_name][int(mon_num)] = vfun.getOBSCastsSubVolThick(info_df_use, df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])
                
                cid_dict[seg_name][int(mon_num)] = np.array(info_df_use.index)
                
                jj_casts[seg_name][int(mon_num)] = np.array(info_df_use['jj_cast'])
                
                ii_casts[seg_name][int(mon_num)] = np.array(info_df_use['ii_cast'])
            
            sub_avg_obs[seg_name][int(mon_num)] = vfun.getOBSAvgBelow(info_df_use, df_use, var, threshold_pct)
            
            sub_wtd_avg_obs[seg_name][int(mon_num)] = vfun.getOBSCastsWtdAvgBelow(info_df_use, df_use, var, threshold_pct, z_rho_grid, land_mask, dv, dz, h, jjj, iii, surf_casts_array[seg_name][int(mon_num)])
        
        
        print(seg_name + ' ' + mon_str + ' ' + str(Ldir['year']) + ' completed after %d sec' % (int(Time()-tt0)))
            
# %%

if Ldir['testing'] == False:

    dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
    
    fn_his = vfun.get_his_fn_from_dt(Ldir, dt) #note change from cfun
    
    # if fn_his.exists():
        
    #     with open((str(save_dir) + '/' + 'sub_thick_LO_his.pkl'), 'wb') as f: 
    #         pickle.dump(sub_thick_LO_his, f)  
        
    #     with open((str(save_dir) + '/' +  'sub_vol_LO_his.pkl'), 'wb') as f: 
    #         pickle.dump(sub_vol_LO_his, f)
    
    # # if Ldir['year'] == 2017:
    
    #     with open((str(save_dir) + '/' + 'sub_casts_array_LO_casts.pkl'), 'wb') as f: 
    #         pickle.dump(sub_casts_array_LO_casts, f)
            
    #     with open((str(save_dir) + '/' +  'sub_thick_LO_casts.pkl'), 'wb') as f: 
    #         pickle.dump(sub_thick_LO_casts, f)
            
    #     with open((str(save_dir) + '/' +  'sub_vol_LO_casts.pkl'), 'wb') as f: 
    #         pickle.dump(sub_vol_LO_casts, f)
    
    if info_fn.exists() & fn.exists():

        if var == 'DO_mg_L':        

            with open((str(save_dir) + '/' + 'sub_casts_array_obs_NEW.pkl'), 'wb') as f: 
                pickle.dump(sub_casts_array_obs, f)       
                
            with open((str(save_dir) + '/' +  'sub_thick_obs_NEW.pkl'), 'wb') as f: 
                pickle.dump(sub_thick_obs, f)
            
            with open((str(save_dir) + '/' + 'sub_vol_obs_NEW.pkl'), 'wb') as f: 
                pickle.dump(sub_vol_obs, f)  
                
            with open((str(save_dir) + '/' +  'surf_casts_array_NEW.pkl'), 'wb') as f: 
                pickle.dump(surf_casts_array, f)  
                
            with open((str(save_dir) + '/' +  'jj_casts_NEW.pkl'), 'wb') as f: 
                pickle.dump(jj_casts, f)
            
            with open((str(save_dir) + '/' +  'ii_casts_NEW.pkl'), 'wb') as f: 
                pickle.dump(ii_casts, f)  
                
            with open((str(save_dir) + '/' +  'cid_dict_NEW.pkl'), 'wb') as f: 
                pickle.dump(cid_dict, f)
            
            with open((str(save_dir) + '/' + 'sub_avg_obs_NEW.pkl'), 'wb') as f: 
                pickle.dump(sub_avg_obs, f)
            
            with open((str(save_dir) + '/' + 'sub_wtd_avg_obs_NEW.pkl'), 'wb') as f: 
                pickle.dump(sub_wtd_avg_obs, f)
                
        elif var == 'T_deg_C':
            
            with open((str(save_dir) + '/' + 'sub_avg_obs_T.pkl'), 'wb') as f: 
                pickle.dump(sub_avg_obs, f)
            
            with open((str(save_dir) + '/' + 'sub_wtd_avg_obs_T.pkl'), 'wb') as f: 
                pickle.dump(sub_wtd_avg_obs, f)
                
        elif var == 'S_g_kg':
            
            with open((str(save_dir) + '/' + 'sub_avg_obs_S.pkl'), 'wb') as f: 
                pickle.dump(sub_avg_obs, f)
            
            with open((str(save_dir) + '/' + 'sub_wtd_avg_obs_S.pkl'), 'wb') as f: 
                pickle.dump(sub_wtd_avg_obs, f)

# %%

print(str(Ldir['year']) + ' completed after %d sec' % (int(Time()-tt1)))

            