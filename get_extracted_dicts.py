"""
IDK YET

Test on mac in ipython:
run get_extracted_dicts -gtx cas6_v0_live -source ecology -otype ctd -year 2017 -test False

"""

import pandas as pd

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
import pickle

import VFC_functions as vfun



# %%


Ldir = exfun.intro() # this handles the argument passing


# %%

dt = pd.Timestamp(str(Ldir['year']) + '-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

if ~fn_his.exists():
    
    dt = pd.Timestamp('2017-01-01 01:30:00')
    fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

    

if Ldir['testing']:
    
    month_num = ['09']
    
    month_str = ['Sep']

    # month_num =  ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    # month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
else:
    
    month_num = ['01', '02','03','04','05','06','07','08','09','10','11','12']
    
    month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

threshold_val = 2 #mg/L DO

var = 'DO_mg_L'

segments = 'whole' #custom (specify string list and string build list), basins, whole domain, sound and strait

# seg_build_list = optional
    
G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)


info_df_dir = (Ldir['LOo'] / 'obs' / 'vfc')

df_dir = (Ldir['LOo'] / 'obs' / 'vfc' )

info_fn = (info_df_dir / ('info_' + str(Ldir['year']) + '.p'))

fn = (df_dir / (str(Ldir['year']) + '.p'))

save_dir = (Ldir['LOo'] / 'extract' / 'vfc' / 'DO_' + str(threshold_val) +'mgL_' + segments + '_months_' + (str[Ldir['year']]) )

Lfun.make_dir(save_dir, clean=False)


# %%

jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

if Ldir['testing']:

    seg_list = ['Strait of Juan de Fuca']
    
# %%

info_df, df = vfun.getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var)

# %%

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
    
    
    for (mon_num, mon_str) in zip(month_num, month_str):
        
        info_df_use = info_df[(info_df['segment'] == seg_name) & (info_df['month'] == int(mon_num))]
        
        df_use = df[(df['segment'] == seg_name) & (df['month'] == int(mon_num))]
        
        
        dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')
        
        fn_his = vfun.get_his_fn_from_dt(Ldir, dt) #note change from cfun
        
        if ~fn_his.exists():
            
            dt = pd.Timestamp('2017-01-01 01:30:00')
            fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
        
            G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)
        
        
        else:
            
            G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)
            
            sub_vol_LO_his[seg_name][int(mon_num)], sub_thick_LO_his[seg_name][int(mon_num)] = vfun.getLOHisSubVolThick(dv, dz, fn_his, jjj, iii, var, threshold_val)
        
        
        surf_casts_array[seg_name][int(mon_num)] = vfun.assignSurfaceToCasts(info_df_use, jjj, iii)
        

       # sub_vol_LO_casts[seg_name][int(mon_num)], sub_thick_LO_casts[seg_name][int(mon_num)], sub_casts_array_LO_casts[seg_name][int(mon_num)] = vfun.getLOCastsSubVolThick(Ldir, info_df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])
        
        
        sub_vol_obs[seg_name][int(mon_num)], sub_thick_obs[seg_name][int(mon_num)], sub_casts_array_obs[seg_name][int(mon_num)] = vfun.getOBSCastsSubVolThick(info_df_use, df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array[seg_name][int(mon_num)])
        
        print(seg_name + ' ' + mon_str + ' ' + str(Ldir['year']))
        
# %%

dt = pd.Timestamp(str(Ldir['year']) + '-'+mon_num+'-01 01:30:00')

fn_his = vfun.get_his_fn_from_dt(Ldir, dt) #note change from cfun

if fn_his.exists():
    
    with open((save_dir + 'sub_thick_LO_his.pkl'), 'wb') as f: 
        pickle.dump(sub_thick_LO_his, f)  
    
    with open((save_dir + 'sub_vol_LO_his.pkl'), 'wb') as f: 
        pickle.dump(sub_vol_LO_his, f)



# with open((save_dir + 'sub_casts_array_LO_casts.pkl'), 'wb') as f: 
#     pickle.dump(sub_casts_array_LO_casts, f)

with open((save_dir + 'sub_casts_array_obs.pkl'), 'wb') as f: 
    pickle.dump(sub_casts_array_obs, f)      

# with open((save_dir + 'sub_thick_LO_casts.pkl'), 'wb') as f: 
#     pickle.dump(sub_thick_LO_casts, f)

# with open((save_dir + 'sub_thick_LO_his.pkl'), 'wb') as f: 
#     pickle.dump(sub_thick_LO_his, f)  
    
with open((save_dir + 'sub_thick_obs.pkl'), 'wb') as f: 
    pickle.dump(sub_thick_obs, f)

# with open((save_dir + 'sub_vol_LO_casts.pkl'), 'wb') as f: 
#     pickle.dump(sub_vol_LO_casts, f)      

# with open((save_dir + 'sub_vol_LO_his.pkl'), 'wb') as f: 
#     pickle.dump(sub_vol_LO_his, f)

with open((save_dir + 'sub_vol_obs.pkl'), 'wb') as f: 
    pickle.dump(sub_vol_obs, f)  
    
with open((save_dir + 'surf_casts_array.pkl'), 'wb') as f: 
    pickle.dump(surf_casts_array, f)  
    
with open((save_dir + 'jj_casts.pkl'), 'wb') as f: 
    pickle.dump(jj_casts, f)

with open((save_dir + 'ii_casts.pkl'), 'wb') as f: 
    pickle.dump(ii_casts, f)  
    
with open((save_dir + 'cid_dict.pkl'), 'wb') as f: 
    pickle.dump(cid_dict, f)