"""
Functions to add biogeochemical fields to a clm file.
"""

import numpy as np
import matplotlib.path as mpath
from lo_tools import Lfun, zfun, zrfun

verbose = False

# D functions

from dateutil.relativedelta import relativedelta

import VFC_functions as vfun

import pandas as pd

from datetime import datetime, timedelta

import copy



def setup_bio_casts(Ldir, source_list=['ecology','dfo1','nceiSalish']):
    
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
    
    vol_dir, v_df, j_dict, i_dict, seg_list = vfun.getSegmentInfo(Ldir)
    
    seg_str_list = 'whole'
    
    jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(seg_str_list, j_dict, i_dict)
    
    info_df_rough = pd.DataFrame()
    
    df_rough = pd.DataFrame()
    
    info_df = pd.DataFrame()
    
    df = pd.DataFrame()
    
    this_dt = datetime.strptime(Ldir['date_string'], Lfun.ds_fmt)
        
    start_dt = this_dt - relativedelta(months = 1)
    
    end_dt = this_dt + relativedelta(months = 1)
    
    if start_dt.year != end_dt.year:
        
        year_list = [start_dt.year, end_dt.year]
        
    else:
        
        year_list = [this_dt.year]
    
    for year in year_list:
       
        for source in source_list:
        
            info_fn_in =  Ldir['LOo'] / 'obs' / source / 'bottle' / ('info_' + str(year) + '.p')
        
            info_df_temp0 = pd.read_pickle(info_fn_in)
        
            info_df_temp0['time'] = info_df_temp0['time'].astype('datetime64[ns]')
            
            info_df_temp0['source'] = source
            
            info_df_temp1 = info_df_temp0[(info_df_temp0['time'] >= start_dt) & (info_df_temp0['time'] <= end_dt)]
               
            info_df_rough = pd.concat([info_df_rough, info_df_temp1], ignore_index=True)
            
            info_df.index.name = 'cid'
            
            
    for year in year_list:
       
        for source in source_list:
            
            fn_in =  Ldir['LOo'] / 'obs' / source / 'bottle' / (+ str(year) + '.p')
        
            df_temp0 = pd.read_pickle(fn_in)
        
            df_temp0['time'] = df_temp0['time'].astype('datetime64[ns]')
            
            df_temp0['source'] = source
            
            df_temp1 = df_temp0[(df_temp0['time'] >= start_dt) & (df_temp0['time'] <= end_dt)]
            
            df_temp1['cid'] = df_temp1['cid'] + info_df_rough[info_df_rough['source'] == source].index.min() 
               
            df_rough = pd.concat([df_rough, df_temp1], ignore_index=True)
            
            
    depth_threshold = 0.2 # percentage of bathymetry the cast can be from the bottom to be accepted
        
    info_df_rough['ix'] = 0

    info_df_rough['iy'] = 0
    
    info_df_rough['segment'] = 'None'
    
    info_df_rough['ii_cast'] = np.nan

    info_df_rough['jj_cast'] = np.nan
    
    
    for cid in info_df_rough.index:

        info_df_rough.loc[cid,'ix'] = zfun.find_nearest_ind(Lon, info_df_rough.loc[cid,'lon'])

        info_df_rough.loc[cid,'iy'] = zfun.find_nearest_ind(Lat, info_df_rough.loc[cid,'lat'])
        
        if land_mask[info_df.loc[cid,'iy'], info_df_rough.loc[cid,'ix']] == 1:
            
            info_df_rough.loc[cid, 'ii_cast'] = info_df_rough.loc[cid, 'ix']
            
            info_df_rough.loc[cid, 'jj_cast'] = info_df_rough.loc[cid, 'iy']
    
    for seg_name in seg_list:
        
        ij_pair = list(zip(iii_dict[seg_name],jjj_dict[seg_name]))
        
        for cid in info_df.index:        
              
            pair = (info_df_rough.loc[cid,'ix'].tolist(), info_df_rough.loc[cid,'iy'].tolist())
            
            if pair in ij_pair:
                info_df_rough.loc[cid,'segment'] = seg_name
    
    info_df_rough = info_df_rough[~(np.isnan(info_df_rough['jj_cast'])) & ~(np.isnan(info_df_rough['ii_cast']))]
    
    info_df_rough = info_df_rough[info_df_rough['segment'] == 'All Segments']
    
    
    df_rough = pd.merge(df_rough, info_df_rough[['ix','iy','ii_cast','jj_cast','segment']], how='left', on=['cid'])
    
    df_rough = df_rough[~(np.isnan(df_rough['jj_cast'])) & ~(np.isnan(df_rough['ii_cast']))]
    
    bad_casts = np.asarray([val for val in info_df_rough.index if val not in df_rough['cid'].unique().astype('int64')])
    
    for bad in bad_casts:
        
        info_df_rough = info_df_rough.drop(bad)
        
        
    min_z = df_rough.groupby(['cid'])['z'].min().to_frame()
    
    min_z.index = min_z.index.astype('int64')
    
    
    for cid in info_df_rough.index:
        
        min_z.loc[cid, 'h'] = -h[info_df_rough.loc[cid,'jj_cast'].astype('int64'),info_df_rough.loc[cid,'ii_cast'].astype('int64')]
        
        
    for cid in min_z.index:

        if (min_z.loc[cid,'z'] - min_z.loc[cid, 'h'] > -depth_threshold*min_z.loc[cid, 'h']):
            
            info_df = info_df_rough.drop(cid)
            
            
    bad_casts = np.asarray([val for val in df_rough['cid'].unique().astype('int64') if val not in info_df_rough.index])   
         
    for bad in bad_casts:
        
        df = df_rough.drop(df_rough.loc[df['cid'] == bad].index) #replaced with reassign instead of inplace...see if this helps
        
    # gotta figure out the spatial averaging thing.........


    surf_casts_array = vfun.assignSurfaceToCasts(info_df, jjj_dict['All Segments'], iii_dict['All Segments'])
        
    return info_df, df, surf_casts_array, jjj_dict['All Segments'], iii_dict['All Segments']




def apply_bio_casts(Ldir, info_df, df, surf_casts_array, jjj, iii, bvn):
    
    dt = pd.Timestamp('2017-01-01 01:30:00')
    fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
    
    G, S, T = zrfun.get_basic_info(fn_his)
    land_mask = G['mask_rho']
    #Lon = G['lon_rho'][0,:]
    #Lat = G['lat_rho'][:,0]
    # plon,plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])
    z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
    # dz = np.diff(z_w_grid,axis=0)
    # dv = dz*G['DX']*G['DY']
    #h = G['h']
    
    surf_casts_array_full = np.empty(np.shape(land_mask))
    surf_casts_array_full.fill(np.nan)
    
    surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array)
    
    surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array)
    
    for vn in bvn:
        
        temp = np.empty(np.shape(z_rho_grid))
        
        if vn == 'NO3':
        
        
    # stopped 8/25/2023 - almost got this
    
    
    return bio_obs


# DM above ^^^^^^^








        
def salish_fields(V, vn, G):
    """
    Modify biogeochemical fields in the Salish Sea, for initial conditions.
    """
    x = [-125.5, -123.5, -121.9, -121.9]
    y = [50.4, 46.8, 46.8, 50.4]
    p = np.ones((len(x),2))
    p[:,0] = x
    p[:,1] = y
    P = mpath.Path(p)
    lon = G['lon_rho']
    lat = G['lat_rho']
    Rlon = lon.flatten()
    Rlat = lat.flatten()
    R = np.ones((len(Rlon),2))
    R[:,0] = Rlon
    R[:,1] = Rlat
    RR = P.contains_points(R) # boolean
    RRm = RR.reshape(lon.shape)
    # print(RRm.shape)
    # print(RRm.size)
    # print(RRm.sum())
    T, N, M, L = V.shape
    for tt in range(T):
        for nn in range(N):
            lay = V[tt, nn, :, :].squeeze()
            if vn == 'NO3':
                lay[RRm] = 27.0
                V[tt, nn, :, :] = lay
            elif vn == 'oxygen':
                lay[RRm] = 219.0
                V[tt, nn, :, :] = lay
            elif vn == 'alkalinity':
                lay[RRm] = 2077.0
                V[tt, nn, :, :] = lay
            elif vn == 'TIC':
                lay[RRm] = 2037.0
                V[tt, nn, :, :] = lay
            else:
                pass
        
    return V

def create_bio_var(salt, vn):
    if verbose:
        print('  -- adding ' + vn)
    if vn == 'NO3':
        # Salinity vs. NO3 [uM], Ryan McCabe 8/2015
        # NO3 = mm*salt + bb
        mm = 0*salt
        bb = 0*salt
        ind = (salt < 31.898)
        mm[ind] = 0
        bb[ind] = 0
        ind = ((salt >= 31.898) & (salt < 33.791))
        mm[ind] = 16.3958
        bb[ind] = -522.989
        ind = ((salt >= 33.791) & (salt < 34.202))
        mm[ind] = 29.6973
        bb[ind] = -972.4545
        ind = ((salt >= 34.202) & (salt < 34.482))
        mm[ind] = 8.0773
        bb[ind] = -233.0007
        ind = ((salt >= 34.482) & (salt < 35))
        mm[ind] = -28.6251
        bb[ind] = 1032.5686
        NO3 = mm*salt + bb
        # Set maximum NO3 to 45 microMolar (found at ~800m depth), based on
        # evidence from historical NO3 data in NODC World Ocean Database.
        NO3[NO3 > 45] = 45
        # Ensure that there are no negative values.
        NO3[NO3 < 0] = 0
        return NO3
    elif vn == 'NH4':
        NH4 = 0 * salt
        return NH4
    elif vn == 'oxygen':
        if np.nanmax(salt) > 36:
            print('Salt out of range for oxgen regression')
        # Salinity vs. oxygen [uM], Ryan McCabe 8/2015
        # oxygen = mm*salt + bb
        mm = 0*salt
        bb = 0*salt
        ind = (salt < 32.167)
        mm[ind] = 0
        bb[ind] = 300
        ind = ((salt >= 32.167) & (salt < 33.849))
        mm[ind] = -113.9481
        bb[ind] = 3965.3897
        ind = ((salt >= 33.849) & (salt < 34.131))
        mm[ind] = -278.3006
        bb[ind] = 9528.5742
        ind = ((salt >= 34.131) & (salt < 34.29))
        mm[ind] = -127.2707
        bb[ind] = 4373.7895
        ind = ((salt >= 34.29) & (salt < 34.478))
        mm[ind] = 34.7556
        bb[ind] = -1182.0779
        ind = ((salt >= 34.478) & (salt < 35))
        mm[ind] = 401.7916
        bb[ind] = -13836.8132
        oxygen = mm*salt + bb
        # Limit values.
        oxygen[oxygen > 450] = 450
        oxygen[oxygen < 0] = 0
        return oxygen
    elif vn == 'TIC':
        if np.nanmax(salt) > 36:
            print('Salt out of range for TIC regression')
        # Salinity vs. TIC [uM]
        # TIC = mm*salt + bb
        mm = 0*salt
        bb = 0*salt
        ind = (salt < 31.887)
        mm[ind] = 27.7967
        bb[ind] = 1112.2027
        ind = ((salt >= 31.887) & (salt < 33.926))
        mm[ind] = 147.002
        bb[ind] = -2688.8534
        ind = ((salt >= 33.926) & (salt < 34.197))
        mm[ind] = 352.9123
        bb[ind] = -9674.5448
        ind = ((salt >= 34.197) & (salt < 34.504))
        mm[ind] = 195.638
        bb[ind] = -4296.2223
        ind = ((salt >= 34.504) & (salt < 35))
        mm[ind] = -12.7457
        bb[ind] = 2893.77
        TIC = mm*salt + bb
        return TIC
    elif vn == 'alkalinity':
        mm = 0*salt
        bb = 0*salt
        if np.nanmax(salt) > 36:
            print('Salt out of range for alkalinity regression')
        # Salinity vs. alkalinity [uM]
        # alkalinity = mm*salt + bb
        ind = (salt < 31.477)
        mm[ind] = 37.0543
        bb[ind] = 1031.0726
        ind = ((salt >= 31.477) & (salt < 33.915))
        mm[ind] = 48.5821
        bb[ind] = 668.2143
        ind = ((salt >= 33.915) & (salt < 35))
        mm[ind] = 246.2214
        bb[ind] = -6034.6841
        alkalinity = mm*salt + bb
        return alkalinity
    elif vn == 'phytoplankton':
        phytoplankton = 0.01 + 0*salt
        return phytoplankton
    elif vn == 'chlorophyll':
        chlorophyll = 0.025 + 0*salt
        return chlorophyll
    elif vn == 'zooplankton':
        zooplankton = 0.01 + 0*salt
        return zooplankton
    elif vn == 'SdetritusN':
        SdetritusN = 0 * salt
        return SdetritusN
    elif vn == 'LdetritusN':
        LdetritusN = 0 * salt
        return LdetritusN
    elif vn == 'SdetritusC':
        SdetritusC = 0 * salt
        return SdetritusC
    elif vn == 'LdetritusC':
        LdetritusC = 0 * salt
        return LdetritusC
        
        