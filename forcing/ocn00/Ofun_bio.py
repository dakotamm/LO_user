"""
Functions to add biogeochemical fields to a clm file.
"""

import numpy as np
import matplotlib.path as mpath
from lo_tools import Lfun, zfun, zrfun

verbose = False




### D ADDITONS >>>>

from dateutil.relativedelta import relativedelta

import VFC_functions as vfun

import pandas as pd

from datetime import datetime, timedelta

import copy


# def apply_bio_casts(Ldir, info_df, df, surf_casts_array, jjj_dict, iii_dict, seg_list, bvn_list):
    
#     dt = pd.Timestamp('2017-01-01 01:30:00')
#     fn_his = vfun.get_his_fn_from_dt(Ldir, dt)
    
#     G, S, T = zrfun.get_basic_info(fn_his)
#     land_mask = G['mask_rho']
#     #Lon = G['lon_rho'][0,:]
#     #Lat = G['lat_rho'][:,0]
#     # plon,plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])
#     z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
#     # dz = np.diff(z_w_grid,axis=0)
#     # dv = dz*G['DX']*G['DY']
#     #h = G['h']
    
#     surf_casts_array_full = np.empty(np.shape(land_mask))
#     surf_casts_array_full.fill(np.nan)
        
#     bio_obs = {}
    
#     for bvn in bvn_list:
        
#         bio_obs[bvn] = np.empty(np.shape(z_rho_grid))
#         bio_obs[bvn].fill(np.nan)
        
        
#     for seg_name in seg_list:
    
#         jjj = jjj_dict[seg_name]
        
#         iii = iii_dict[seg_name]
    
#         surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array[seg_name])
    
#         surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array[seg_name])
        
#         info_df_use = info_df[info_df['segment'] == seg_name]
        
#         df_use = df[df['segment'] == seg_name]
        
        
#         if not info_df_use.empty:
        
        
#             for cid in info_df_use.index:
        
        
#                 df_temp = df_use[df_use['cid'] == cid]
                
#                 cast_idx = np.where(surf_casts_array_full == cid)
                
        
#                 if len(cast_idx[0]) > 0:
                        
#                     for n in range(len(cast_idx)):
                                                    
#                         for cell in range(len(z_rho_grid[:,cast_idx[0][n],cast_idx[1][n]])):
                            
#                             near_depth_idx = zfun.find_nearest_ind(df_temp['z'].to_numpy(), z_rho_grid[cell, cast_idx[0][n], cast_idx[1][n]])
                            
#                             for bvn in bvn_list:
                                
#                                 if bvn == 'NO3':
                                
#                                     bio_obs[bvn][cell, cast_idx[0][n], cast_idx[1][n]] = df_temp['NO3 (uM)'].to_numpy()[near_depth_idx] + df_temp['NO2 (uM)'].to_numpy()[near_depth_idx]
                                    
#                                 elif bvn == 'NH4':
                                    
#                                     bio_obs[bvn][cell, cast_idx[0][n], cast_idx[1][n]] = df_temp['NH4 (uM)'].to_numpy()[near_depth_idx]
                
#                                 elif bvn == 'oxygen':
                                    
#                                     bio_obs[bvn][cell, cast_idx[0][n], cast_idx[1][n]] = df_temp['DO (uM)'].to_numpy()[near_depth_idx]
                                    
#                                 elif bvn == 'chlorophyll':
                                    
#                                     bio_obs[bvn][cell, cast_idx[0][n], cast_idx[1][n]] = df_temp['Chl (mg m-3)'].to_numpy()[near_depth_idx]
                                
#                                 elif bvn == 'alkalinity':
                                    
#                                     bio_obs[bvn][cell, cast_idx[0][n], cast_idx[1][n]] = df_temp['TA (uM)'].to_numpy()[near_depth_idx]
                                    
#                                 elif bvn == 'TIC':
                                    
#                                     bio_obs[bvn][cell, cast_idx[0][n], cast_idx[1][n]] = df_temp['DIC (uM)'].to_numpy()[near_depth_idx]
            
#     # for bvn in bio_obs_3D.keys():
        
#     #     bio_obs[bvn] = bio_obs_3D[bvn][:,jjj,iii]
    
#     return bio_obs


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
        
        