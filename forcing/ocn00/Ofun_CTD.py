"""
Functions for adding CTD data to an extrapolation.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from datetime import datetime, timedelta

import Ofun
from lo_tools import zfun, Lfun, zrfun


# D BELOW >>>>


import VFC_functions as vfun

import copy





def apply_CTD_casts(Ldir, info_df, df, surf_casts_array, jjj_dict, iii_dict, seg_list):
    
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
    
    T_array = np.empty(np.shape(z_rho_grid))
    T_array.fill(np.nan)

    S_array = np.empty(np.shape(z_rho_grid))
    S_array.fill(np.nan)
    
    for seg_name in seg_list:
    
        jjj = jjj_dict[seg_name]
        
        iii = iii_dict[seg_name]
    
        surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array[seg_name])
    
        surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array[seg_name])
        
        info_df_use = info_df[info_df['segment'] == seg_name]
        
        df_use = df[df['segment'] == seg_name]
        
        if not info_df_use.empty:
    
            for cid in info_df_use.index:
                        
                df_temp = df_use[df_use['cid'] == cid]
                
                cast_idx = np.where(surf_casts_array_full == cid)
                
                if len(cast_idx[0]) > 0:
                        
                    for n in range(len(cast_idx[0])):
                                                    
                        for cell in range(len(z_rho_grid[:,cast_idx[0][n],cast_idx[1][n]])):
                            
                            near_depth_idx = zfun.find_nearest_ind(df_temp['z'].to_numpy(), z_rho_grid[cell, cast_idx[0][n], cast_idx[1][n]])
                            
                            T_array[cell, cast_idx[0][n], cast_idx[1][n]] = df_temp['CT'].to_numpy()[near_depth_idx]
                            
                            S_array[cell, cast_idx[0][n], cast_idx[1][n]] = df_temp['SA'].to_numpy()[near_depth_idx]
        
    return T_array, S_array, z_rho_grid


# ^^^^ DM ABOVE








def get_casts(Ldir):
    
    year = 2017
    
    month = 1
    
    # ^^^ double check with OG
    
    # +++ load ecology CTD cast data +++
    dir0 = Ldir['parent'] / 'ptools_data' / 'ecology'
    # load processed station info and data
    sta_df = pd.read_pickle(dir0 / 'sta_df.p')
    # add Canadian data
    dir1 = Ldir['parent'] / 'ptools_data' / 'canada'
    # load processed station info and data
    sta_df_ca = pd.read_pickle(dir1 / 'sta_df.p')
    sta_df = pd.concat((sta_df, sta_df_ca), sort=False)
    Casts = pd.read_pickle(dir0 / ('Casts_' + str(year) + '.p'))
    Casts_ca = pd.read_pickle(dir1 / ('Casts_' + str(year) + '.p'))
    Casts = pd.concat((Casts, Casts_ca), sort=False)

    # limit the stations used, if desired
    sta_list = [s for s in sta_df.index]# if ('WPA' not in s) and ('GYS' not in s)]
    # keep only certain columns
    sta_df = sta_df.loc[sta_list,['Max_Depth', 'Latitude', 'Longitude']]
    #

    # start a dict to store one cast per station (if it has data in the year)
    Cast_dict = dict()

    for station in sta_list:
        casts = Casts[Casts['Station'] == station]
        casts = casts.set_index('Date')
        casts = casts.loc[:,['Salinity', 'Temperature','Z']] # keep only selected columns
        # identify a single cast by its date
        alldates = casts.index
        castdates = alldates.unique() # a short list of unique dates (1 per cast)
    
        # get the CTD cast data for this station, in the nearest month
        cdv = castdates.month.values # all the months with casts
        if len(cdv) > 0:
            # get the cast closest to the selected month
            imo = zfun.find_nearest_ind(cdv, month)
            new_mo = cdv[imo]
            cast = casts[casts.index==castdates[imo]]
            Cast = cast.set_index('Z') # reorganize so that the index is Z
            Cast = Cast.dropna() # clean up
            # store cast in a dict
            Cast_dict[station] = Cast
            # save the month, just so we know
            sta_df.loc[station,'Month'] = new_mo
            print('  - Ofun_CTD.get_casts: including : '
                + station + ' month=' + str(new_mo))
        else:
            print('  - Ofun_CTD.get_casts:' +station + ': no data')
    # Cast_dict.keys() is the "official" list of stations to loop over
    return Cast_dict, sta_df
    
def get_orig(Cast_dict, sta_df, X, Y, fld, lon, lat, zz, vn):
    
    verbose = False
    
    #  make vectors or 1- or 2-column arrays (*) of the good points to feed to cKDTree
    xyorig = np.array((X[~fld.mask],Y[~fld.mask])).T
    fldorig = fld[~fld.mask]

    #========================================================================

    # +++ append good points from CTD data to our arrays (*) +++

    goodcount = 0

    for station in Cast_dict.keys():
    
        Cast = Cast_dict[station]
        cz = Cast.index.values
        izc = zfun.find_nearest_ind(cz, zz)
    
        # only take data from this cast if its bottom depth is at or above
        # the chosen hycom level
        czbot = -sta_df.loc[station,'Max_Depth']
        if czbot <= zz:
            # because we used find_nearest above we should always
            # get data in the steps below
            if vn == 't3d':
                this_fld = Cast.iloc[izc]['Temperature']
            elif vn == 's3d':
                this_fld = Cast.iloc[izc]['Salinity']
            # and store in sta_df (to align with lat, lon)
            sta_df.loc[station,'fld'] = this_fld
            goodcount += 1
        else:
            pass
        
    if goodcount >= 1:
    
        # drop stations that don't have T and s values at this depth
        sta_df = sta_df.dropna()
        # and for later convenience make a new list of stations
        sta_list = list(sta_df.index)
    
        # if we got any good points then append them
        if verbose:
            print('  - Ofun_CTD.get_orig: goodcount = %d, len(sta_df) = %d'
                % (goodcount, len(sta_df)))
    
        # append CTD values to the good points from HYCOM
        x_sta = sta_df['Longitude'].values
        y_sta = sta_df['Latitude'].values
        xx_sta, yy_sta = zfun.ll2xy(x_sta, y_sta, lon.mean(), lat.mean())
        xy_sta = np.stack((xx_sta,yy_sta), axis=1)
        xyorig = np.concatenate((xyorig, xy_sta))
    
        fld_arr = sta_df['fld'].values
        fldorig = np.concatenate((fldorig, np.array(fld_arr,ndmin=1)))
    
    else:
        if verbose:
            print('  - Ofun_CTD.get_orig: No points added')
    
    return xyorig, fldorig
    
def extrap_nearest_to_masked_CTD(X,Y,fld,xyorig=[],fldorig=[],fld0=0):
    
    # first make sure nans are masked
    if np.ma.is_masked(fld) == False:
        fld = np.ma.masked_where(np.isnan(fld), fld)
    
    if fld.all() is np.ma.masked:
        print('  - Ofun_CTD.extrap_nearest_to_masked_CTD: filling with '
            + str(fld0))
        fldf = fld0 * np.ones(fld.data.shape)
        fldd = fldf.data
        Ofun.checknan(fldd)
        return fldd
    else:
        fldf = fld.copy()
        # array of the missing points that we want to fill
        xynew = np.array((X[fld.mask],Y[fld.mask])).T
        # array of indices for points nearest to the missing points
        a = cKDTree(xyorig).query(xynew)
        aa = a[1]

        # use those indices to fill in using the good data
        fldf[fld.mask] = fldorig[aa]
            
        fldd = fldf.data
        Ofun.checknan(fldd)
        return fldd
