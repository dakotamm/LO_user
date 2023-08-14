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


# D functions

from dateutil.relativedelta import relativedelta

import VFC_functions as vfun


def setup_CTD_casts(Ldir, source_list=['ecology','dfo1','nceiSalish']):
    
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
        
            info_fn_in =  Ldir['LOo'] / 'obs' / source / 'ctd' / ('info_' + str(year) + '.p')
        
            info_df_temp0 = pd.read_pickle(info_fn_in)
        
            info_df_temp0['time'] = info_df_temp0['time'].astype('datetime64[ns]')
            
            info_df_temp0['source'] = source
            
            info_df_temp1 = info_df_temp0[(info_df_temp0['time'] >= start_dt) & (info_df_temp0['time'] <= end_dt)]
               
            info_df_rough = pd.concat([info_df_rough, info_df_temp1], ignore_index=True)
            
            info_df.index.name = 'cid'
            
            
    for year in year_list:
       
        for source in source_list:
            
            fn_in =  Ldir['LOo'] / 'obs' / source / 'ctd' / (+ str(year) + '.p')
        
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


    surf_casts_array = vfun.assignSurfaceToCasts(info_df, jjj_dict['whole'], iii_dict['whole'])
        
    return info_df, df, surf_casts_array



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
