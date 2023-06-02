"""
Functions for DM's VFC method!

Created 2023/03/23.

"""

import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from lo_tools import Lfun, zfun, zrfun
from lo_tools import extract_argfun as exfun
import cast_functions as cfun
from lo_tools import plotting_functions as pfun
import tef_fun as tfun
import pickle

from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import KDTree

import itertools



def extractCastsPerSegments(Ldir, fn, cast_start, cast_end, cast_year_str, fn_mon_str, fn_year_str, segment):
    """
    
    Extracts casts in designated segmented for designated time in LO history file; uses obs locations for designated period.
        
    Inputs:
        - LDir dictionary
        - fn: history file to extract from
        - cast_start: datetime to bound casts (using obs locations)
        - cast_end: datetime to bound casts (using obs locations)
        - cast_year_str: where to get location from in obs data (year)
        - fn_mon_str: month of fn history file used (***need to generalize)
        - fn_year_str: year of fn history file used (***need to generalize)
        - segment string or list of strings
        
    """    
    #grid info
    G, S, T = zrfun.get_basic_info(fn)
    Lon = G['lon_rho'][0,:]
    Lat = G['lat_rho'][:,0]
    
    # get segment info
    vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
    v_df = pd.read_pickle(vol_dir / 'volumes.p')
    j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
    i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
    seg_list = list(v_df.index)
    
    out_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' /
        (Ldir['source'] + '_' + Ldir['otype'] + '_' + cast_year_str))
    Lfun.make_dir(out_dir, clean=False)
    
    info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + cast_year_str + '.p')
    
    ii= 0
    
    for seg_name in seg_list:
        
        if segment in seg_name:
                
            jjj = j_dict[seg_name]
            iii = i_dict[seg_name]
            
            info_df = pd.read_pickle(info_fn)
            
            N = len(info_df.index)
            Nproc = Ldir['Nproc']
            proc_list = []
            
            for cid in info_df.index:
                
                if info_df.loc[cid,'time'] >= cast_start and info_df.loc[cid,'time'] <= cast_end:
            
                    lon = info_df.loc[cid, 'lon']
                    lat = info_df.loc[cid, 'lat']
                    
                    ix = zfun.find_nearest_ind(Lon, lon)
                    iy = zfun.find_nearest_ind(Lat, lat)
                    
                    
                    if (ix in iii) and (iy in jjj):
                    
                        out_fn = out_dir / (str(int(cid)) + '_' +segment + '_'+str(cast_start.month)+'-'+str(cast_end.month)+'_'+cast_year_str+'_' + fn_mon_str + '_' +fn_year_str+'.nc')
                        
                        # check on which bio variables to get
                        if ii == 0:
                            ds = xr.open_dataset(fn)
                            if 'NH4' in ds.data_vars:
                                npzd = 'new'
                            elif 'NO3' in ds.data_vars:
                                npzd = 'old'
                            else:
                                npzd = 'none'
                            ds.close()
                        
                        print('Get ' + out_fn.name)
                        sys.stdout.flush()
                        
                        
                        # Nproc controls how many subprocesses we allow to stack up
                        # before we require them all to finish.
                        cmd_list = ['python','cast_worker.py',
                        '-out_fn',str(out_fn),
                        '-fn',str(fn),
                        '-lon',str(lon),
                        '-lat',str(lat),
                        '-npzd',npzd]
                        proc = Po(cmd_list, stdout=Pi, stderr=Pi)
                        proc_list.append(proc)
                        # run a collection of processes
                        if ((np.mod(ii,Nproc) == 0) and (ii > 0)) or (ii == N-1) or (Ldir['testing'] and (ii > 3)):
                            for proc in proc_list:
                                if Ldir['testing']:
                                    print('executing proc.communicate()')
                                stdout, stderr = proc.communicate()
                                if len(stdout) > 0:
                                    print('\n' + ' sdtout '.center(60,'-'))
                                    print(stdout.decode())
                                if len(stderr) > 0:
                                    print('\n' + ' stderr '.center(60,'-'))
                                    print(stderr.decode())
                            proc_list = []
                        # ======================================
                        ii += 1


def getAllSegmentIndices(info_fn, seg_list, j_dict, i_dict):
    
    """
    Assigns indices from grid to segments for ease of use.
    
    Inputs:
        - info_fn
        - seg_list
        - j_dict
        - i_dict
        
    Returns
        - jjj_dict
        - jjj_all
        - iii_dict
        - iii_all
        
    """
    
    jjj_all = []
    iii_all = []
    
    jjj_dict = {}
    iii_dict = {}
    
   # info_df = pd.read_pickle(info_fn)

    for seg_name in seg_list:
        
        jjj_temp = j_dict[seg_name]
        iii_temp = i_dict[seg_name]
        
        jjj_all.extend(jjj_temp[:])
        iii_all.extend(iii_temp[:])
        
        jjj_dict[seg_name] = jjj_temp
        iii_dict[seg_name] = iii_temp
    
    jjj_dict['all'] = np.asarray(jjj_all)
    iii_dict['all'] = np.asarray(iii_all)
    
    return jjj_dict, iii_dict
    

def assignSurfaceToCasts(Ldir, info_fn, cast_start, cast_end, Lon, Lat, jjj, iii, land_mask):
    """
    
    Assigns surface domain indices to casts using KDTree algorithm. Currently takes a list of segments or default NONE 
    to use every segment.
        
    Inputs:
        - LDir dictionary
        - info_fn (where casts are coming from, a pickle file usually from obs usually); currently just one at a time***
        - cast_start datetime to bound casts (using obs locations)
        - cast_end datetime to bound casts (using obs locations)
        - Lon field from LO grid info
        - Lat field from LO grid info
        - segment string (just one in loop usually)
        
    Returns:
        - masked array with partitioned domain by cast
        - jj_cast (j indices of casts)
        - ii_cast (i indices of casts)
        
    ***should make based on processed casts instead
    
    """
    
    info_df = pd.read_pickle(info_fn)
        
    jj_cast = []
    ii_cast = []
    
   # jj_cast_temp = []
    #ii_cast_temp  = []
            
    for cid in info_df.index:
        
        if info_df.loc[cid,'time'] >= cast_start and info_df.loc[cid,'time'] <= cast_end:
        
            lon = info_df.loc[cid, 'lon']
            lat = info_df.loc[cid, 'lat']
            
            jj = zfun.find_nearest_ind(Lat, lat)
            ii = zfun.find_nearest_ind(Lon, lon)
            
            if (ii in iii) and (jj in jjj):
                
                if land_mask[jj,ii] == 1:
                
                    jj_cast.append(jj)
                    ii_cast.append(ii)
                
    xx = np.arange(min(iii), max(iii)+1)
    yy = np.arange(min(jjj), max(jjj)+1)
    x, y = np.meshgrid(xx, yy)
            
    a = np.full([len(yy),len(xx)], -99)
    a[jjj-min(jjj),iii-min(iii)] = -1
    a = np.ma.masked_array(a,a==-99)

    b = a.copy()
    
    for n in range(len(jj_cast)):
        b[jj_cast[n]-min(jjj), ii_cast[n]-min(iii)] = n
        
    c = b.copy()
    c = np.ma.masked_array(c,c==-1)
        
    xy_water = np.array((x[~a.mask],y[~a.mask])).T

    xy_casts = np.array((x[~c.mask],y[~c.mask])).T

    tree = KDTree(xy_casts)

    tree_query = tree.query(xy_water)[1]

    surf_casts_array = a.copy()
    
    surf_casts_array[~a.mask] = b[~c.mask][tree_query]
    
    return surf_casts_array, jj_cast, ii_cast




def getLOSubVolThick(fn_his, jjj, iii, var, threshold_val):
    """
    
    Gets LO subthreshold thickness and volume from specified LO_output.
    
    ***I want to add more here, like bottom area.***
        
    Inputs:
        - fn_his: LO history file to use (one at a time for now)
        - jjj: array of j indices in segment domain
        - iii: array of i indices in segment domain
        - var: variable to which to apply thresholding
        - threshold_val: value below which var is considered subthreshold (i.e., hypoxia)
        
    Returns:
        - sub_vol_sum: sum of sub-theshold volume (one value) [m^3]
        - sub_thick_sum: array of sub-threshold thickness throughout domain [m]
        - var_array: for use in matching to casts
    
    """
    
    G, S, T = zrfun.get_basic_info(fn_his)
    z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
    dz = np.diff(z_w_grid,axis=0)
    dv = dz*G['DX']*G['DY']
    
   # dv_sliced = dv[:,min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    #dz_sliced = dz[:,min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    dv_sliced = dv[:,jjj,iii]
    
    dz_sliced = dz[:,jjj,iii]

    ds_his = xr.open_dataset(fn_his)
    
    if var =='oxygen':
        
        var_array = (ds_his.oxygen.squeeze()*32/1000).to_numpy() #molar mass of O2 ###double check this
        
    else:
        
        var_array = (ds_his[var].squeeze()).to_numpy() #implement other var conversions if necessary??? Parker has dict for this
    
   # var_array = var_array[:,min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    var_array = var_array[:,jjj,iii]
    
    dv_sub = dv_sliced.copy()
    
    dv_sub[var_array > threshold_val] = 0
    
   # dv_sub[np.isnan(var_array)] = np.nan
        
    sub_vol_sum = np.nansum(dv_sub)
        
    dz_sub = dz_sliced.copy()
    
    dz_sub[var_array > threshold_val] = 0
    
    #dz_sub[np.isnan(var_array)] = np.nan
    
    sub_thick_sum = np.nansum(dz_sub, axis=0) #units of m...how thick hypoxic column is...depth agnostic
    
    #sub_thick_sum[np.isnan(var_array[0,:,:])] = np.nan
    
    return var_array, sub_vol_sum, sub_thick_sum


def getCastsAttrs(in_dir,fn_list):
    """
    
    Gets attributes from individual casts.
    
    ***Can add more.
        
    Inputs:
        - in_dir: directory to find processed casts
        - fn_list: cast files
        
    Returns:
        - z_rho: depth [m]
        - oxygen: DO [mg/L]
        - sal: salinity [g/kg]***
        - temp: temperature [C]***
        - s0: bottom salinity (bottom sigma layer)
        - s1: surface salinity (top sigma layer)
        - t0: bottom temp
        - t1: surface temp
        
    NEED TO GENERALIZE
    
    """
    # foo = xr.open_dataset(fn_list[0])
    # for vn in foo.data_vars:
    #     print('%14s: %s' % (vn, str(foo[vn].shape)))
    # foo.close()
    
   # s0 = []; s1 = []
    #t0 = []; t1 = []
    #sal = []
    #temp = []
    z_rho = []
    oxygen = []
    
    for fn in fn_list:
        ds = xr.open_dataset(fn)
        z_rho.append(ds.z_rho.values)
        oxygen.append(ds.oxygen.values*32/1000)  # mg/L
        #sal.append(ds.salt.values)
        #temp.append(ds.temp.values)
        #s0.append(ds.salt[0].values)
        #t0.append(ds.temp[0].values)
        #s1.append(ds.salt[-1].values)
        #t1.append(ds.temp[-1].values)
        ds.close()
        
    return z_rho, oxygen #, sal, temp, s0, t0, s1, t1
        
        


def getCastsSubVolThick(in_dir, fn_list, var, threshold_val, fn_his, jjj, iii, ii_cast, surf_casts_array, var_array):
    """
    
    Gets subthreshold volume and thickness from casts (volume-from-casts method).
    
    ***Can add more.
        
    Inputs:
        - ii_cast: i indices to assign cast #s
        - var: variable to work with, from "getCastsAttrs"
        - threshold_val: define threshold for var (under which is considered here)
        - fn_his: LO grid to work with (used both with LO casts and obs)
        - z_rho: depth from "getCastsAttrs"
        - jjj: j indices of domain
        - iii: i indices of domain
        - surf_casts_array: array with casts assigned
        - var_array: from "getLOSubVolThick" - allows precise domain matching between the two methods
        
    Returns:
        - sub_vol: sub-threshold volume (one value)
        - sub_thick: sub-threshold thickness (array of thicknesses in domain)
        
    GENERALIZE
    
    """
    
    z_rho, var_out = getCastsAttrs(in_dir, fn_list) #need to generalize
            
    G, S, T = zrfun.get_basic_info(fn_his)
    z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
    dz = np.diff(z_w_grid,axis=0)
    dv = dz*G['DX']*G['DY']
        
    cast_no = list(map(str, np.arange(len(ii_cast))))

    df0 = pd.DataFrame(var_out)
    df0['cast_no'] = cast_no
    df0 = pd.melt(df0, id_vars=['cast_no'], var_name='sigma_level', value_name = var)

    df1 = pd.DataFrame(z_rho)
    df1['cast_no'] = cast_no
    df1 = pd.melt(df1, id_vars=['cast_no'], var_name='sigma_level', value_name='z_rho')

    df = df0.merge(df1,on=['cast_no','sigma_level'])

    df_sub = df[df[var] < threshold_val]

    sub_casts = df_sub['cast_no'].unique()

    sub_casts = [int(l) for l in sub_casts]

    sub_casts_array = surf_casts_array.copy()

    sub_casts_array = [[ele if ele in sub_casts else -99 for ele in line] for line in sub_casts_array]

    sub_casts_array = np.array(sub_casts_array)

    sub_casts_array =np.ma.masked_array(sub_casts_array,sub_casts_array==-99)

    max_z_sub = []

    sub_vol = 0

    sub_thick_array = np.empty(np.shape(z_rho_grid))
    sub_thick_array.fill(0)

    for n in range(len(sub_casts)):
        df_temp = df_sub[df_sub['cast_no']==str(sub_casts[n])]
        max_z_sub = df_temp['z_rho'].max()
        idx_sub = np.where(sub_casts_array == sub_casts[n])
        sub_array = np.empty(np.shape(z_rho_grid))
        sub_array.fill(0)
        
        for nn in range(len(idx_sub[0])):
            jjj_sub = idx_sub[0][nn] + min(jjj)
            iii_sub = idx_sub[1][nn] + min(iii)
            zzz_sub= np.where(z_rho_grid[:,jjj_sub,iii_sub] <= max_z_sub)
            if zzz_sub:
                sub_array[zzz_sub,jjj_sub,iii_sub] = dv[zzz_sub,jjj_sub,iii_sub]
                sub_thick_array[zzz_sub,jjj_sub,iii_sub] = dz[zzz_sub,jjj_sub,iii_sub]
                        
        sub_vol = sub_vol + np.sum(sub_array)
        
    sub_thick = np.sum(sub_thick_array, axis=0)
    
    sub_thick = sub_thick[min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    sub_thick[np.isnan(var_array[0,:,:])] = np.nan
        
    return sub_vol, sub_thick      
        
        