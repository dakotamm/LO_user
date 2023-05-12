"""
Functions for DM's VFC method! REVISED

Created 2023/05/11.

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

# I WANT CONCURRENT BOTTLE HANDLING


def getGridInfo(fn):
    
    G, S, T = zrfun.get_basic_info(fn)
    land_mask = G['mask_rho']
    Lon = G['lon_rho'][0,:]
    Lat = G['lat_rho'][:,0]
    z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
    dz = np.diff(z_w_grid,axis=0)
    dv = dz*G['DX']*G['DY']
    
    return G, S, T, land_mask, Lon, Lat, z_rho_grid, dz, dv

    
    
def getSegmentInfo(Ldir):
    
    vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
    v_df = pd.read_pickle(vol_dir / 'volumes.p')
    j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
    i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
    seg_list = list(v_df.index)
    
    return vol_dir, v_df, j_dict, i_dict, seg_list



def defineSegmentIndices(seg_str_list, seg_list_build, j_dict, i_dict):
    
    """
    Assigns indices from grid to segments for ease of use; this currently keeps all existing segments in dict that are called as parts of new segments.
    
    CAN OPTIONALLY NOT DO THIS AND JUST USE J_DICT, I_DICT
    
    Inputs:
        - seg_str: list of strings indicating desired segment names
        - seg_list_build: list of strings indicating segments to build new segments (can be an existing single segment or list of lists of segments; each item in the main list will become a segment)
        - j_dict: big dict from the LO history file
        - i_dict: ^^^
        
    Returns
        - jjj_dict
        - iii_dict
        
    """
    
    jjj_all = []
    iii_all = []
        
    jjj_dict = {}
    iii_dict = {}
    
    for (seg_str, seg_object) in zip(seg_str_list, seg_list_build):
        
        if isinstance(seg_object, list):
                
            for seg_name in seg_object:
                
                jjj_temp = j_dict[seg_name]
                iii_temp = i_dict[seg_name]
                
                jjj_all.extend(jjj_temp[:])
                iii_all.extend(iii_temp[:])
                
                if jjj_dict[seg_name] and iii_dict[seg_name]:
                    
                    continue
                
                else:
                    
                    jjj_dict[seg_name] = jjj_temp
                    iii_dict[seg_name] = iii_temp
                    
            jjj_dict[seg_str] = np.asarray(jjj_all)
            iii_dict[seg_str] = np.asarray(iii_all)
            
        else:
            
            seg_name = seg_object
            
            jjj_temp = j_dict[seg_name]
            iii_temp = i_dict[seg_name]
            
            jjj_dict[seg_str] = j_dict[seg_name]
            iii_dict[seg_str] = i_dict[seg_name]
            
    
    return jjj_dict, iii_dict




def getCleanInfoDF(info_fn, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict):
    
    """
    Cleans the info_df (applicable to both LO casts and obs casts). Need to have defined segments and LO grid components.
    
    Input:
        - info_fn
        - land_mask:  extracted from "G"
        - Lon: extracted from "G"
        - Lat: extracted from "G"
        - seg_list: list of segments (either given or defined using defineSegmentIndices)
        - jjj_dict:
        - iii_dict:
    
    """
    
    info_df = pd.read_pickle(info_fn)
    
    info_df['ix'] = 0

    info_df['iy'] = 0
    
    info_df['segment'] = 'None'
    
    info_df['ii_cast'] = np.nan

    info_df['jj_cast'] = np.nan
    

    for cid in info_df.index:

        info_df.loc[cid,'ix'] = zfun.find_nearest_ind(Lon, info_df.loc[cid,'lon'])

        info_df.loc[cid,'iy'] = zfun.find_nearest_ind(Lat, info_df.loc[cid,'lat'])

        for seg_name in seg_list:
            
            if (info_df.loc[cid,'ix'] in iii_dict[seg_name]) and (info_df.loc[cid, 'iy'] in jjj_dict[seg_name]):
                
                info_df.loc[cid,'segment'] = seg_name
                
            else:
                continue
        
        if land_mask[info_df.loc[cid,'iy'], info_df.loc[cid,'ix']] == 1:
            
            info_df.loc[cid, 'ii_cast'] = info_df.loc[cid, 'ix']
            
            info_df.loc[cid, 'jj_cast'] = info_df.loc[cid, 'iy']
            
        else:
            continue
        
    return info_df
    


def getCleanDF(fn, info_df):
    
    """
    Cleans the df (applicable to obs casts). (MODIFIES DO COLUMN) - works with bottles and ctds
    
    Input:
        - fn
        - info_df: clean info_df
    
    """
    
    df = pd.read_pickle(fn)
    
    df = pd.merge(df, info_df[['ix','iy','ii_cast','jj_cast','segment']], how='left', on=['cid'])

    df['DO_mg_L'] = df['DO (uM)']*32/1000
        
    return df
    
    


def extractLOCasts(Ldir, info_df, fn):
    
    """
    
    Extracts casts in designated segmented for designated time in LO history file; uses obs locations for designated period.
        
    Inputs:
        - LDir dictionary
        - info_df: 
        - fn: history file to be used
        ********** NOT USED BELOW
        - fn_mon_str: month of fn history file used (***need to generalize)
        - fn_year_str: year of fn history file used (***need to generalize)
        - segment string or list of strings (IF WANT TO USE DIFFERENT DATES******)
        
    Outputs:
        - NONE: saves to output directory for future use
        
    """   
    
    LO_casts_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' / (Ldir['source'] + '_' + Ldir['otype'] + '_' + str(Ldir['year']))) #/ (str(info_df['segment'][0]) + '_' + str(info_df['time'].min()) + '_' + str(info_df['time'].max())) )

    Lfun.make_dir(LO_casts_dir, clean=False) # clean=True clobbers the entire directory so be wary kids
    
    ii = 0
    
    N = len(info_df.index)
    Nproc = Ldir['Nproc']
    proc_list = []
    
    for cid in info_df.index:
        
        out_fn = LO_casts_dir / (str(int(cid)) + '.nc')
        
        lon = info_df.loc[cid, 'lon']
        lat = info_df.loc[cid, 'lat']
                        
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
        

    

def assignSurfaceToCasts(info_df, jjj, iii):
    """
    
    Assigns surface domain indices to casts using KDTree algorithm. Currently takes a list of segments or default NONE 
    to use every segment.
        
    Inputs:
        - info_df: cleaned data frame for specific time/space
        - jjj: indices for this segment
        - iii: indices for this segment
        
    Returns:
        - masked array with partitioned domain by cast
        
    ***should make based on processed casts instead
    
    """
                
    xx = np.arange(min(iii), max(iii)+1)
    yy = np.arange(min(jjj), max(jjj)+1)
    x, y = np.meshgrid(xx, yy)
            
    a = np.full([len(yy),len(xx)], -99)
    a[jjj-min(jjj),iii-min(iii)] = -1
    a = np.ma.masked_array(a,a==-99)

    b = a.copy()
    
    for cid in info_df.index:
        b[int(info_df.loc[cid, 'jj_cast'])-min(jjj), int(info_df.loc[cid, 'ii_cast'])-min(iii)] = cid
        
    c = b.copy()
    c = np.ma.masked_array(c,c==-1)
        
    xy_water = np.array((x[~a.mask],y[~a.mask])).T

    xy_casts = np.array((x[~c.mask],y[~c.mask])).T

    tree = KDTree(xy_casts)

    tree_query = tree.query(xy_water)[1]

    surf_casts_array = a.copy()
    
    surf_casts_array[~a.mask] = b[~c.mask][tree_query]
    
    return surf_casts_array



def getLOHisSubVolThick(fn_his, jjj, iii, var, threshold_val):
    """
    
    Gets LO subthreshold thickness and volume from specified LO_output.
            
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
    
    dv_sliced = dv[:,min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    dz_sliced = dz[:,min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    ds_his = xr.open_dataset(fn_his)
    
    if var =='oxygen':
        
        var_array = (ds_his.oxygen.squeeze()*32/1000).to_numpy() #molar mass of O2 ###double check this
        
    else:
        
        var_array = (ds_his[var].squeeze()).to_numpy() #implement other var conversions if necessary??? Parker has dict for this
    
    var_array = var_array[:,min(jjj):max(jjj)+1,min(iii):max(iii)+1]
    
    dv_sub = dv_sliced.copy()
    
    dv_sub[var_array > threshold_val] = 0
    
    dv_sub[np.isnan(var_array)] = np.nan
        
    sub_vol_sum = np.nansum(dv_sub)
        
    dz_sub = dz_sliced.copy()
    
    dz_sub[var_array > threshold_val] = 0
    
    dz_sub[np.isnan(var_array)] = np.nan
    
    sub_thick_sum = np.nansum(dz_sub, axis=0) #units of m...how thick hypoxic column is...depth agnostic
    
    sub_thick_sum[np.isnan(var_array[0,:,:])] = np.nan
    
    return var_array, sub_vol_sum, sub_thick_sum




def getLOCastsAttrs(fn):
    """
    
    Gets attributes from individual casts.
    
    ***Can add more.
        
    Inputs:
        - LO_casts_dir: directory to find processed casts
        - fn_list: cast files
        
    Returns:
        - z_rho: depth [m]
        - oxygen: DO [mg/L]
        # - sal: salinity [g/kg]***
        # - temp: temperature [C]***
        # - s0: bottom salinity (bottom sigma layer)
        # - s1: surface salinity (top sigma layer)
        # - t0: bottom temp
        # - t1: surface temp
        
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
        
        


def getLOCastsSubVolThick(Ldir, info_df, var, threshold_val, fn_his, jjj, iii, surf_casts_array, var_array):
    """
    
    Gets subthreshold volume and thickness from LO casts (volume-from-casts method).
    
    ***Can add more.
        
    Inputs:
        - Ldir: big directory situation 
        - info_df: all the cast info (cleaned)
        - var: variable to work with, from "getCastsAttrs"
        - threshold_val: define threshold for var (under which is considered here)
        - fn_his: LO grid to work with
        - z_rho: depth from "getCastsAttrs"
        - jjj: j indices of domain
        - iii: i indices of domain
        - surf_casts_array: array with casts assigned
        - var_array: from "getLOSubVolThick" - allows precise domain matching between the two methods
        
    Returns:
        - sub_vol: sub-threshold volume (one value)
        - sub_thick: sub-threshold thickness (array of thicknesses in domain)
        
    GENERALIZE TO OTHER VARIABLES
    
    """
    
    LO_casts_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' /
        (Ldir['source'] + '_' + Ldir['otype'] + '_' + str(Ldir['year'])) / str(info_df['segment'][0]) + '_' + str(info_df['time'].min()) + '_' + str(info_df['time'].max()) )
    
    df0 = pd.DataFrame()
    
    df0['cid'] = []
            
    df0['z_rho'] = []
    
    df0[var] = []
    
    
    for cid in info_df.index:
        
        df_temp = pd.DataFrame()    
        
        fn = (LO_casts_dir / str(cid) + '.nc')
        
        z_rho, var_out = getLOCastsAttrs(fn) #need to generalize
                
        df_temp['z_rho'] = z_rho
        
        df_temp[var] = var_out
        
        df_temp['cid'] = cid
        
        df0 = pd.concat(df0, df_temp)
        
    G, S, T = zrfun.get_basic_info(fn_his)
    z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
    dz = np.diff(z_w_grid,axis=0)
    dv = dz*G['DX']*G['DY']

    df_sub = df0[df0[var] < threshold_val]

    sub_casts = df_sub['cid']

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




def getOBSCastsSubVolThick(info_df, df, var, threshold_val, z_rho_grid, dv, dz, land_mask, jjj, iii):
    
    """
    
    Gets subthreshold volume and thickness from casts (volume-from-casts method).
            
    Inputs:
        - info_df_use: all the cast info (cleaned)
        - df_use: all the 
        - var: variable to work with, from "getCastsAttrs"
        - threshold_val: define threshold for var (under which is considered here)
        - dv: from grid
        - dz: from grid
        - land_mask: from grid
        - jjj: j indices of domain
        - iii: i indices of domain
        
    Returns:
        - sub_vol: sub-threshold volume (one value)
        - sub_thick: sub-threshold thickness (array of thicknesses in domain)
        
    GENERALIZE TO OTHER VARIABLES
    
    """
    
    
    
    
    sub_vol = 0
    
    sub_thick_array = np.empty(np.shape(z_rho_grid))
    sub_thick_array.fill(0)
    
    
    if df.empty: # if there are no casts in this time period
    
        sub_thick_array.fill(np.nan)
    
        sub_thick = np.sum(sub_thick_array, axis=0)
        
        sub_thick = sub_thick[min(jjj):max(jjj)+1,min(iii):max(iii)+1]
                    
        sub_vol = np.nan
                
        
    else: # if there ARE casts in this time period
            
        df_sub = df[df[var] < threshold_val]
    
        if df_sub.empty: # if there are no subthreshold volumes
        
            sub_thick = np.sum(sub_thick_array, axis=0)
            
            sub_thick[land_mask == 0] = np.nan # ESSENTIALLY - NEED TO REAPPLY LAND MASK FOR PLOTTING
            
            sub_thick = sub_thick[min(jjj):max(jjj)+1, min(iii):max(iii)+1]
            
        
        else: # if there ARE subthreshold volumes
        
            surf_casts_array = assignSurfaceToCasts(info_df, jjj, iii)
        
            info_df_sub = info_df.copy()
        
            for cid in info_df.index:
                
                if ~(cid in df_sub['cid'].unique()):
                    
                    info_df_sub.drop([cid])
            
            sub_casts_array = surf_casts_array.copy()
    
            sub_casts_array = [[ele if ele in df_sub['cid'].unique() else -99 for ele in line] for line in sub_casts_array]
    
            sub_casts_array = np.array(sub_casts_array)
    
            sub_casts_array =np.ma.masked_array(sub_casts_array,sub_casts_array==-99)
            
            max_z_sub = []
            
            for cid in info_df_sub.index:
                
                df_temp = df_sub[df_sub['cid'] == cid]
                
                max_z_sub = df_temp['z'].max()
                
                idx_sub = np.where(sub_casts_array == cid)
                
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
            
            sub_thick[land_mask == 0] = np.nan # ESSENTIALLY - NEED TO REAPPLY LAND MASK FOR PLOTTING
            
            sub_thick = sub_thick[min(jjj):max(jjj)+1,min(iii):max(iii)+1]
            
            
            # land mask match for the volume sum???????

        
    return sub_vol, sub_thick