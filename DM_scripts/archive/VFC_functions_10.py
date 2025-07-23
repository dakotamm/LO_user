"""
Functions for DM's VFC method! REVISED AGAIN

Created 2023/08/08.

"""

import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from lo_tools import Lfun, zfun, zrfun #original LO tools
import extract_argfun_DM as exfun
import cast_functions_DM as cfun
from lo_tools import plotting_functions as pfun #original LO tools
import tef_fun_DM as tfun
import pickle

from time import time as Time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import KDTree

import itertools

import copy

from pathlib import PosixPath

from datetime import timedelta

# TO DO:
    # plotting functions
    # tidally-averaged grid, or take history file when obs cast was taken? - space inefficient!
    # speed limitations - this is clunky
    # error catching
    # fix extract cast - maybe check to see if the cast is already there? probably inefficient to have duplicate casts in different folders; revert to original cast extraction idea
    # underprediction or overprediction based on if cutoff is inclusive or not
    
    



def get_his_fn_from_dt(Ldir, dt): # RIPPED FROM CFUN to support perigee usage
    # This creates the Path of a history file from its datetime
    tt0 = Time()
    if dt.hour == 0:
        # perfect restart does not write the 0001 file
        dt = dt - timedelta(days=1)
        his_num = '0025'
    else:
        his_num = ('0000' + str(dt.hour + 1))[-4:]
    date_string = dt.strftime(Ldir['ds_fmt'])
    
    if Ldir['lo_env'] == 'dm_mac':
        
        fn = Ldir['roms_out'] / Ldir['gtagex'] / ('f' + date_string) / ('ocean_his_' + his_num + '.nc')
        
    elif Ldir['lo_env'] == 'dm_perigee':
        
        fn = (PosixPath('/data1/parker/LO_roms/') / Ldir['gtagex'] / ('f' + date_string) / ('ocean_his_' + his_num + '.nc'))
    
    print('get_his_fn_from_dt = %d sec' % (int(Time()-tt0)))
              
    return fn
    

def getGridInfo(fn):
    
    tt0 = Time()
    G, S, T = zrfun.get_basic_info(fn)
    land_mask = G['mask_rho']
    Lon = G['lon_rho'][0,:]
    Lat = G['lat_rho'][:,0]
    plon,plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])
    z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
    dz = np.diff(z_w_grid,axis=0)
    dv = dz*G['DX']*G['DY']
    h = G['h']
    
    print('getGridInfo = %d sec' % (int(Time()-tt0)))
    
    return G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h

    
    
def getSegmentInfo(Ldir):
    
    tt0 = Time()
    vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
    v_df = pd.read_pickle(vol_dir / 'volumes.p')
    j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
    i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
    seg_list = list(v_df.index)
    
    print('getSegmentInfo = %d sec' % (int(Time()-tt0)))
    
    return vol_dir, v_df, j_dict, i_dict, seg_list


def buildInfoDF(Ldir, info_fn_in, info_fn):
    
    tt0 = Time()
    
    info_df_temp = pd.read_pickle(info_fn_in)
        
    info_df_temp['source'] = Ldir['source']
    
    info_df_temp['type'] = Ldir['otype']
    
    info_df_temp['time'] = info_df_temp['time'].astype('datetime64[ns]')
        
    if info_fn.exists():
        
        info_df = pd.read_pickle(info_fn)
        
        if info_df[(info_df['type'] == Ldir['otype']) & (info_df['source'] == Ldir['source'])].empty:
    
            for col in info_df.columns:
                
                if col not in info_df_temp.columns:
                    
                    info_df_temp[col] = np.nan
            
            for col in info_df_temp.columns:
                
                if col not in info_df.columns:
                    
                    info_df[col] = np.nan
                    
            info_df = pd.concat([info_df, info_df_temp], ignore_index=True)
            
            info_df.index.name = 'cid'
            
            info_df.to_pickle(info_fn)
            
        else:
            print('Data already added to info_df.')
            
        
    else:
        
        info_df = info_df_temp.copy(deep=True)
        
        info_df.to_pickle(info_fn)
        
        
    print('buildInfoDF = %d sec' % (int(Time()-tt0)))
        
        
    return info_df


def buildDF(Ldir, fn_in, fn, info_df):
    
    tt0 = Time()
    
    df_temp = pd.read_pickle(fn_in)
        
    df_temp['source'] = Ldir['source']
    
    df_temp['type'] = Ldir['otype']
    
    df_temp['time'] = df_temp['time'].astype('datetime64[ns]')
    
    if fn.exists():
        
        df = pd.read_pickle(fn)
        
        if df[(df['type'] == Ldir['otype']) & (df['source'] == Ldir['source'])].empty:

            for col in df.columns:
                    
                if col not in df_temp.columns:
                        
                    df_temp[col] = np.nan
                    
            for col in df_temp.columns:
                
                if col not in df.columns:
                    
                    df[col] = np.nan
                        
            df_temp['cid'] = df_temp['cid'] + info_df[(info_df['type'] == Ldir['otype']) & (info_df['source'] == Ldir['source'])].index.min()
            
            df = pd.concat([df, df_temp])
            
            df.to_pickle(fn)
        
        else:
            print('Data already added to df.')
            
    else:
        
        df = df_temp.copy(deep=True)
        
        df.to_pickle(fn)
        
    
    print('buildDF = %d sec' % (int(Time()-tt0)))
        
        
    return df


def defineSegmentIndices(seg_str_list, j_dict, i_dict, seg_list_build=['']):
    
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
    
    tt0 = Time()
    
    if seg_str_list == 'all':
        
        seg_str_list = ['A1','A2','A3', 'H1','H2','H3','H4','H5','H6','H7','H8', 'M1','M2','M3','M4','M5','M6', 'S1','S2','S3','S4', 'T1', 'T2', 'W1','W2','W3','W4', 'G1','G2','G3','G4','G5','G6','J1','J2','J3','J4']
        
        jjj_dict = j_dict
        
        iii_dict = i_dict
    
    elif seg_str_list == 'basins':
        
        seg_str_list = ['Admiralty Inlet', 'Strait of Georgia','Hood Canal','Strait of Juan de Fuca', 'Main Basin', 'South Sound', 'Tacoma Narrows', 'Whidbey Basin']
        
        seg_list_build = [['A1','A2','A3'], ['G1','G2','G3','G4','G5','G6'], ['H1','H2','H3','H4','H5','H6','H7','H8'], ['J1','J2','J3','J4'], ['M1','M2','M3','M4','M5','M6'], ['S1','S2','S3','S4'], ['T1', 'T2'], ['W1','W2','W3','W4']]
    
    elif seg_str_list == 'sound_straits':
    
        seg_str_list = ['Puget Sound', 'Strait of Georgia','Strait of Juan de Fuca']
        
        seg_list_build = [['A1','A2','A3', 'H1','H2','H3','H4','H5','H6','H7','H8', 'M1','M2','M3','M4','M5','M6', 'S1','S2','S3','S4', 'T1', 'T2', 'W1','W2','W3','W4'], ['G1','G2','G3','G4','G5','G6'], ['J1','J2','J3','J4']]
     
    elif seg_str_list == 'whole':
        
        seg_str_list = ['All Segments']
        
        seg_list_build = [['A1','A2','A3', 'H1','H2','H3','H4','H5','H6','H7','H8', 'M1','M2','M3','M4','M5','M6', 'S1','S2','S3','S4', 'T1', 'T2', 'W1','W2','W3','W4', 'G1','G2','G3','G4','G5','G6','J1','J2','J3','J4']]
        
    
    if seg_list_build:
        
        jjj_dict = {}
        
        iii_dict = {}
        
        for (seg_str, seg_object) in zip(seg_str_list, seg_list_build):
            
            if isinstance(seg_object, list):
                
                jjj_all = []
                iii_all = []
                    
                for seg_name in seg_object:
                    
                    jjj_temp = j_dict[seg_name]
                    iii_temp = i_dict[seg_name]
                    
                    jjj_all.extend(jjj_temp[:])
                    iii_all.extend(iii_temp[:])
                        
                jjj_dict[seg_str] = np.asarray(jjj_all)
                iii_dict[seg_str] = np.asarray(iii_all)
                
            else:
                
                seg_name = seg_object
                
                jjj_dict[seg_str] = j_dict[seg_name]
                iii_dict[seg_str] = i_dict[seg_name]
            
    print('defineSegmentIndices = %d sec' % (int(Time()-tt0)))
    
    return jjj_dict, iii_dict, seg_str_list




def getCleanDataFrames(info_fn, fn, h, land_mask, Lon, Lat, seg_list, jjj_dict, iii_dict, var):
    
    """
    Cleans the info_df (applicable to both LO casts and obs casts). Need to have defined segments and LO grid components. Cleans the df (applicable to obs casts). (MODIFIES DO COLUMN) - works with bottles and ctds
    
    Input:
        - info_fn
        - fn
        - land_mask:  extracted from "G"
        - Lon: extracted from "G"
        - Lat: extracted from "G"
        - seg_list: list of segments (either given or defined using defineSegmentIndices)
        - jjj_dict:
        - iii_dict:
    
    """
    
    tt0=Time()
    
    depth_threshold = 0.2 # percentage of bathymetry the cast can be from the bottom to be accepted
    
    info_df = pd.read_pickle(info_fn)
    
    info_df['ix'] = 0

    info_df['iy'] = 0
    
    info_df['segment'] = 'None'
    
    info_df['ii_cast'] = np.nan

    info_df['jj_cast'] = np.nan
    

    for cid in info_df.index:

        info_df.loc[cid,'ix'] = zfun.find_nearest_ind(Lon, info_df.loc[cid,'lon'])

        info_df.loc[cid,'iy'] = zfun.find_nearest_ind(Lat, info_df.loc[cid,'lat'])
        
        if land_mask[info_df.loc[cid,'iy'], info_df.loc[cid,'ix']] == 1:
            
            info_df.loc[cid, 'ii_cast'] = info_df.loc[cid, 'ix']
            
            info_df.loc[cid, 'jj_cast'] = info_df.loc[cid, 'iy']
            
    for seg_name in seg_list:
        
        ij_pair = list(zip(iii_dict[seg_name],jjj_dict[seg_name]))
        
        for cid in info_df.index:        
              
            pair = (info_df.loc[cid,'ix'].tolist(), info_df.loc[cid,'iy'].tolist())
            
            if pair in ij_pair:
                info_df.loc[cid,'segment'] = seg_name
            
    info_df = info_df[~(np.isnan(info_df['jj_cast'])) & ~(np.isnan(info_df['ii_cast']))]
    
    # DUPLICATE HANDLING!!!! average the properties????
    
    df = pd.read_pickle(fn)
    
    df = pd.merge(df, info_df[['ix','iy','ii_cast','jj_cast','segment']], how='left', on=['cid'])
    
    df = df[~(np.isnan(df['jj_cast'])) & ~(np.isnan(df['ii_cast']))]
    
    if var == 'DO_mg_L':

        df[var] = df['DO (uM)']*32/1000
        
    elif var == 'T_deg_C':
        
        df[var] = df['CT']
        
    elif var == 'S_g_kg':
        
        df[var] = df['SA']
    
    
    
    df = df[~np.isnan(df[var])]
    
    bad_casts = np.asarray([val for val in info_df.index if val not in df['cid'].unique().astype('int64')])
    
    for bad in bad_casts:
        
        info_df = info_df.drop(bad)
        
    
    min_z = df.groupby(['cid'])['z'].min().to_frame()
    
    min_z.index = min_z.index.astype('int64')
    
    
    for cid in info_df.index:
        
        min_z.loc[cid, 'h'] = -h[info_df.loc[cid,'jj_cast'].astype('int64'),info_df.loc[cid,'ii_cast'].astype('int64')]
        
        
    for cid in min_z.index:

        if (min_z.loc[cid,'z'] - min_z.loc[cid, 'h'] > -depth_threshold*min_z.loc[cid, 'h']):
            
            info_df = info_df.drop(cid)
            
            
    bad_casts = np.asarray([val for val in df['cid'].unique().astype('int64') if val not in info_df.index])   
         
    for bad in bad_casts:
        
        df = df.drop(df.loc[df['cid'] == bad].index) #replaced with reassign instead of inplace...see if this helps
            
    
    print('getCleanDataFrames = %d sec' % (int(Time()-tt0)))
    
    return info_df, df
    
    


def extractLOCasts(Ldir, info_df_use, fn_his):
    
    """
    NEED TO SPIFF THIS UP!!!!!!!!!!
    
    Extracts casts in designated segment for designated time in LO history file; uses obs locations for designated period.
        
    Inputs:
        - LDir dictionary
        - info_df: 
        - fn: history file to be used
        - CLUNKY NEEDS REVISE ***
        
    Outputs:
        - NONE: saves to output directory for future use
        
    """   
    tt0 = Time()
    
    if not info_df_use.empty:
        
        LO_casts_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' / (str(Ldir['year'])) / (str(info_df_use['segment'].iloc[0]) + '_' + str(info_df_use['time'].dt.date.min()) + '_' + str(info_df_use['time'].dt.date.max()) ) )
        
        # if Ldir['lo_env'] == 'dm_mac':
                        
        # elif Ldir['lo_env'] == 'dm_perigee':
            
        #     info_fn_in = PosixPath('/data1/parker/LO_output/obs/' + Ldir['source'] + '/' + Ldir['otype'] + '/info_' + str(Ldir['year']) + '.p')
        
        if not LO_casts_dir.exists():
                
            Lfun.make_dir(LO_casts_dir, clean=False) # clean=True clobbers the entire directory so be wary kids
                
            ii = 0
            
            N = len(info_df_use.index)
            Nproc = Ldir['Nproc']
            proc_list = []
            
            for cid in info_df_use.index:
                
                out_fn = LO_casts_dir / (str(int(cid)) + '.nc')
                
                if out_fn.exists():
                    
                    continue
                
                else:
                    
                    lon = info_df_use.loc[cid, 'lon']
                    lat = info_df_use.loc[cid, 'lat']
                                
                    # check on which bio variables to get
                    if ii == 0:
                        ds = xr.open_dataset(fn_his)
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
                    cmd_list = ['python','cast_worker_DM.py',
                    '-out_fn',str(out_fn),
                    '-fn',str(fn_his),
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
        else:
            print('LO cast extraction exists')
        
        print('LO casts extracted')
        
    else:
        print('no casts to extract in LO')
        
    print('extractLOCasts = %d sec' % (int(Time()-tt0)))
        

    

def assignSurfaceToCasts(info_df_use, jjj, iii):
    """
    
    Assigns surface domain indices to casts using KDTree algorithm. Currently takes a list of segments or default NONE 
    to use every segment.
        
    Inputs:
        - info_df: cleaned data frame for specific time/space
        - jjj: indices for this segment
        - iii: indices for this segment
        
    Returns:
        - masked array with partitioned domain by cast
            
    """
    
    tt0 = Time()
    
    xx = np.arange(min(iii), max(iii)+1)
    yy = np.arange(min(jjj), max(jjj)+1)
                
    if info_df_use.empty:
        
        surf_casts_array = np.empty([len(yy),len(xx)])
        surf_casts_array.fill(np.nan)
        
        print('no casts for surface assignment')
                
    else:
        
        x, y = np.meshgrid(xx, yy)
                
        a = np.full([len(yy),len(xx)], -99)
        a[jjj-min(jjj),iii-min(iii)] = -1
        a = np.ma.masked_array(a,a==-99)
    
        b = copy.deepcopy(a)
        
        for cid in info_df_use.index:
            b[int(info_df_use.loc[cid, 'jj_cast'])-min(jjj), int(info_df_use.loc[cid, 'ii_cast'])-min(iii)] = cid # use string to see if it helps plotting?
            
        c = copy.deepcopy(b)
        c = np.ma.masked_array(c,c==-1)
            
        xy_water = np.array((x[~a.mask],y[~a.mask])).T
    
        xy_casts = np.array((x[~c.mask],y[~c.mask])).T
    
        tree = KDTree(xy_casts)
    
        tree_query = tree.query(xy_water)[1]
    
        surf_casts_array = copy.deepcopy(a)
        
        surf_casts_array[~a.mask] = b[~c.mask][tree_query]
        
        print('surface assigned to casts')
            
    print('assignSurfaceToCasts = %d sec' % (int(Time()-tt0)))
    
    return surf_casts_array



def getLOHisSubVolThick(dv, dz, fn_his, jjj, iii, var, threshold_val):
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
    
    tt0 = Time()
    
    print('LO His available')
    
    dv_sliced = dv[:, jjj, iii].copy()
    
    dz_sliced = dz[:, jjj, iii].copy()
    
    ds_his = xr.open_dataset(fn_his)
    
    if var =='DO_mg_L':
        
        var_array = (ds_his.oxygen.squeeze()*32/1000).to_numpy() #molar mass of O2 ###double check this
        
    else:
        
        var_array = (ds_his[var].squeeze()).to_numpy() #implement other var conversions if necessary??? Parker has dict for this
    
    var_array = var_array[:, jjj, iii]
    
    dv_sub = dv_sliced.copy()
    
    dv_sub[var_array > threshold_val] = 0
    
    sub_vol_sum = np.sum(dv_sub)
        
    dz_sub = dz_sliced.copy()
    
    dz_sub[var_array > threshold_val] = 0
    
    sub_thick_sum = np.sum(dz_sub, axis=0) #units of m...how thick hypoxic column is...depth agnostic
    
    print('getLOHisSubVolThick = %d sec' % (int(Time()-tt0)))
    
    return sub_vol_sum, sub_thick_sum




def getLOCastsAttrs(fn, var):
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
    
    tt0 = Time()
    
    ds = xr.open_dataset(fn)
    z_rho = ds.z_rho.values
    
    if var == 'DO_mg_L':
        
        oxygen = ds.oxygen.values*32/1000  # mg/L
        
        return z_rho, oxygen
    
        ds.close()
    
    # elif var == 'T_deg_C':
        
    #     temperature = ds.temp.values # deg C
        
    #     return z_rho, temp
    
    # elif var == 'S_g_kg-1':
        
    #     salinity = ds.sal.values
        
    #     return z_rho, sal
    
    print('getLOCastsAttrs within getLOCastsSubVolThick = %d sec' % (int(Time()-tt0)))
    
    # return z_rho, oxygen
        
        


def getLOCastsSubVolThick(Ldir, info_df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array):
    """
    
    Gets subthreshold volume and thickness from LO casts (volume-from-casts method).
    
    ***Can add more.
        
    Inputs:
        - Ldir: big directory situation 
        - info_df_use: all the cast info (cleaned)
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
    
    tt0 = Time()
    
    print('LO His available for LO VFC attempt')
    
    surf_casts_array_full = np.empty(np.shape(land_mask))
    surf_casts_array_full.fill(np.nan)
        
    surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array)

    sub_thick_array = np.empty(np.shape(z_rho_grid))
    
    sub_casts_array_full = np.empty(np.shape(surf_casts_array_full))
    
    sub_casts_array_full.fill(np.nan)

    
        
    if info_df_use.empty: #if no casts in this time period and region
        
        sub_thick_array.fill(np.nan)
    
        sub_thick_temp = np.sum(sub_thick_array, axis=0)
        
        sub_thick = sub_thick_temp[jjj,iii]
                            
        sub_vol = np.nan
                
        sub_casts_array = sub_casts_array_full[jjj,iii]
        
        print('no LO casts')

        
    else: #if there are casts in this time and region
    
        domain_flag = False
        
        num_casts = len(info_df_use.index)
        
        if num_casts == 1:
            
            domain_flag = True
            
            print('too few LO casts')
        
        else:
     
            for cid in info_df_use.index:

                test = np.where(surf_casts_array == cid)
                
                if num_casts == 2:
                    
                    if (48.48498053545606 < info_df_use.loc[cid,'lat'] < 48.68777542575437) & (-123.58171407533055 < info_df_use.loc[cid,'lon'] < -123.44409346988729):
                        
                        domain_flag = True
                        
                        print('bad - only two LO casts and in Saanich Inlet')
                    
                    if np.size(test, axis=1) > np.size(jjj)*0.8:
                        
                        domain_flag = True
                        
                        print('bad - only two LO casts and one exceeds 80% of surface')
                else:
                    
                    if (48.48498053545606 < info_df_use.loc[cid,'lat'] < 48.68777542575437) & (-123.58171407533055 < info_df_use.loc[cid,'lon'] < -123.44409346988729):

                        if np.size(test,axis=1) > np.size(jjj)*0.1:
                                            
                            domain_flag = True
                            
                            print('bad - Saanich Inlet LO casts take up more than 10% of surface')
                            
                    if np.size(test,axis=1) > np.size(jjj)*0.5:
                        
                        domain_flag = True
                        
                        print('bad - one LO cast takes up more than 50% of domain')
        
        if domain_flag: #if too few casts for domain
        
            sub_thick_array.fill(np.nan)
        
            sub_thick_temp = np.sum(sub_thick_array, axis=0)
            
            sub_thick = sub_thick_temp[jjj,iii]
                                
            sub_vol = np.nan
                        
            sub_casts_array = sub_casts_array_full[jjj,iii]
            
            #print('not enough spatial coverage LO casts')
            
            
        else: #if enough spatial coverage
        
            print('sufficient spatial coverage LO casts')
            
    
            LO_casts_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' / (str(Ldir['year'])) / (str(info_df_use['segment'].iloc[0]) + '_' + str(info_df_use['time'].dt.date.min()) + '_' + str(info_df_use['time'].dt.date.max()) ) )
        
            df0 = pd.DataFrame()
        
            df0['cid'] = []
                
            df0['z_rho'] = []
        
            df0[var] = []
        
            for cid in info_df_use.index:
                
                df_temp = pd.DataFrame()    
                
                fn = (LO_casts_dir) / (str(cid) + '.nc')
                
                if fn.exists(): 
                
                    z_rho, var_out = getLOCastsAttrs(fn) #need to generalize
                            
                    df_temp['z_rho'] = z_rho
                    
                    df_temp[var] = var_out
                    
                    df_temp['cid'] = cid
                    
                    df0 = pd.concat([df0, df_temp])
                    
            df_sub = df0[df0[var] < threshold_val]
            
            
            if df_sub.empty: # if no subthreshold values
                
                sub_vol = 0
                
                sub_thick_array.fill(0)
                
                sub_thick_temp = np.sum(sub_thick_array, axis=0)
                
                sub_thick = sub_thick_temp[jjj,iii]
                                        
                sub_casts_array = sub_casts_array_full[jjj,iii]
                
                print('no subthreshold value LO casts')
    
                
            else: # if subthreshold values!
            
                 print('subthreshold value LO casts exist')
                                       
                 info_df_sub = info_df_use.copy(deep=True)
             
                 for cid in info_df_use.index:
                     
                     if cid not in df_sub['cid'].unique():
                         
                         info_df_sub = info_df_sub.drop(cid)
                 
                 sub_casts_array_temp = copy.deepcopy(surf_casts_array)
         
                 sub_casts_array_temp0 = [[ele if ele in df_sub['cid'].unique() else -99 for ele in line] for line in sub_casts_array_temp]
         
                 sub_casts_array_temp1 = np.array(sub_casts_array_temp0)
         
                 sub_casts_array =np.ma.masked_array(sub_casts_array_temp1,sub_casts_array_temp1==-99)
                 
                 sub_casts_array_full[min(jjj):max(jjj)+1, min(iii):max(iii)+1] = sub_casts_array.copy()
                 
                 sub_array = np.empty(np.shape(z_rho_grid))
                 sub_array.fill(0)
    
                 sub_thick_array.fill(0)
                              
                 
     
                 for cid in info_df_sub.index:
                     
                     df_temp = df_sub[df_sub['cid']==cid]
                     
                     z_rho_array = z_rho_grid[:, int(info_df_use.loc[cid,'jj_cast']), int(info_df_use.loc[cid,'ii_cast'])].copy()
                     
                     n = 0
                     
                     for depth in z_rho_array:
                         
                         if depth not in np.asarray(df_temp['z_rho']):
                             
                             z_rho_array[n] = np.nan
                             
                         n +=1
                             
                     z_rho_array_full = np.repeat(z_rho_array[:,np.newaxis], np.size(z_rho_grid, axis=1), axis=1)
                     
                     z_rho_array_full_3d = np.repeat(z_rho_array_full[:,:,np.newaxis], np.size(z_rho_grid, axis=2), axis=2)
                     
                     sub_casts_array_full_3d = np.repeat(sub_casts_array_full[np.newaxis,:,:], np.size(z_rho_grid, axis=0), axis=0)
                                                       
                     sub_array[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))] = dv[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))].copy()
                     
                     sub_thick_array[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))] = dz[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))].copy()
                              
                     
                        
                 sub_vol = np.sum(sub_array)
                
                 sub_thick_temp = np.sum(sub_thick_array, axis=0)
                
                 sub_thick = sub_thick_temp[jjj,iii]
                
                 sub_casts_array = sub_casts_array_full[jjj,iii]
    
    print('getLOCastsSubVolThick = %d sec' % (int(Time()-tt0)))

             

             
    return sub_vol, sub_thick, sub_casts_array




def getOBSCastsSubVolThick(info_df_use, df_use, var, threshold_val, z_rho_grid, land_mask, dv, dz, jjj, iii, surf_casts_array):
    
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
        - sub_array: where the casts are beneath a threshold in surface domain
        
    GENERALIZE TO OTHER VARIABLES
    
    """

    tt0 = Time()
    
    surf_casts_array_full = np.empty(np.shape(land_mask))
    surf_casts_array_full.fill(np.nan)
    
    surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array)

    sub_thick_array = np.empty(np.shape(z_rho_grid))
    
    sub_casts_array_full = np.empty(np.shape(surf_casts_array_full))
    
    sub_casts_array_full.fill(np.nan)
    
    
    if info_df_use.empty: # if there are no casts in this time period
    
        sub_thick_array.fill(np.nan)
    
        sub_thick_temp = np.sum(sub_thick_array, axis=0)
        
        sub_thick = sub_thick_temp[jjj,iii]
                            
        sub_vol = np.nan
                
        sub_casts_array = sub_casts_array_full[jjj,iii]
        
        print('no obs casts')
                        
        
    else: # if there ARE casts in this time period
            
        domain_flag = False
        
        num_casts = len(info_df_use.index)
        
        if num_casts == 1:
            
            domain_flag = True
            
            print('too few obs casts')
        
        else:
     
            for cid in info_df_use.index:

                test = np.where(surf_casts_array == cid)
                
                if num_casts == 2:
                    
                    if (48.48498053545606 < info_df_use.loc[cid,'lat'] < 48.68777542575437) & (-123.58171407533055 < info_df_use.loc[cid,'lon'] < -123.44409346988729):
                        
                        domain_flag = True
                        
                        print('bad - only two obs casts and in Saanich Inlet')
                    
                    if np.size(test, axis=1) > np.size(jjj)*0.8:
                        
                        domain_flag = True
                        
                        print('bad - only two obs casts and one exceeds 80% of surface')
                else:
                    
                    if (48.48498053545606 < info_df_use.loc[cid,'lat'] < 48.68777542575437) & (-123.58171407533055 < info_df_use.loc[cid,'lon'] < -123.44409346988729):

                        if np.size(test,axis=1) > np.size(jjj)*0.1:
                                            
                            domain_flag = True
                            
                            print('bad - Saanich Inlet obs casts take up more than 10% of surface')
                            
                    if np.size(test,axis=1) > np.size(jjj)*0.5:
                        
                        domain_flag = True
                        
                        print('bad - one obs cast takes up more than 50% of domain')
                

                        
        if domain_flag: #if too few casts for domain
        
            sub_thick_array.fill(np.nan)
        
            sub_thick_temp = np.sum(sub_thick_array, axis=0)
            
            sub_thick = sub_thick_temp[jjj,iii]
                                
            sub_vol = np.nan
                        
            sub_casts_array = sub_casts_array_full[jjj,iii]
            
            # print('not enough spatial coverage obs')
            
            
        # NEED CONDITION TO ACCOUNT FOR IF THERE IS NO DO DATA, perhaps separate than the below condition handling
        
        
        else: #if enough spatial coverage
        
            print('sufficient spatial coverage obs casts')
        
            df_sub = df_use[df_use[var] < threshold_val]
        
            if df_sub.empty: # if there are no subthreshold volumes
            
                sub_vol = 0
                
                sub_thick_array.fill(0)
                
                sub_thick_temp = np.sum(sub_thick_array, axis=0)
                
                sub_thick = sub_thick_temp[jjj,iii]
                                        
                sub_casts_array = sub_casts_array_full[jjj,iii]
                
                print('no subthreshold value obs casts')
                        
            else: # if there ARE subthreshold volumes
            
                print('subthreshold value obs casts exist')
                    
                info_df_sub = info_df_use.copy(deep=True)
            
                for cid in info_df_use.index:
                    
                    if cid not in df_sub['cid'].unique():
                        
                        info_df_sub = info_df_sub.drop(cid) 
                
                sub_casts_array_temp = copy.deepcopy(surf_casts_array)
        
                sub_casts_array_temp0 = [[ele if ele in df_sub['cid'].unique() else -99 for ele in line] for line in sub_casts_array_temp]
        
                sub_casts_array_temp1 = np.array(sub_casts_array_temp0)
        
                sub_casts_array =np.ma.masked_array(sub_casts_array_temp1,sub_casts_array_temp1==-99)
                
                sub_casts_array_full[min(jjj):max(jjj)+1, min(iii):max(iii)+1] = sub_casts_array.copy()
                
                sub_array = np.empty(np.shape(z_rho_grid))
                sub_array.fill(0)
    
                sub_thick_array.fill(0)
                             
                
    
                for cid in info_df_sub.index:
                    
                    #df_temp = df_sub[df_sub['cid']==cid]
                    
                    df_temp = df_use[df_use['cid'] == cid].sort_values('z').reset_index()
                    
                    cross_below_to_above = []
                    
                    cross_above_to_below = []
                    
                    for idx in df_temp.index:
                        
                        if idx < df_temp.index.max():
                            
                            if (df_temp.loc[idx, 'DO_mg_L'] < threshold_val) & (df_temp.loc[idx+1, 'DO_mg_L'] > threshold_val):
                                
                                cross_point = ((df_temp.loc[idx+1, 'z'] - df_temp.loc[idx,'z'])/2) + df_temp.loc[idx,'z']
                                
                                cross_below_to_above.append(cross_point)
                                
                            elif (df_temp.loc[idx, 'DO_mg_L'] > threshold_val) & (df_temp.loc[idx+1, 'DO_mg_L'] < threshold_val):
                                
                                cross_point = ((df_temp.loc[idx+1, 'z'] - df_temp.loc[idx,'z'])/2) + df_temp.loc[idx,'z']
                                
                                cross_above_to_below.append(cross_point)
                                
                    
                    z_rho_array = z_rho_grid[:, int(info_df_use.loc[cid,'jj_cast']), int(info_df_use.loc[cid, 'ii_cast'])].copy()
                                                                        
                    
                    if (len(cross_below_to_above) > 0) & (len(cross_above_to_below) > 0):
                        
                        z_rho_array_temp = np.empty(np.shape(z_rho_array))
                        
                        z_rho_array_temp.fill(np.nan)
                        
                        
                        if len(cross_below_to_above) > len(cross_above_to_below): #always going to be either equal or offset by 1 only
                                                    
                            for n in range(len(cross_below_to_above)):
                                
                                if n == 0:
                                
                                    z_rho_array_temp[z_rho_array < cross_below_to_above[n]] = z_rho_array[z_rho_array < cross_below_to_above[n]]
                                    
                                elif (n != 0) & (n!= len(cross_below_to_above)-1):
                                                                                                    
                                    z_rho_array_temp[(z_rho_array >= cross_above_to_below[n-1]) & (z_rho_array < cross_below_to_above[n])] = z_rho_array[(z_rho_array >= cross_above_to_below[n-1]) & (z_rho_array < cross_below_to_above[n])]
                                                        
                        elif len(cross_below_to_above) < len(cross_above_to_below):
                            
                            for n in range(len(cross_above_to_below)):
                                
                                if n != len(cross_above_to_below)-1:
                                
                                    z_rho_array_temp[(z_rho_array >= cross_above_to_below[n]) & (z_rho_array < cross_below_to_above[n])] = z_rho_array[(z_rho_array >= cross_above_to_below[n]) & (z_rho_array < cross_below_to_above[n])]
                        
                                else:
                                    
                                    z_rho_array_temp[z_rho_array >= cross_above_to_below[n]] = z_rho_array[z_rho_array >= cross_above_to_below[n]] 
                                
                        else:
                            
                            if cross_below_to_above[0] < cross_above_to_below[0]:
                                
                                for n in range(len(cross_below_to_above)):
                                    
                                    if len(cross_below_to_above) > 1:
                                    
                                        if n==0:
                                            
                                            z_rho_array_temp[z_rho_array < cross_below_to_above[n]] = z_rho_array[z_rho_array < cross_below_to_above[n]]
                                            
                                        elif (n!=0) & (n != len(cross_above_to_below)-1):
                                            
                                            z_rho_array_temp[(z_rho_array >= cross_above_to_below[n-1]) & (z_rho_array < cross_below_to_above[n])] = z_rho_array[(z_rho_array >= cross_above_to_below[n-1]) & (z_rho_array < cross_below_to_above[n])]
        
                                        else:
                                            
                                            z_rho_array_temp[z_rho_array >= cross_above_to_below[n]] = z_rho_array[z_rho_array >= cross_above_to_below[n]]
                                    
                                    else:
                                        
                                            z_rho_array_temp[z_rho_array < cross_below_to_above[n]] = z_rho_array[z_rho_array < cross_below_to_above[n]]
                                            
                                            z_rho_array_temp[z_rho_array >= cross_above_to_below[n]] = z_rho_array[z_rho_array >= cross_above_to_below[n]]
    
    
                            else:
                                
                                for n in range(len(cross_above_to_below)):
                                                                                                    
                                    z_rho_array_temp[(z_rho_array >= cross_above_to_below[n]) & (z_rho_array < cross_below_to_above[n])] = z_rho_array[(z_rho_array >= cross_above_to_below[n]) & (z_rho_array < cross_below_to_above[n])]
                            
                            
                        z_rho_array[np.isnan(z_rho_array_temp)] = np.nan
                        
                    
                    elif (len(cross_below_to_above) > 0) & (len(cross_above_to_below) == 0):
                        
                        z_rho_array_temp = np.empty(np.shape(z_rho_array))
                        
                        z_rho_array_temp.fill(np.nan)
                        
                        for n in range(len(cross_below_to_above)):
                        
                            z_rho_array_temp[z_rho_array < cross_below_to_above[n]] = z_rho_array[z_rho_array < cross_below_to_above[n]]
                            
                        z_rho_array[np.isnan(z_rho_array_temp)] = np.nan
                        
                    elif (len(cross_below_to_above) == 0) & (len(cross_above_to_below) > 0):
                        
                        z_rho_array_temp = np.empty(np.shape(z_rho_array))
                        
                        z_rho_array_temp.fill(np.nan)
                        
                        for n in range(len(cross_above_to_below)):
                            
                            z_rho_array_temp[z_rho_array >= cross_above_to_below[n]] = z_rho_array[z_rho_array >= cross_above_to_below[n]]
                            
                        z_rho_array[np.isnan(z_rho_array_temp)] = np.nan
                        

                    z_rho_array_full = np.repeat(z_rho_array[:,np.newaxis], np.size(z_rho_grid, axis=1), axis=1)
                    
                    z_rho_array_full_3d = np.repeat(z_rho_array_full[:,:,np.newaxis], np.size(z_rho_grid, axis=2), axis=2)
                                    
                    sub_casts_array_full_3d = np.repeat(sub_casts_array_full[np.newaxis,:,:], np.size(z_rho_grid, axis=0), axis=0)
                                                      
                    sub_array[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))] = dv[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))].copy()
                    
                    sub_thick_array[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))] = dz[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))].copy()



                sub_vol = np.sum(sub_array)
                
                sub_thick_temp = np.sum(sub_thick_array, axis=0)
                
                sub_thick = sub_thick_temp[jjj,iii]
                
                sub_casts_array = sub_casts_array_full[jjj,iii]

    print('getOBSCastsSubVolThick = %d sec' % (int(Time()-tt0)))

    
    return sub_vol, sub_thick, sub_casts_array



def getOBSCastsWtdAvgBelow(info_df_use, df_use, var, threshold_pct, z_rho_grid, land_mask, dv, dz, h, jjj, iii, surf_casts_array):
    
    """
    THRESHOLD DEPTH IS NOW PERCENTAGE
    
    """
    
    tt0 = Time()

    surf_casts_array_full = np.empty(np.shape(land_mask))
    surf_casts_array_full.fill(np.nan)
    
    surf_casts_array_full[min(jjj):max(jjj)+1,min(iii):max(iii)+1] = copy.deepcopy(surf_casts_array)

    # sub_thick_array = np.empty(np.shape(z_rho_grid))
    
    sub_casts_array_full = np.empty(np.shape(surf_casts_array_full))
    
    sub_casts_array_full.fill(np.nan)
    
    
    if info_df_use.empty: # if there are no casts in this time period
    
        # sub_thick_array.fill(np.nan)
    
        # sub_thick_temp = np.sum(sub_thick_array, axis=0)
        
        # sub_thick = sub_thick_temp[jjj,iii]
                            
        sub_wtd_avg = np.nan
                
        # sub_casts_array = sub_casts_array_full[jjj,iii]
        
        print('no obs casts for wtd avg')
                        
        
    else: # if casts in time period
            
          
        
        df_max_z = df_use[['cid','z']].groupby('cid').min().reset_index()
        
        df_max_z['max_z'] = df_max_z['z']
                          
        df_temp = pd.merge(df_use, df_max_z[['cid','max_z']], how='left', on = 'cid')
        
        df_sub = df_temp[df_temp['z'] < (1-threshold_pct)*df_temp['max_z']]
        
        ###
        df_wtd_avg = df_sub[['cid', var]].groupby('cid').mean()
        
        df_wtd_avg['vol_km_3'] = np.nan
                
        if df_sub.empty: # if it isn't deep enough
        
            if var == 'DO_mg_L':
        
                df_wtd_avg['DO_wtd_mg_L'] = np.nan
            
            elif var == 'T_deg_C':
                
                df_wtd_avg['T_wtd_deg_C'] = np.nan
                
            elif var == 'S_g_kg':
            
                df_wtd_avg['S_wtd_g_kg'] = np.nan
                

            sub_wtd_avg = np.nan
            
            print('not deep enough obs casts for wtd avg')
        
        
        else: # if it is deep enough
        
            print('obs casts deep enough for wtd avg')
                
            info_df_sub = info_df_use.copy(deep=True)
        
            for cid in info_df_use.index:
                
                if cid not in df_sub['cid'].unique():
                    
                    info_df_sub = info_df_sub.drop(cid) 
            
            sub_casts_array_temp = copy.deepcopy(surf_casts_array)
    
            sub_casts_array_temp0 = [[ele if ele in df_sub['cid'].unique() else -99 for ele in line] for line in sub_casts_array_temp]
    
            sub_casts_array_temp1 = np.array(sub_casts_array_temp0)
    
            sub_casts_array =np.ma.masked_array(sub_casts_array_temp1,sub_casts_array_temp1==-99)
            
            sub_casts_array_full[min(jjj):max(jjj)+1, min(iii):max(iii)+1] = sub_casts_array.copy()
            
            # sub_array = np.empty(np.shape(z_rho_grid))
            # sub_array.fill(0)

            #sub_thick_array.fill(0)
            
            
            for cid in info_df_use.index:
                
                threshold_depth = (1-threshold_pct)*-h[info_df_use.loc[cid, 'jj_cast'].astype('int64'), info_df_use.loc[cid,'ii_cast'].astype('int64')]
                
                sub_array = np.empty(np.shape(z_rho_grid))
                sub_array.fill(0)
                
                z_rho_array = z_rho_grid[:, int(info_df_use.loc[cid,'jj_cast']), int(info_df_use.loc[cid, 'ii_cast'])].copy()
            
                z_rho_array[z_rho_array > threshold_depth] = np.nan
                
                # gotta write the linear interpolation bit... 
            
                z_rho_array_full = np.repeat(z_rho_array[:,np.newaxis], np.size(z_rho_grid, axis=1), axis=1)
                
                z_rho_array_full_3d = np.repeat(z_rho_array_full[:,:,np.newaxis], np.size(z_rho_grid, axis=2), axis=2)
                            
                sub_casts_array_full_3d = np.repeat(sub_casts_array_full[np.newaxis,:,:], np.size(z_rho_grid, axis=0), axis=0)
                                              
                sub_array[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))] = dv[(sub_casts_array_full_3d == cid) & ~(np.isnan(z_rho_array_full_3d))].copy()
            
                df_wtd_avg.loc[df_wtd_avg.index == cid, 'vol_m_3'] = np.sum(sub_array)
            

            if var == 'DO_mg_L':
        
                df_wtd_avg['DO_wtd_mg_L'] = df_wtd_avg[var]*df_wtd_avg['vol_m_3']
                
                sub_wtd_avg = np.nansum(df_wtd_avg['DO_wtd_mg_L'])/np.nansum(df_wtd_avg['vol_m_3'])
            
            elif var == 'T_deg_C':
                
                df_wtd_avg['T_wtd_deg_C'] = df_wtd_avg[var]*df_wtd_avg['vol_m_3']
                
                sub_wtd_avg = np.nansum(df_wtd_avg['T_wtd_deg_C'])/np.nansum(df_wtd_avg['vol_m_3'])
                
            elif var == 'S_g_kg':
            
                df_wtd_avg['S_wtd_g_kg'] = df_wtd_avg[var]*df_wtd_avg['vol_m_3']
                
                sub_wtd_avg = np.nansum(df_wtd_avg['S_wtd_g_kg'])/np.nansum(df_wtd_avg['vol_m_3'])                       
            

    print('getOBSCastsWtdAvgBelow = %d sec' % (int(Time()-tt0)))
    
    return sub_wtd_avg



def getOBSAvgBelow(info_df_use, df_use, var, threshold_pct):
    
    """
    
    """
    
    tt0 = Time()
    
    if info_df_use.empty: # if there are no casts in this time period
    
        # sub_thick_array.fill(np.nan)
    
        # sub_thick_temp = np.sum(sub_thick_array, axis=0)
        
        # sub_thick = sub_thick_temp[jjj,iii]
                            
        sub_avg = np.nan
                
        # sub_casts_array = sub_casts_array_full[jjj,iii]
        
        print('no obs casts for sub-depth avg')
        
        
    else: # if casts in time period
            
        df_max_z = df_use[['cid','z']].groupby('cid').min().reset_index()
        
        df_max_z['max_z'] = df_max_z['z']
                          
        df_temp = pd.merge(df_use, df_max_z[['cid','max_z']], how='left', on = 'cid')
        
        # should I use the absolute depth? should be close but for next time
        
        df_sub = df_temp[df_temp['z'] < (1-threshold_pct)*df_temp['max_z']]
        
        if df_sub.empty: # if it isn't deep enough
        
            print('not deep enough obs casts for sub-depth avg')
        
            sub_avg = np.nan
                    
        else:  #if it is deep enough
        
            print('obs casts deep enough for sub-depth avg')
        
            sub_avg = df_sub[var].mean()
            
            
            
    print('getOBSAvgBelow = %d sec' % (int(Time()-tt0)))
            
    return sub_avg