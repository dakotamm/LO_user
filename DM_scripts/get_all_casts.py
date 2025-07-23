"""

run get_all_casts -gtx cas6_v0_live -test True

"""

import pandas as pd

from lo_tools import Lfun, zfun
from lo_tools import extract_argfun as exfun
from lo_tools import plotting_functions as pfun
from lo_tools import forcing_argfun as ffun
import pickle

from time import time as Time

import matplotlib.pyplot as plt


import numpy as np

import VFC_functions_temp3 as vfun

from pathlib import PosixPath


# %%

#Ldir = ffun.intro() # this handles the argument passing

Ldir = exfun.intro() # this handles the argument passing


# %%

year_list = np.arange(1930,2023)

source_list = ['dfo1', 'ecology', 'nceiSalish']

otype_list = ['ctd','bottle']

dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

vol_dir, v_df, j_dict, i_dict, all_seg_list = vfun.getSegmentInfo(Ldir)

# %%

if Ldir['testing']:
    
    source_list = ['ecology']
    
    otype_list = ['ctd']
    
    year_list = ['2017', '2018']

# %%

segments = 'regions'

jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices(segments, j_dict, i_dict)

if Ldir['testing']:

    seg_list = ['Main Basin']
    
# %%

id_vars = ['cid','cruise','time','lat','lon','name','z','source','type','year']

casts_df = pd.DataFrame()

for source in source_list:
    
    for otype in otype_list:
        
        for year in year_list:
            
            info_fn_in = Ldir['LOo'] / 'obs' / source / otype / ('info_' + str(year) + '.p')
            
            fn_in = Ldir['LOo'] / 'obs' / source / otype / (str(year) + '.p')    
            
            if info_fn_in.exists() & fn_in.exists():
            
                if casts_df.empty:
                                
                    df_temp = pd.read_pickle(fn_in)
                    
                    df_temp['source'] = source
                    
                    df_temp['type'] = otype
                    
                    df_temp['year'] = year
                    
                    if (source == 'ecology') & (otype == 'bottle'):
                        
                        df_temp['CT'] =  np.nan
                        
                        df_temp['SA'] =  np.nan
                        
                        df_temp['DO (uM)'] =  np.nan
                                                
                    value_vars = [x for x in df_temp.columns.tolist() if x not in id_vars]
                    
                    df_temp = pd.melt(df_temp, id_vars=id_vars, value_vars=value_vars, var_name ='var', value_name='val')
                    
                    min_z = df_temp.groupby(['cid','var'], as_index=False)[['cid','var','z']].min()
                    
                    df_temp = pd.merge(min_z, df_temp, on=['cid','var','z'], how='inner')
                    
                    df_temp = df_temp.drop_duplicates(subset=['cid','var'], ignore_index=True)
                                        
                    casts_df = df_temp.copy(deep=True)
                                                            
                else:
                    
                    df_temp = pd.read_pickle(fn_in)
                                        
                    df_temp['source'] = source
                    
                    df_temp['type'] = otype
                    
                    df_temp['year'] = year
                            
                    if (source == 'ecology') & (otype == 'bottle'):
                        
                        df_temp['CT'] = np.nan
                        
                        df_temp['SA'] =  np.nan
                        
                        df_temp['DO (uM)'] =  np.nan
                                            
                    df_temp['cid'] = df_temp['cid'] + casts_df['cid'].max() + 1
                        
                    value_vars = [x for x in df_temp.columns.tolist() if x not in id_vars]
                                        
                    df_temp = pd.melt(df_temp, id_vars=id_vars, value_vars=value_vars, var_name ='var', value_name='val')
                    
                    min_z = df_temp.groupby(['cid','var'], as_index=False)[['cid','var','z']].min()
                    
                    df_temp = pd.merge(min_z, df_temp, on=['cid','var','z'], how='inner')
                    
                    df_temp = df_temp.drop_duplicates(subset=['cid','var'], ignore_index=True)
                    
                    casts_df = pd.concat([casts_df, df_temp])
                    
casts_df = casts_df.reset_index(drop=True)

casts_df['month'] = casts_df['time'].dt.month
                   
                                    
                
# %%

casts_df['ix'] = 0

casts_df['iy'] = 0

casts_df['segment'] = 'None'

casts_df['ii_cast'] = np.nan

casts_df['jj_cast'] = np.nan

for idx in casts_df.index:

    casts_df.loc[idx,'ix'] = zfun.find_nearest_ind(Lon, casts_df.loc[idx,'lon'])

    casts_df.loc[idx,'iy'] = zfun.find_nearest_ind(Lat, casts_df.loc[idx,'lat'])
    
    if land_mask[casts_df.loc[idx,'iy'], casts_df.loc[idx,'ix']] == 1:
        
        casts_df.loc[idx, 'ii_cast'] = casts_df.loc[idx, 'ix']
        
        casts_df.loc[idx, 'jj_cast'] = casts_df.loc[idx, 'iy']
        
        
casts_df = casts_df[~(np.isnan(casts_df['jj_cast'])) & ~(np.isnan(casts_df['ii_cast']))]
        
casts_df['ij_list'] = casts_df.apply(lambda row: (row['ii_cast'], row['jj_cast']), axis=1)

for seg_name in seg_list:
    
    ij_pair = list(zip(iii_dict[seg_name],jjj_dict[seg_name]))
    
    casts_df.loc[casts_df['ij_list'].isin(ij_pair), 'segment'] = seg_name

# %%

if ~Ldir['testing']:
    
    casts_df.to_pickle('/Users/dakotamascarenas/Desktop/casts_df.p')
    
# %%

