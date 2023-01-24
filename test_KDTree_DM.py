"""
Test of nearest-neighbor extrapolation to fill missing (masked) values
on a plaid grid.

Test on mac in ipython:
run test_KDTree_DM -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test True

"""
import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from lo_tools import Lfun, zfun, zrfun
from lo_tools import extract_argfun as exfun
import cast_functions as cfun
import tef_fun as tfun
import pickle

from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import cKDTree

Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

dt = pd.Timestamp('2022-11-30 01:30:00')
fn = cfun.get_his_fn_from_dt(Ldir, dt)

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

info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')

ii= 0

for seg_name in seg_list:
    
    if 'G1' in seg_name:
            
        jjj = j_dict[seg_name]
        iii = i_dict[seg_name]
        
        i_ = np.unique(iii)
        j_ = np.unique(jjj)
        
        m,n = np.meshgrid(i_,j_)
        
        info_df = pd.read_pickle(info_fn)
        
        ix_ = []
        iy_ = []

        for cid in info_df.index:
            
            lon = info_df.loc[cid, 'lon']
            lat = info_df.loc[cid, 'lat']
            
            ix = zfun.find_nearest_ind(Lon, lon)
            iy = zfun.find_nearest_ind(Lat, lat)
            
            if (ix in iii) and (iy in jjj):
                
                ix_.append(ix)
                iy_.append(iy)
                
                
        #query_tree = cKDTree(list(zip(iii,jjj)))
    
        #ref_points = list(zip(ix_,iy_))
        
        casts = np.arange(len(ix_))
        
        fill_value = -99
        
        casts_arr = np.arange(len(j_),len(i_)+1).reshape(len(j_),len(i_))
        
        casts_arr[:] = fill_value
        
        
        
        
        
        
    
        
    
    
    
                
                
                



# # create a data array, with masked areas
# a = np.arange(100).reshape(10,10)
# fill_value=-99
# a[2:4,3:8] = fill_value
# a[7:,7:] = fill_value
# a = ma.masked_array(a,a==fill_value)

# # create axes
# xx = np.linspace(0, 100, a.shape[1])
# yy = np.linspace(0, 10, a.shape[0])
# x, y = np.meshgrid(xx, yy)

# # do the extrapolation
# from scipy.spatial import cKDTree
# xygood = np.array((x[~a.mask],y[~a.mask])).T
# xybad = np.array((x[a.mask],y[a.mask])).T
# b = a.copy()
# b[a.mask] = a[~a.mask][cKDTree(xygood).query(xybad)[1]]
# # asking for [1] gives the index of the nearest good value
# # and [0] would return the distance, I think.

# print(a)

# print(b)

