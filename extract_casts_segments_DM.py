"""
This is code for doing cast extractions.

Refactored 2022_07 to conform to the new cast data format.

DM - modified to create cast field using DFO locations and select LO history files.

Test on mac in ipython:
run extract_casts_segments_DM -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test True

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

Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

#hacky tef segment implementation
# sect_df = tfun.get_sect_df('cas6')

# min_lon = sect_df.at['sog5','x0']
# max_lon = sect_df.at['sog1','x1']
# min_lat = sect_df.at['sog1','y0']
# max_lat = sect_df.at['sog5','y0']

dt = pd.Timestamp('2022-11-30 01:30:00')
fn = cfun.get_his_fn_from_dt(Ldir, dt)

#grid info
G, S, T = zrfun.get_basic_info(fn)
#h = G['h']
#DA = G['DX'] * G['DY']
#DA3 = DA.reshape((1,G['M'],G['L']))
Lon = G['lon_rho'][0,:]
Lat = G['lat_rho'][:,0]

# get segment info
vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
v_df = pd.read_pickle(vol_dir / 'volumes.p')
j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
seg_list = list(v_df.index)

out_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' /
    (Ldir['source'] + '_' + Ldir['otype'] + '_' + year_str))
Lfun.make_dir(out_dir, clean=True)

info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')

ii = 0

for seg_name in seg_list:
    
    if 'G6' in seg_name:
            
        jjj = j_dict[seg_name]
        iii = i_dict[seg_name]
        
        info_df = pd.read_pickle(info_fn)
        
        for cid in info_df.index:
        
            lon = info_df.loc[cid, 'lon']
            lat = info_df.loc[cid, 'lat']
            
            ix = zfun.find_nearest_ind(Lon, lon)
            iy = zfun.find_nearest_ind(Lat, lat)
            
            if (ix in iii) and (iy in jjj):
            
                out_fn = out_dir / (str(int(cid)) + '_sog.nc')
                
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
                cfun.get_cast(out_fn, fn, lon, lat, npzd)
                ii += 1
                
                # if Ldir['testing'] and (ii > 20):
                #     print(ii)
                #     break
            