# calculate hourly volume, hourly salt*volume and EminusP
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from lo_tools import zrfun, Lfun
import pickle
from datetime import datetime, timedelta
from time import time
import sys
import pandas as pd

tt0 = time()

# adapting code from JX LO_user starting 20260305

#
Ldir = Lfun.Lstart(gridname = 'wb1', tag = 'r0', ex_name = 'xn11b')
Ldir['collection_tag'] = 'pc0'
Ldir['ds0'] = '2017.09.05'
Ldir['ds1'] = '2017.09.17'
Ldir['riv'] = 'riv00'
sect_gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']


#% load penn cove j,i
dir0 = Ldir['LOo'] / 'extract' / 'tef2'
seg_info_dict_fn = dir0 / ('seg_info_dict_' + sect_gctag + '_' + Ldir['riv'] + '.p')
seg_df = pd.read_pickle(seg_info_dict_fn)
ji_list = seg_df['pc0_m']['ji_list']
jj = [x[0] for x in ji_list]
ii = [x[1] for x in ji_list]


in_dir = Ldir['roms_out'] / Ldir['gtagex']
G, S, T = zrfun.get_basic_info(in_dir / ('f' + Ldir['ds0']) / 'ocean_his_0002.nc')

fn0 = xr.open_dataset(in_dir / ('f' + Ldir['ds0']) / 'ocean_his_0002.nc')
dx = 1/fn0.pm.values
dy = 1/fn0.pn.values
area = dx * dy

dt0 = datetime.strptime(Ldir['ds0'], Lfun.ds_fmt)
dt1 = datetime.strptime(Ldir['ds1'], Lfun.ds_fmt)
dt00 = dt0

t = []
SV = [] # sum(salt*vol) in the whole domain, hourly
vol_hrly = []  # total volume in the domain, hourly
surf_s_flux = []  # EminusP: m s-1, see /ROMS/External/varinfo.ymal for more descriptions

while dt00 <= dt1:
    print(dt00)
    sys.stdout.flush()
    ds00 = dt00.strftime(Lfun.ds_fmt)
    fn_list = Lfun.get_fn_list('hourly', Ldir, ds00, ds00)
    
    for fn in fn_list[0:-1]: 
        ds_his = xr.open_dataset(fn)
        EminusP = ds_his.EminusP.values.squeeze()  # EminusP
        salt_surf = ds_his.salt.values.squeeze()[-1,:,:]  # surface salinity
        #print(fn)
        h = ds_his.h.values      
        zeta = ds_his.zeta.values.squeeze()
        zw = zrfun.get_z(h, zeta, S, only_w=True)
        dz = np.diff(zw, axis=0)
        vol = dx*dy*dz
            
        #-------- salt*vol --------
        salt_tmp = ds_his.salt.values.squeeze()
        SV.append(np.nansum(salt_tmp[:,jj,ii] * vol[:,jj,ii]))  # salt*vol
        vol_hrly.append(np.nansum(vol[:,jj,ii]))
        tmp = salt_surf*EminusP*area  # also see Parker's github /LO/extract/tef2/tracer_budget.py line 177
        surf_s_flux.append(np.nansum(tmp[jj,ii]))
        t.append(ds_his.ocean_time.values)
        
    dt00 = dt00 + timedelta(days=1)
    
dict_tmp = {'t': t,
            'salt_vol_sum_hrly': SV,
            'vol_hrly': vol_hrly,
            'surf_s_flux': surf_s_flux
           }
dir1 = Ldir['LOo'] / 'extract' / 'tef2'
pickle.dump(dict_tmp, open(dir1 / ("vol_SV_hrly_"+Ldir['ds0'] +'_' + Ldir['ds1']+'_'+Ldir['gtagex']+'.p'),'wb'))
