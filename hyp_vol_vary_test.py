import numpy as np
import xarray as xr
import pickle
from datetime import datetime, timedelta
import pandas as pd
from cmocean import cm
import sys
import os

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
#import pinfo
from importlib import reload
#reload(pfun)
#reload(pinfo)

Ldir = Lfun.Lstart()
if '_mac' in Ldir['lo_env']: # mac version
    pass
else: # remote linux version
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

path = '/Users/dakotamascarenas/LO_roms/cas6_v0_live/'

dir_list = os.listdir(path)

dir_list.remove('.DS_Store')

hyp_vol_array = np.empty(shape=(15,2))

c=0

for i in dir_list:
    
    fn = path + i + '/ocean_his_0001.nc'

    ds = xr.open_dataset(fn)

    G,S,T = zrfun.get_basic_info(fn)

    z_rho, z_w = zrfun.get_z(G['h'], 0*G['h'], S)

    dz = np.diff(z_w,axis=0)

    dv = dz*G['DX']*G['DY']

    oxygen_mg_L = ds.oxygen*32/1000 #molar mass of O2
    
    oxygen_mg_L_np = oxygen_mg_L.isel(ocean_time = 0).to_numpy().reshape(30,1302,663)

    idx_hyp = np.where(oxygen_mg_L_np <= 2)

    idx_ok = np.where(oxygen_mg_L_np > 2)

    #hyp_array = np.empty([30,1302,663])

    #hyp_array.fill(np.nan)
    
    hyp_vol = 0

    for i in range(np.shape(idx_hyp)[1]):
        s_rho_idx = idx_hyp[0][i]
        eta_rho_idx = idx_hyp[1][i]
        xi_rho_idx = idx_hyp[2][i]
        #hyp_array[s_rho_idx, eta_rho_idx, xi_rho_idx] = oxygen_mg_L_np[s_rho_idx, eta_rho_idx, xi_rho_idx]
        hyp_vol = hyp_vol + dv[s_rho_idx, eta_rho_idx, xi_rho_idx]
        
    hyp_vol_array[0,c] = np.datetime64(T['dt'])
    hyp_vol_array[1,c] = hyp_vol