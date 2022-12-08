import numpy as np
import xarray as xr
import pickle
from datetime import datetime, timedelta
import pandas as pd
from cmocean import cm
import sys

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

fn = '/Users/dakotamascarenas/LO_roms/cas6_v0_live/f2022.11.30/ocean_his_0001.nc'

ds = xr.open_dataset(fn)

G,S,T = zrfun.get_basic_info(fn)

z_rho, z_w = zrfun.get_z(G['h'], 0*G['h'], S)

dz = np.diff(z_w,axis=0)

dv = dz*G['DX']*G['DY']

oxygen_mg_L = ds.oxygen*32/1000 #molar mass of O2

oxygen_mg_L_surf = oxygen_mg_L.isel(s_rho = -1).to_numpy().reshape(1302,663)

oxygen_mg_L_deep = oxygen_mg_L.isel(s_rho = 0).to_numpy().reshape(1302,663)

plon, plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])

fig0, (ax00,ax01) = plt.subplots(1,2)

cs00 = ax00.pcolormesh(plon,plat,oxygen_mg_L_surf,vmin = 2,vmax=12)

ax00.set_title('[O_2] (mg/L) at Surface (S=30)')

pfun.add_coast(ax00)

fig0.colorbar(cs00, ax=ax00)

ax00.set_xlim([-125.5, -122])
ax00.set_ylim([46.5, 50.5])

cs01 = ax01.pcolormesh(plon,plat,oxygen_mg_L_deep,vmin = 2,vmax=12)

ax01.set_title('[O_2] (mg/L) at Bottom (S=1)')

pfun.add_coast(ax01)

fig0.colorbar(cs01, ax=ax01)

ax01.set_xlim([-125.5, -122])
ax01.set_ylim([46.5, 50.5])

plt.show()

oxygen_mg_L_np = oxygen_mg_L.isel(ocean_time = 0).to_numpy().reshape(30,1302,663)

idx_hyp = np.where(oxygen_mg_L_np <= 2)

idx_ok = np.where(oxygen_mg_L_np > 2)

hyp_vol = 0

hyp_array = np.empty([30,1302,663])

hyp_array.fill(np.nan)

for i in range(np.shape(idx_hyp)[1]):
    s_rho_idx = idx_hyp[0][i]
    eta_rho_idx = idx_hyp[1][i]
    xi_rho_idx = idx_hyp[2][i]
    hyp_array[s_rho_idx, eta_rho_idx, xi_rho_idx] = oxygen_mg_L_np[s_rho_idx, eta_rho_idx, xi_rho_idx]
    hyp_vol = hyp_vol + dv[s_rho_idx, eta_rho_idx, xi_rho_idx]

hyp_array_surf = hyp_array[-1,:,:]

hyp_array_deep = hyp_array[0,:,:]

fig1,(ax10,ax11) = plt.subplots(1,2)

cs10 = ax10.pcolormesh(plon,plat,hyp_array_surf,vmin = 0,vmax=2)

ax10.set_title('Hypoxic [O_2] (mg/L) at Surface (S=30)')

pfun.add_coast(ax10)

fig1.colorbar(cs10, ax=ax10)

ax10.set_xlim([-125.5, -122])
ax10.set_ylim([46.5, 50.5])

cs11 = ax11.pcolormesh(plon,plat,hyp_array_deep,vmin = 0,vmax=2)

ax11.set_title('Hypoxic [O_2] (mg/L) at Bottom (S=1)')

pfun.add_coast(ax11)

fig1.colorbar(cs11, ax=ax11)

ax11.set_xlim([-125.5, -122])
ax11.set_ylim([46.5, 50.5])

plt.show()




