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

hyp_vol_dict = {}

hyp_array_dict = {}

wtd_avg_dict = {}

# %%

for i in dir_list:
    
    fn = path + i + '/ocean_his_0001.nc'
    
    ds = xr.open_dataset(fn)

    G,S,T = zrfun.get_basic_info(fn)
    
    dt = T['dt']

    z_rho, z_w = zrfun.get_z(G['h'], 0*G['h'], S)

    dz = np.diff(z_w,axis=0)

    dv = dz*G['DX']*G['DY']

    oxygen_mg_L = ds.oxygen*32/1000 #molar mass of O2
    
    oxygen_mg_L_np = oxygen_mg_L.isel(ocean_time = 0).to_numpy().reshape(30,1302,663)
        
    wtd_avg_conc = np.nanmean(dv*1000*oxygen_mg_L_np)/np.nanmean(dv*1000)
        
    hyp_array = np.ma.masked_where(oxygen_mg_L_np > 2, oxygen_mg_L_np).filled(fill_value = np.nan)
    
    hyp_vol = np.ma.masked_where(oxygen_mg_L_np > 2, dv).filled(fill_value = np.nan)
        
    hyp_vol_sum = np.nansum(hyp_vol)
    
    hyp_array_dict[dt] = hyp_array
    
    hyp_vol_dict[dt] = hyp_vol_sum
    
    wtd_avg_dict[dt] = wtd_avg_conc
    
    
    hyp_deep = hyp_array[0,:,:].reshape(1302,663)

    plon, plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])

    fig0, ax0 = plt.subplots(figsize=(8,10))

    cs0 = ax0.pcolormesh(plon,plat,hyp_deep,vmin = 0,vmax=2)

    ax0.set_title('[O_2] (mg/L) at Bottom (S=0) ' + str(dt))

    pfun.add_coast(ax0)

    fig0.colorbar(cs0, ax=ax0)

    ax0.set_xlim([-125.5, -122])
    ax0.set_ylim([46.5, 50.5])
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(dt).replace(" ","_").replace(":","-") +'_deep.png', dpi = 200)
    

    hyp_surf = hyp_array[-1,:,:].reshape(1302,663)

    plon, plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])

    fig1, ax1 = plt.subplots(figsize=(8,10))

    cs1 = ax1.pcolormesh(plon,plat,hyp_surf,vmin = 0,vmax=2)

    ax1.set_title('[O_2] (mg/L) at Surface (S=29) ' + str(dt))

    pfun.add_coast(ax1)

    fig1.colorbar(cs1, ax=ax1)

    ax1.set_xlim([-125.5, -122])
    ax1.set_ylim([46.5, 50.5])
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/' + str(dt).replace(" ","_").replace(":","-") +'_surf.png', dpi = 200)
    
    
# %%

fig2, ax2 = plt.subplots(figsize=(10,5))

hyp_vol_ordered = sorted(hyp_vol_dict.items())

x, y = zip(*hyp_vol_ordered)

ax2.set_title('Total Hypoxic Volume from LO [m^3]')

plt.plot(x,y)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/hyp_vol_2022-11-16--2022-11-30.png', dpi = 200)


# %%

fig3, ax3 = plt.subplots(figsize=(10,5))

wtd_avg_ordered = sorted(wtd_avg_dict.items())

x, y = zip(*wtd_avg_ordered)

ax3.set_title('Avg O2 Concentration from LO [mg/L]')

plt.plot(x,y)

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/avg_conc_2022-11-16--2022-11-30.png', dpi = 200)


        
    # hyp_vol = np.array([oxygen_mg_L_np if (oxygen_mg_L_np <= 2).any() else np.nan for oxygen_mg_L_np in oxygen_mg_L_np], dtype=np.float64) #idk why it's float32
    
    # hyp_vol_dict[dt] = hyp_vol
    

    #hyp_array = np.empty([30,1302,663])

    #hyp_array.fill(np.nan)
    
    # hyp_vol = 0

    # for i in range(np.shape(idx_hyp)[1]):
    #     s_rho_idx = idx_hyp[0][i]
    #     eta_rho_idx = idx_hyp[1][i]
    #     xi_rho_idx = idx_hyp[2][i]
    #     #hyp_array[s_rho_idx, eta_rho_idx, xi_rho_idx] = oxygen_mg_L_np[s_rho_idx, eta_rho_idx, xi_rho_idx]
    #     hyp_vol = hyp_vol + dv[s_rho_idx, eta_rho_idx, xi_rho_idx]
        
    # hyp_vol_array[0,c] = np.datetime64(T['dt'])
    # hyp_vol_array[1,c] = hyp_vol