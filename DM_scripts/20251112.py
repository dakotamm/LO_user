"""
Generic code to plot any mooring extraction
"""
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import matplotlib.colors as mcolors


import cmocean

Ldir = Lfun.Lstart()

verbose = True

# %%

moor_dir = Ldir['LOo'] / 'extract'

ms = ['M1']

g = 'wb1_r0_xn11b'

temp_range = (5, 15)
salt_range = (14, 32)
DO_range   = (0, 10)
vel_range  = (-0.5, 0.5)

for m in ms:

    # load dataset
    fn = m + '_2017.01.02_2017.12.30.nc'
    moor_fn = moor_dir / g / 'moor' / 'pc1'/ fn
    ds = xr.open_dataset(moor_fn)

    salt = ds['salt']
    z_rho = ds['z_rho']
    temp = ds['temp']
    oxygen = ds['oxygen']*32/1000
    u = ds['u']
    v = ds['v']

    fig, axs = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    mesh0 = axs[0].pcolormesh(temp['ocean_time'], z_rho.T, temp.T,
                              shading='auto', cmap=cmocean.cm.thermal,
                              vmin=temp_range[0], vmax=temp_range[1])
    fig.colorbar(mesh0, ax=axs[0], label='°C')
    axs[0].set_ylabel('Depth (m)')
    axs[0].set_title('Temperature (°C)')

    mesh1 = axs[1].pcolormesh(salt['ocean_time'], z_rho.T, salt.T,
                              shading='auto', cmap=cmocean.cm.haline,
                              vmin=salt_range[0], vmax=salt_range[1])
    fig.colorbar(mesh1, ax=axs[1], label='g/kg')
    axs[1].set_ylabel('Depth (m)')
    axs[1].set_title('Salinity (g/kg)')

    mesh2 = axs[2].pcolormesh(oxygen['ocean_time'], z_rho.T, oxygen.T,
                              shading='auto', cmap=cmocean.cm.oxy,
                              vmin=DO_range[0], vmax=DO_range[1])
    # time_2d = np.tile(oxygen['ocean_time'].values, (z_rho.shape[1], 1))
    # contours = axs[2].contour(time_2d, z_rho.T, oxygen.T,
    #                           levels=[2], colors='black', linewidths=4)
    #axs[2].clabel(contours, fmt='%2.1f mg/L', colors='black')
    fig.colorbar(mesh2, ax=axs[2], label='mg/L')
    axs[2].set_ylabel('Depth (m)')
    axs[2].set_title('Dissolved Oxygen (mg/L)')

    mesh3 = axs[3].pcolormesh(u['ocean_time'], z_rho.T, u.T,
                              shading='auto', cmap=cmocean.cm.balance,
                              vmin=vel_range[0], vmax=vel_range[1])
    fig.colorbar(mesh3, ax=axs[3], label='m/s')
    axs[3].set_ylabel('Depth (m)')
    axs[3].set_title('U Velocity (m/s)')

    mesh4 = axs[4].pcolormesh(v['ocean_time'], z_rho.T, v.T,
                              shading='auto', cmap=cmocean.cm.balance,
                              vmin=vel_range[0], vmax=vel_range[1])
    fig.colorbar(mesh4, ax=axs[4], label='m/s')
    axs[4].set_ylabel('Depth (m)')
    axs[4].set_title('V Velocity (m/s)')

    axs[4].set_xlabel('Time')

    fig.suptitle(m)
    plt.tight_layout()
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+ m + '_pc1_lowpass_CT_SA_DO_u_v.png',
                dpi=500, transparent=False, bbox_inches='tight')
