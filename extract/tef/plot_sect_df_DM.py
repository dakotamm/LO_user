#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 08:00:08 2023

@author: dakotamascarenas
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import matplotlib.pyplot as plt
import matplotlib.path as mpth
import xarray as xr
import numpy as np
import pandas as pd

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

import tef_fun_old as tfun

import VFC_functions as vfun

Ldir = Lfun.Lstart(gridname='cas6')

sect_df = tfun.get_sect_df(Ldir['gridname'])

vol_dir, v_df, j_dict, i_dict, seg_list = vfun.getSegmentInfo(Ldir)

Ldir['gtagex'] = 'cas6_traps2_x2b'

dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

G, S, T, land_mask, Lon, Lat, plon, plat, z_rho_grid, z_w_grid, dz, dv, h = vfun.getGridInfo(fn_his)

jjj_dict, iii_dict, seg_list = vfun.defineSegmentIndices('regions', j_dict, i_dict)

# %%


fig, ax = plt.subplots(figsize=(10,15))
# map
pfun.add_coast(ax)
pfun.dar(ax)
aa = [-125.5, -122, 46.5, 50.5]
ax.axis(aa)

for sect in sect_df.index:
    
    x0 = sect_df.loc[sect, 'x0']
    
    x1 = sect_df.loc[sect, 'x1']

    y0 = sect_df.loc[sect, 'y0']
    
    y1 = sect_df.loc[sect, 'y1']


    plt.plot([x0,x1],[y0,y1], label =sect, linewidth = 3)
    
ax.legend()

fig.tight_layout()

plt.savefig('/Users/dakotamascarenas/Desktop/pltz/cas6_sect.png')

# %%


#land_mask_plot = land_mask.copy()

#land_mask_plot[land_mask ==0] = np.nan

#c00 = ax.pcolormesh(plon, plat, land_mask_plot, cmap='Greys')

c = 0

for seg_name in seg_list:
    
    fig, ax = plt.subplots(figsize=(10,15))

    pfun.add_coast(ax)
    pfun.dar(ax)
    aa = [-125.5, -122, 46.5, 50.5]
    ax.axis(aa)
    
    seg_plot = np.full_like(land_mask.copy(), np.nan)
    
    jjj = jjj_dict[seg_name]
    
    iii = iii_dict[seg_name] # THIS IS AN ISSUE, OVERLAPPING SOJ AND HOOD CANAL
    
    seg_plot[jjj,iii] = c
    
    c0 = ax.pcolormesh(plon, plat, seg_plot, cmap = 'Set2')
    
    c+=1
    
    ax.set_title(seg_name)
    
    fig.tight_layout()

    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/cas6_' + seg_name.replace(' ','') + '.png')











