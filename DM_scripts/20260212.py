#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:19:09 2026

@author: dakotamascarenas
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from matplotlib.path import Path
import gsw
from cmcrameri import cm as cmc

from cmocean import cm as cmo# have to import after matplotlib to work on remote machine

# %%

vol_df = pd.read_pickle('/Users/dakotamascarenas/LO_output/extract/tef2/vol_df_wb1_pc0.p')

# %%

gridname = 'wb1'

tag = 'r0'

ex_name = 'xn11b'

Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

Ldir['list_type'] = 'hourly0'


sect_path = Ldir['LOo'] / 'section_lines'
sect = 'pc.p'

sect_fn = sect_path / sect

pdict = pickle.load(open(sect_fn, 'rb'))

grid_fn = Ldir['grid'] / 'grid.nc'

gctag = 'wb1_pc0'

out_dir = Ldir['LOo'] / 'extract' / 'tef2'


# get grid data
ds = xr.open_dataset(grid_fn)
h = ds.h.values
m = ds.mask_rho.values
# these are used for making segment volumes
H = h.copy()
DX = 1/ds.pm.values
DY = 1/ds.pn.values
DA = DX * DY
DV = H * DA
lon_rho = ds.lon_rho.values
lat_rho = ds.lat_rho.values
# depth for plotting
h[m==0] = np.nan
# coordinates for plotting
plon, plat = pfun.get_plon_plat(ds.lon_rho.values, ds.lat_rho.values)
aa = pfun.get_aa(ds)
# coordinates for convenience in plotting
lor = ds.lon_rho[0,:].values
lar = ds.lat_rho[:,0].values
lou = ds.lon_u[0,:].values
lau = ds.lat_u[:,0].values
lov = ds.lon_v[0,:].values
lav = ds.lat_v[:,0].values
ds.close

sect_df = pd.read_pickle(out_dir / ('sect_df_' + gctag + '.p'))

# %%

Budget_df = pd.read_pickle('/Users/dakotamascarenas/LO_output/extract/wb1_r0_xn11b/tef2/Budgets__2017.09.01_2017.09.30/Hood_Canal.p')
