#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 16:51:55 2023

@author: dakotamascarenas
"""

import VFC_functions as vfun

import pandas as pd

import numpy as np

import zrfun

from lo_tools import forcing_argfun2 as ffun

from lo_tools import zrfun, pfun

import matplotlib.pyplot as plt



Ldir = ffun.intro() # this handles all the argument passing

min_lat = Lat[min(jjj) - 10]
max_lat = Lat[max(jjj) + 10]

min_lon = Lon[min(iii) - 10]
max_lon = Lon[max(iii) + 10]


dt = pd.Timestamp('2017-01-01 01:30:00')
fn_his = vfun.get_his_fn_from_dt(Ldir, dt)

G, S, T = zrfun.get_basic_info(fn_his)
land_mask = G['mask_rho']
Lon = G['lon_rho'][0,:]
Lat = G['lat_rho'][:,0]
plon,plat = pfun.get_plon_plat(G['lon_rho'], G['lat_rho'])
z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
dz = np.diff(z_w_grid,axis=0)
dv = dz*G['DX']*G['DY']
h = G['h']

var_list = ['salt','temp','NO3','NH4','chlorophyll','TIC','alkalinity','oxygen']


for var in V.keys():
    
    if var in var_list:

        for cell in range(30):
        
            pfun.start_plot(fs=14, figsize=(15,15))
            fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
            
            plot_var = V[var][0,cell, :,:]
            
            c0 = ax[0,0].pcolormesh(plon,plat, plot_var)
            
            # for cid in info_df.index:
                        
            #     ax[0,0].plot(info_df.loc[cid, 'lon'], info_df.loc[cid, 'lat'],'o', c = 'white', markeredgecolor='black', markersize=10) #c = cmap0(n)
                
            
            # ax[0,0].set_xlim([min_lon,max_lon])
            # ax[0,0].set_ylim([min_lat,max_lat])
            ax[0,0].tick_params(labelrotation=45)
            ax[0,0].set_title(var + 'Layer ' + str(cell))
            
            fig.colorbar(c0,ax=ax[0,0])
            
            
            
            pfun.add_coast(ax[0,0])
            pfun.dar(ax[0,0])
    
            fig.tight_layout()
            plt.savefig('/Users/dakotamascarenas/Desktop/pltz/'+var + '_layer_' + str(cell), bbox_inches='tight')

        
        
        

                    