#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:17:28 2025

@author: dakotamascarenas
"""

# making transects of observations

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import matplotlib.pyplot as plt
import matplotlib.path as mpth
import xarray as xr
import numpy as np
import pandas as pd
import datetime

from warnings import filterwarnings
filterwarnings('ignore') # skip some warning messages

import seaborn as sns

import scipy.stats as stats

import D_functions as dfun

import pickle

import math

from scipy.interpolate import interp1d

import gsw

import matplotlib.path as mpth

import matplotlib.patches as patches

import cmocean




Ldir = Lfun.Lstart(gridname='cas7')


fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)
lon_rho = dsg.lon_rho.values
lat_rho = dsg.lat_rho.values #these are cell centers anyway...

lon_psi = dsg.lon_psi.values
lat_psi = dsg.lat_psi.values #these are cell edges...

m = dsg.mask_rho.values
xp, yp = pfun.get_plon_plat(lon_rho,lat_rho)
depths = dsg.h.values
depths[m==0] = 0 #np.nan #set to 0 on landmask

lon_rho_1D = lon_rho[0,:]

lat_rho_1D = lat_rho[:,0]

lon_psi_1D = lon_psi[0,:]

lat_psi_1D = lat_psi[:,0]

# weird, to fix

mask_rho = np.transpose(dsg.mask_rho.values)
zm = -depths.copy()
zm[np.transpose(mask_rho) == 0] = np.nan
zm[np.transpose(mask_rho) != 0] = -1

zm_inverse = zm.copy()

zm_inverse[np.isnan(zm)] = -1

zm_inverse[zm==-1] = np.nan


X = lon_rho[0,:] # grid cell X values
Y = lat_rho[:,0] # grid cell Y values

plon, plat = pfun.get_plon_plat(lon_rho,lat_rho)


j1 = 570
j2 = 1170
i1 = 220
i2 = 652




#poly_list = ['carr_inlet_mid', 'lynch_cove_mid', 'near_seattle_offshore', 'saratoga_passage_mid', 'point_jefferson'] # 5 sites + 4 basins

poly_list = ['ps']

odf_dict, path_dict = dfun.getPolyData(Ldir, poly_list, source_list=['kc_whidbey'], otype_list=['bottle', 'ctd'], year_list=np.arange(1930,2025))


basin_list = list(odf_dict.keys())

var_list = ['DO_mg_L','SA', 'CT'] #, 'NO3_uM', 'Chl_mg_m3'] #, 'NO2 (uM), 'NH4_uM', 'SiO4 (uM)', 'PO4 (uM)', 'TA (uM)','DIC (uM)', 'DO (uM)']


odf = dfun.dictToDF(odf_dict, var_list, lon_rho_1D, lat_rho_1D, depths, lon_rho, lat_rho, poly_list, path_dict, basin_list)


pc_sites = ['PENNCOVEWEST', 'PENNCOVEENT', 'SARATOGARP']

pcwest_lat = 48.2249

pcwest_lon = -122.72

pcent_lat = 48.237

pcent_lon = -122.655

srp_lat = 48.24

srp_lon = -122.55

start_lat = 48.22

start_lon = -122.733578

end_lat = 48.245867

end_lon = -122.515921


# assemble transect line of cell edges

x_e = []

y_e = []

for l in [start_lon, pcwest_lon, pcent_lon, srp_lon, end_lon]:
    
    x_e.append(zfun.find_nearest(lon_psi_1D, l))
    
for ll in [start_lat, pcwest_lat, pcent_lat, srp_lat, end_lat]:
    
    y_e.append(zfun.find_nearest(lat_psi_1D, ll))
    


# assemble transect line of cell edges

x_center_idx = []

y_center_idx = []

for l in [start_lon, pcwest_lon, pcent_lon, srp_lon, end_lon]:
    
    x_center_idx.append(zfun.find_nearest_ind(lon_rho_1D, l))
    
for ll in [start_lat, pcwest_lat, pcent_lat, srp_lat, end_lat]:
    
    y_center_idx.append(zfun.find_nearest_ind(lat_rho_1D, ll))



x_center_idx_dumb_interp = [553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,580,581,582,583,584,585,586]

y_center_idx_dumb_interp = [880,881,882,882, 882,882, 883,883,883,884,884,884,884,884,884,884,884,884,884,884,884,885,885, 885, 885, 885, 885, 885, 885, 885, 885, 886,886,886]



lons_plot = lon_rho_1D[x_center_idx_dumb_interp]

lats_plot = lat_rho_1D[y_center_idx_dumb_interp]


depths_plot = depths[y_center_idx_dumb_interp, x_center_idx_dumb_interp]



cast_location_lat = odf[odf['name'].isin(pc_sites)].groupby('name').first().reset_index().dropna()['lat'].to_numpy()

cast_location_lon = odf[odf['name'].isin(pc_sites)].groupby('name').first().reset_index().dropna()['lon'].to_numpy()

# %%


from scipy.spatial import KDTree


c=0

for date in [738189]: #odf['date_ordinal'].unique():
    
    plot_df = odf[(odf['name'].isin(pc_sites)) & (odf['date_ordinal'] == date) & (odf['var'] == 'DO_mg_L')]

    xx = np.linspace(lon_rho_1D[553], lon_rho_1D[586],2000)
    zz = np.linspace(-100,0,500)
    x_grid, z_grid = np.meshgrid(xx, zz)
    
    depth_array = np.full([len(zz),len(xx)], -99)
        
    for i in range(len(xx)):
        
        point_idx = zfun.find_nearest_ind(lons_plot, xx[i])
    
        depth_array[:, i] = depths_plot[point_idx]*-1
        
    water_array = z_grid.copy()
    
    water_array[water_array < depth_array] = -99
    
    water_array = np.ma.masked_array(water_array,water_array==-99)
    
    water_array[~water_array.mask] = -1
    
    cast_value_array = water_array.copy()
    
    for idx in plot_df.index:
        
        xxx = plot_df.loc[idx, 'lon']
        
        zzz = plot_df.loc[idx, 'z']
        
        x_idx = zfun.find_nearest_ind(xx, xxx)
        
        z_idx = zfun.find_nearest_ind(zz, zzz)
        
        cast_value_array[z_idx, x_idx] = plot_df.loc[idx, 'val']
        

        
    cast_value_masked_array = cast_value_array.copy()
    cast_value_masked_array = np.ma.masked_array(cast_value_masked_array, cast_value_masked_array==-1)
    
    xy_water = np.array((x_grid[~water_array.mask],z_grid[~water_array.mask])).T
    xy_land = np.array((x_grid[water_array.mask],z_grid[water_array.mask])).T
    
    xy_casts = np.array((x_grid[~cast_value_masked_array.mask],z_grid[~cast_value_masked_array.mask])).T
    
    tree = KDTree(xy_casts)
    
    tree_query = tree.query(xy_water)[1]
    
    
    
    
    distances, indices = tree.query(xy_water,k=2)
        
    result_array = water_array.copy()
    result_array[~water_array.mask] = cast_value_array[~cast_value_masked_array.mask][tree_query]
    


    # Perform linear interpolation
    weights = 1 / distances
    
    weights_0 = weights[:,0]
    
    weights_1 = weights[:,1]
    
    indices_0 = indices[:,0]
    
    indices_1 = indices[:,1]
    
    value_0 = np.full_like(indices_0, 0)
    
    value_1 = np.full_like(indices_0, 0)
    
    for i in range(len(indices)):
        
        value_0 = cast_value_masked_array[indices_0[i]]
        
        value_1 = cast_value_masked_array[indices_1[i]]
    
    

    
    values = indices.copy()
        
    
    values[:,:] = [cast_value_masked_array[indices[:,0]], cast_value_masked_array[indices[:,1]]]
            
    weighted_values = cast_value_masked_array[indices].reshape(np.shape(indices)) * weights
    interpolated_value = np.sum(weighted_values,axis=1) / np.sum(weights,axis=1)
    
    result_array = water_array.copy()
    
    result_array[~water_array.mask] = cast_value_array[~cast_value_masked_array.mask][interpolated_value]
    
    
    fig, ax= plt.subplots(nrows=2, sharex=True, figsize=(8,6))
    
    ax[0].plot(lons_plot, lats_plot)
    
    ax[0].scatter(cast_location_lon, cast_location_lat) 
    
    ax[0].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
      
    ax[0].set_ylim(lat_rho_1D[878], lat_rho_1D[888])
    
    pfun.dar(ax[0])  
      
    pfun.add_coast(ax[0])
    
    
    
    
    
    ax[1].pcolormesh(xx,zz,result_array)
    
    ax[1].set_ylim(-100,0)
    
    ax[1].scatter(plot_df['lon'], plot_df['z'], s=0.01, color='k')
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_transecet_DO_00' + "{:02d}".format(c) +'.png', bbox_inches='tight', dpi=500, transparent=False)
    


    


# %%



from scipy.spatial import KDTree

from scipy.interpolate import LinearNDInterpolator



c=0

for date in [738189]: #odf['date_ordinal'].unique():
    
    plot_df = odf[(odf['name'].isin(pc_sites)) & (odf['date_ordinal'] == date) & (odf['var'] == 'DO_mg_L')]

    xx = np.linspace(lon_rho_1D[553], lon_rho_1D[586],2000)
    zz = np.linspace(-100,0,500)
    x_grid, z_grid = np.meshgrid(xx, zz)
    
    depth_array = np.full([len(zz),len(xx)], -99)
        
    for i in range(len(xx)):
        
        point_idx = zfun.find_nearest_ind(lons_plot, xx[i])
    
        depth_array[:, i] = depths_plot[point_idx]*-1
        
    water_array = z_grid.copy()
    
    water_array[water_array < depth_array] = -99
    
    water_array = np.ma.masked_array(water_array,water_array==-99)
    
    water_array[~water_array.mask] = -1
    
    cast_value_array = water_array.copy()
    
    for idx in plot_df.index:
        
        xxx = plot_df.loc[idx, 'lon']
        
        zzz = plot_df.loc[idx, 'z']
        
        x_idx = zfun.find_nearest_ind(xx, xxx)
        
        z_idx = zfun.find_nearest_ind(zz, zzz)
        
        cast_value_array[z_idx, x_idx] = plot_df.loc[idx, 'val']
        
    cast_value_array[cast_value_array == -1] = np.nan
    


    
    fig, ax= plt.subplots(nrows=2, sharex=True, figsize=(8,6))
    
    ax[0].plot(lons_plot, lats_plot)
    
    ax[0].scatter(cast_location_lon, cast_location_lat) 
    
    ax[0].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
       
    ax[0].set_ylim(lat_rho_1D[878], lat_rho_1D[888]) 
    
    pfun.dar(ax[0])  
      
    pfun.add_coast(ax[0]) 
    
    cast_value_array[cast_value_array == -1] = np.nan
    
    ax[1].contourf(x_grid,z_grid, cast_value_array)
    
    ax[1].set_ylim(-100,0)
    
    ax[1].scatter(plot_df['lon'], plot_df['z'], s=0.01, color='k')
    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_transecet_DO_00' + "{:02d}".format(c) +'.png', bbox_inches='tight', dpi=500, transparent=False)



# %%






c=0

for date in odf['date_ordinal'].unique():
    
    plot_df = odf[(odf['name'].isin(pc_sites)) & (odf['date_ordinal'] == date) & (odf['var'] == 'DO_mg_L')]
    
    vals_grid = plot_df[['lon','z','val']].pivot_table(index='lon', columns='z', values='val').T.values
    
    # xx = np.linspace(lon_rho_1D[553], lon_rho_1D[586],2000)
    # zz = np.linspace(-100,0,500)
    # x_grid, z_grid = np.meshgrid(xx, zz)
    

    # X_unique = np.sort(plot_df['lon'].unique())
    # Z_unique = np.sort(plot_df['z'].unique())
    # X_grid, Z_grid = np.meshgrid(X_unique, Z_unique)
    

    
    fig, ax= plt.subplots(nrows=2, sharex=True, figsize=(9,6))
    
    ax[0].plot(lons_plot, lats_plot)
    
    ax[0].scatter(cast_location_lon, cast_location_lat, s=5, color='k')
    
     
    ax[0].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
     
    ax[0].set_ylim(lat_rho_1D[878], lat_rho_1D[888])
    
    ax[0].text(0.1,0.9, str(plot_df['datetime'].iloc[0]), transform=ax[0].transAxes, fontweight='bold')

    
     
    #plot_df = odf[odf['name'].isin(pc_sites)] #FILTER THIS TO SPECIFIC TIME PERIOD
     
    
    pfun.dar(ax[0])  
      
    pfun.add_coast(ax[0])
    
    #ax[1].contourf(X_grid,Z_grid, vals_grid)
    
    
    tcf = ax[1].tricontourf(plot_df['lon'], plot_df['z'], plot_df['val'], vmin=0, vmax=12,cmap="RdBu")#, levels=10, line widths=0.5, colors='k')
    
   # cntr2 = ax[1].tricontourf(plot_df['lon'], plot_df['lat'], plot_df['val'], levels=10, cmap="RdBu_r")
   
    #ax[1].colorbar(tcf)

    
    ax[1].scatter(plot_df['lon'], plot_df['z'], s=0.01, color='k')
    
    ax[1].plot(lons_plot,depths_plot*-1) 
    
    
    ax[1].set_ylim(-100,0)
    
    ax[1].set_ylabel('depth [m]')

    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_transecet_DO_00' + "{:02d}".format(c) +'.png', bbox_inches='tight', dpi=500, transparent=False)

    c+=1

# %%

c=0

for date in odf['date_ordinal'].unique():
    
    plot_df = odf[(odf['name'].isin(pc_sites)) & (odf['date_ordinal'] == date) & (odf['var'] == 'CT')]
    
    vals_grid = plot_df[['lon','z','val']].pivot_table(index='lon', columns='z', values='val').T.values
    
    # xx = np.linspace(lon_rho_1D[553], lon_rho_1D[586],2000)
    # zz = np.linspace(-100,0,500)
    # x_grid, z_grid = np.meshgrid(xx, zz)
    

    # X_unique = np.sort(plot_df['lon'].unique())
    # Z_unique = np.sort(plot_df['z'].unique())
    # X_grid, Z_grid = np.meshgrid(X_unique, Z_unique)
    

    
    fig, ax= plt.subplots(nrows=2, sharex=True, figsize=(9,6))
    
    ax[0].plot(lons_plot, lats_plot)
    
    ax[0].scatter(cast_location_lon, cast_location_lat, s=5, color='k')
    
     
    ax[0].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
     
    ax[0].set_ylim(lat_rho_1D[878], lat_rho_1D[888])
    
    ax[0].text(0.1,0.9, str(plot_df['datetime'].iloc[0]), transform=ax[0].transAxes, fontweight='bold')

    
     
    #plot_df = odf[odf['name'].isin(pc_sites)] #FILTER THIS TO SPECIFIC TIME PERIOD
     
    
    pfun.dar(ax[0])  
      
    pfun.add_coast(ax[0])
    
    #ax[1].contourf(X_grid,Z_grid, vals_grid)
    
    
    tcf = ax[1].tricontourf(plot_df['lon'], plot_df['z'], plot_df['val'], vmin=5, vmax=25,cmap="PuRd")#, levels=10, line widths=0.5, colors='k')
    
   # cntr2 = ax[1].tricontourf(plot_df['lon'], plot_df['lat'], plot_df['val'], levels=10, cmap="RdBu_r")
   
    #ax[1].colorbar(tcf)

    
    ax[1].scatter(plot_df['lon'], plot_df['z'], s=0.01, color='k')
    
    ax[1].plot(lons_plot,depths_plot*-1) 
    
    
    ax[1].set_ylim(-100,0)
    
    ax[1].set_ylabel('depth [m]')

    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_transecet_CT_00' + "{:02d}".format(c) +'.png', bbox_inches='tight', dpi=500, transparent=False)

    c+=1

# %%

c=0

for date in odf['date_ordinal'].unique():
    
    plot_df = odf[(odf['name'].isin(pc_sites)) & (odf['date_ordinal'] == date) & (odf['var'] == 'SA')]
    
    vals_grid = plot_df[['lon','z','val']].pivot_table(index='lon', columns='z', values='val').T.values
    
    # xx = np.linspace(lon_rho_1D[553], lon_rho_1D[586],2000)
    # zz = np.linspace(-100,0,500)
    # x_grid, z_grid = np.meshgrid(xx, zz)
    

    # X_unique = np.sort(plot_df['lon'].unique())
    # Z_unique = np.sort(plot_df['z'].unique())
    # X_grid, Z_grid = np.meshgrid(X_unique, Z_unique)
    

    
    fig, ax= plt.subplots(nrows=2, sharex=True, figsize=(9,6))
    
    ax[0].plot(lons_plot, lats_plot)
    
    ax[0].scatter(cast_location_lon, cast_location_lat, s=5, color='k')
    
     
    ax[0].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
     
    ax[0].set_ylim(lat_rho_1D[878], lat_rho_1D[888])
    
    ax[0].text(0.1,0.9, str(plot_df['datetime'].iloc[0]), transform=ax[0].transAxes, fontweight='bold')

    
     
    #plot_df = odf[odf['name'].isin(pc_sites)] #FILTER THIS TO SPECIFIC TIME PERIOD
     
    
    pfun.dar(ax[0])  
      
    pfun.add_coast(ax[0])
    
    #ax[1].contourf(X_grid,Z_grid, vals_grid)
    
    
    tcf = ax[1].tricontourf(plot_df['lon'], plot_df['z'], plot_df['val'], vmin=15, vmax=35,cmap="GnBu_r")#, levels=10, line widths=0.5, colors='k')
    
   # cntr2 = ax[1].tricontourf(plot_df['lon'], plot_df['lat'], plot_df['val'], levels=10, cmap="RdBu_r")
   
    #ax[1].colorbar(tcf)

    
    ax[1].scatter(plot_df['lon'], plot_df['z'], s=0.01, color='k')
    
    ax[1].plot(lons_plot,depths_plot*-1) 
    
    
    ax[1].set_ylim(-100,0)
    
    ax[1].set_ylabel('depth [m]')

    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_transecet_SA_00' + "{:02d}".format(c) +'.png', bbox_inches='tight', dpi=500, transparent=False)

    c+=1
    
# %%
    
c=0

for date in odf['date_ordinal'].unique():
    
    fig, ax= plt.subplots(nrows=4, sharex=True, figsize=(9,12))
    
    ax[0].plot(lons_plot, lats_plot)
    
    ax[0].scatter(cast_location_lon, cast_location_lat, s=5, color='k')
    
     
    ax[0].set_xlim(lon_rho_1D[553], lon_rho_1D[586])
     
    ax[0].set_ylim(lat_rho_1D[878], lat_rho_1D[888])
    
    
    pfun.dar(ax[0])  
      
    pfun.add_coast(ax[0])
    

    
    for var in var_list:
    
        plot_df = odf[(odf['name'].isin(pc_sites)) & (odf['date_ordinal'] == date) & (odf['var'] == var)]
        
        
        if var == 'CT':
            ax[0].text(0.1,0.9, str(plot_df['datetime'].iloc[0]), transform=ax[0].transAxes, fontweight='bold')

            
            vmin=8
            vmax=24
            cmap = 'PuRd'
            ax_num = 1
            unit='degC'
            
        elif var == 'DO_mg_L':
            
            vmin=0
            vmax=12
            cmap = 'RdBu'
            ax_num = 3
            unit = 'mg/L'
        
        elif var == 'SA':
            
            vmin=16
            vmax=32
            cmap = 'GnBu_r'
            ax_num = 2
            unit = 'g/kg'


        ax[ax_num].text(0.1,0.1, var + ' (' + str(vmin) + '-' + str(vmax) + unit + ')', transform=ax[ax_num].transAxes, fontweight='bold')

    
        tcf = ax[ax_num].tricontourf(plot_df['lon'], plot_df['z'], plot_df['val'], vmin=vmin, vmax=vmax,cmap=cmap)#, levels=10, line widths=0.5, colors='k')
    
   

    
        ax[ax_num].scatter(plot_df['lon'], plot_df['z'], s=0.01, color='k')
        
        ax[ax_num].plot(lons_plot,depths_plot*-1) 
        
        
        ax[ax_num].set_ylim(-100,0)
        
        ax[ax_num].set_ylabel('depth [m]')

    
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/pc_transect_00' + "{:02d}".format(c) +'.png', bbox_inches='tight', dpi=500, transparent=False)

    c+=1

