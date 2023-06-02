"""
Make transects with LO output data and volume-from-casts method.

Test on mac in ipython:
run make_transect_DM -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test False

"""
import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from lo_tools import Lfun, zfun, zrfun
from lo_tools import extract_argfun as exfun
import cast_functions as cfun
from lo_tools import plotting_functions as pfun
import tef_fun as tfun
import pickle

from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from scipy.spatial import KDTree

import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import itertools

import pinfo
from importlib import reload
reload(pinfo)

Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

month_num = ['01','02','03','04','05','06','07','08','09','10','11','12']

month_str = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# get segment info
vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
v_df = pd.read_pickle(vol_dir / 'volumes.p')
j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
seg_list = list(v_df.index)

dt = pd.Timestamp('2022-01-01 01:30:00')
fn_his = cfun.get_his_fn_from_dt(Ldir, dt)

#grid info
G, S, T = zrfun.get_basic_info(fn_his)
Lon = G['lon_rho'][0, :]
Lat = G['lat_rho'][:, 0]

# %%

def get_sect(fn, vn, x_e, y_e):
    """
    Code to extract a section of any variable from a history file and return
    fields useful for plotting.
    
    The section plotting concept is that we form fields:
        dist_se: array of distance along section [km] (layer, dist) on cell edges
        zw_se: array of z position along section [m] (layer, dist) on cell edges
        fld_s: array of field values in section (layer, dist), on cell centers
    These are then what we use in pcolormesh.
    Naming conventions:
        [] is a vector on cell centers
        []_e is a vector on cell edges
        []_s is an array on cell centers
        []_se is an array on cell edges
    This should work for variables on any of the horizontal grids (u, v, rho).
    """
    
    G, S, T = zrfun.get_basic_info(fn)
    ds = xr.open_dataset(fn)

    # Make points x_e, y_e between for interpolating the field onto, on cell centers.
    x = x_e[:-1] + np.diff(x_e)/2
    y = y_e[:-1] + np.diff(y_e)/2

    # Gather some fields, making sure we use the appropriate lon, lat grids.
    if 'eta_u' in ds[vn].dims:
        lon = G['lon_u']
        lat = G['lat_u']
    elif 'eta_v' in ds[vn].dims:
        lon = G['lon_v']
        lat = G['lat_v']
    elif 'eta_rho' in ds[vn].dims:
        lon = G['lon_rho']
        lat = G['lat_rho']
    mask = G['mask_rho']
    h = G['h']
    zeta = ds['zeta'].values.squeeze()
    # Do this to make sure we don't end up with nan's in our zw field.
    h[mask==0] = 0
    zeta[mask==0] = 0

    # get zw field, used for cell edges.
    zw = zrfun.get_z(h, zeta, S, only_w=True)
    N = zw.shape[0]

    # Get 3-D field that we will full the section from
    fld = ds[vn].values.squeeze()
    # Force fld to be on the s_rho grid in the vertical if it is not already.
    if 's_w' in ds[vn].dims:
        fld = fld[:-1,:,:] + np.diff(fld,axis=0)/2

    def get_dist(x,y):
        # Create a vector of distance [km] along a track
        # defined by lon, lat points (x and y in the arguments)
        earth_rad = zfun.earth_rad(np.mean(y)) # m
        xrad = np.pi * x /180
        yrad = np.pi * y / 180
        dx = earth_rad * np.cos(yrad[1:]) * np.diff(xrad)
        dy = earth_rad * np.diff(yrad)
        ddist = np.sqrt(dx**2 + dy**2)
        dist = np.zeros(len(x))
        dist[1:] = ddist.cumsum()/1000 # km
        return dist
    
    dist_e = get_dist(x_e,y_e) # cell edges
    dist = get_dist(x,y) # cell centers

    # Make dist_e into a 2-D array for plotting.
    dist_se = np.ones((N,1)) * dist_e.reshape((1,-1))
    # the -1 means infer size from the array

    def get_sect(x, y, fld, lon, lat):
        # Interpolate a 2-D or 3-D field along a 2-D track
        # defined by lon and lat vectors x and y.
        # We assume that the lon, lat arrays are plaid.
        col0, col1, colf = zfun.get_interpolant(x, lon[1,:])
        row0, row1, rowf = zfun.get_interpolant(y, lat[:,1])
        colff = 1 - colf
        rowff = 1 - rowf
        if len(fld.shape) == 3:
            fld_s = (rowff*(colff*fld[:, row0, col0] + colf*fld[:, row0, col1])
                + rowf*(colff*fld[:, row1, col0] + colf*fld[:, row1, col1]))
        elif len(fld.shape) == 2:
            fld_s = (rowff*(colff*fld[row0, col0] + colf*fld[row0, col1])
                + rowf*(colff*fld[row1, col0] + colf*fld[row1, col1]))
        return fld_s

    # Do the section extractions for zw (edges) and sv (centers)
    zw_se = get_sect(x_e, y_e, zw, lon, lat)
    fld_s = get_sect(x, y, fld, lon, lat )

    # Also generate top and bottom lines, with appropriate masking
    zbot = -get_sect(x, y, h, lon, lat )
    ztop = get_sect(x, y, zeta, lon, lat )
    zbot[np.isnan(fld_s[-1,:])] = 0
    ztop[np.isnan(fld_s[-1,:])] = 0
    
    return x, y, dist, dist_e, zbot, ztop, dist_se, zw_se, fld_s, lon, lat


# %%

for seg_name in seg_list:
    
    if 'G1' in seg_name:
            
        jjj = j_dict[seg_name]
        iii = i_dict[seg_name]
    
import pinfo
from importlib import reload
reload(pinfo)

min_lat = [48, 48.4]
max_lat = [49, 48.7]
min_lon = [-124, -123.4]
max_lon = [-122.25,-122.4]



min_lon_sect = Lon[min(iii)]
max_lon_sect = Lon[max(iii)]
max_lat_sect = Lat[max(jjj)]

vn = 'oxygen'

x_e = np.linspace(min_lon_sect, max_lon_sect, 10000)
y_e = max_lat_sect * np.ones(x_e.shape)

cmap=pinfo.cmap_dict[vn]


for (mon_num, mon_str) in zip(month_num,month_str):

    dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
    fn = cfun.get_his_fn_from_dt(Ldir, dt)
    
    x, y, dist, dist_e, zbot, ztop, dist_se, zw_se, fld_s, lon, lat = get_sect(fn, vn, x_e, y_e)
    
    ds = xr.open_dataset(fn)

    plt.close('all')
    pfun.start_plot(figsize=(14,8))
    fig = plt.figure()
    #cmap = 'jet'
    
    # map with section line
    ax = fig.add_subplot(2,1,1)
    plon, plat = pfun.get_plon_plat(lon,lat)
    cs = ax.pcolormesh(plon,plat,ds[vn][0,-1,:,:]*32/1000,
               cmap=pinfo.cmap_dict[vn], vmin = 0, vmax = 12)
    fig.colorbar(cs, ax=ax, label = 'DO [mg/L]')
    pfun.add_coast(ax)
    aaf = [min_lon[1], max_lon[1], min_lat[1], max_lat[1]] # focus domain
    ax.axis(aaf)
    pfun.dar(ax)
    # try:
    #     units_str = ds[vn].units
    #     ax.set_title('Field = %s [%s]' % (vn, units_str))
    # except AttributeError:
    #     ax.set_title('Field = %s' % (vn))
    # add section track
    ax.plot(x, y, '-k', linewidth=2)
    ax.plot(x[0], y[0], 'ok', markersize=10, markerfacecolor='w')
    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_title('DO Transect ('+mon_str+')')
    
    # section
    ax = fig.add_subplot(2,1,2)
    ax.plot(dist, zbot, '-k', linewidth=2)
    ax.plot(dist, ztop, '-b', linewidth=1)
    ax.set_xlim(dist.min(), dist.max())
    ax.set_ylim(-350, 5)
    cs = ax.pcolormesh(dist_se,zw_se,fld_s*32/1000,
               cmap=pinfo.cmap_dict[vn], vmin = 0, vmax =12)
    fig.colorbar(cs, ax=ax, label = 'DO [mg/L]')
    ax.set_xlabel('Distance along Section [km]')
    ax.set_ylabel('Z [m]')
    
    #plt.title('DO Transect ('+mon_str+')')
    fig.tight_layout()
    
   # plt.show()
        
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/sect_LO_'+mon_str+'.png')

# %%
    
    
    
    
    
    
    
#     # PLOT CODE
#     vn = 'oxygen'#'phytoplankton'
#     # if vn == 'salt':
#     #     pinfo.cmap_dict[vn] = 'jet'
#     # GET DATA
#     G, S, T = zrfun.get_basic_info(fn_his)
#     # CREATE THE SECTION
#     # create track by hand
#     lon = G['lon_rho']
#     lat = G['lat_rho']
#     zdeep = -350 
#     x = np.linspace(min_lon_sect, max_lon_sect, 10000)
#     y = max_lat_sect * np.ones(x.shape)
    
#     v2, v3, dist, idist0 = pfun.get_section(ds, vn, x, y, in_dict)
    
#     # COLOR
#     # scaled section data 
#     sf = pinfo.fac_dict[vn] * v3['sectvarf']
#     # now we use the scaled section as the preferred field for setting the
#     # color limits of both figures in the case -avl True
#     # if in_dict['auto_vlims']:
#     #     pinfo.vlims_dict[vn] = pfun.auto_lims(sf)
    
#     # PLOTTING
#     # map with section line
    
    
    
#     plt.close('all')
    
#     fs = 14
#     pfun.start_plot(fs=fs, figsize=(20,9))
#     fig6 = plt.figure()
    
#     ax6 = fig6.add_subplot(1, 3, 1)
#     cs = pfun.add_map_field(ax6, ds, vn, pinfo.vlims_dict,
#             cmap=pinfo.cmap_dict[vn], fac=pinfo.fac_dict[vn], do_mask_edges=True)
#     # fig.colorbar(cs, ax=ax) # It is identical to that of the section
#     pfun.add_coast(ax6)
#     aaf = [min_lon[1], max_lon[1], min_lat[1], max_lat[1]] # focus domain
#     ax6.axis(aaf)
#     pfun.dar(ax6)
#     pfun.add_info(ax6, fn_his, loc='upper_right')
#     ax6.set_title('Surface %s %s' % (pinfo.tstr_dict[vn],pinfo.units_dict[vn]))
#     ax6.set_xlabel('Longitude')
#     ax6.set_ylabel('Latitude')
#     # add section track
#     ax6.plot(x, y, '-r', linewidth=2)
#     ax6.plot(x[idist0], y[idist0], 'or', markersize=5, markerfacecolor='w',
#         markeredgecolor='r', markeredgewidth=2)
#     #ax6.set_xticks([-125, -124, -123])
#     #ax6.set_yticks([47, 48, 49, 50])
#     # section
#     ax6 = fig6.add_subplot(1, 3, (2, 3))
#     ax6.plot(dist, v2['zbot'], '-k', linewidth=2)
#     ax6.plot(dist, v2['zeta'], '-b', linewidth=1)
#     ax6.set_xlim(dist.min(), dist.max())
#     ax6.set_ylim(zdeep, 5)
#     # plot section
#     svlims = pinfo.vlims_dict[vn]
#     cs = ax6.pcolormesh(v3['distf'], v3['zrf'], sf,
#                        vmin=svlims[0], vmax=svlims[1], cmap=pinfo.cmap_dict[vn])
#     fig6.colorbar(cs, ax=ax6)
#     ax6.set_xlabel('Distance (km)')
#     ax6.set_ylabel('Z (m)')
#     ax6.set_title('Section %s %s' % (pinfo.tstr_dict[vn],pinfo.units_dict[vn]))
#     fig6.tight_layout()
#     # FINISH
#     ds.close()
#     pfun.end_plot()
#     plt.savefig('/Users/dakotamascarenas/Desktop/pltz/sect_LO_'+mon_str+'.png')

