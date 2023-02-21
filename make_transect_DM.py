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



for (mon_num, mon_str) in zip(month_num,month_str):

    dt = pd.Timestamp('2022-' + mon_num +'-01 01:30:00')
    fn_his = cfun.get_his_fn_from_dt(Ldir, dt)
    
    in_dict = {}

    in_dict['fn'] = fn_his
    
    
    ds = xr.open_dataset(fn_his)
    # PLOT CODE
    vn = 'oxygen'#'phytoplankton'
    # if vn == 'salt':
    #     pinfo.cmap_dict[vn] = 'jet'
    # GET DATA
    G, S, T = zrfun.get_basic_info(fn_his)
    # CREATE THE SECTION
    # create track by hand
    lon = G['lon_rho']
    lat = G['lat_rho']
    zdeep = -350 
    x = np.linspace(min_lon_sect, max_lon_sect, 10000)
    y = max_lat_sect * np.ones(x.shape)
    
    v2, v3, dist, idist0 = pfun.get_section(ds, vn, x, y, in_dict)
    
    # COLOR
    # scaled section data 
    sf = pinfo.fac_dict[vn] * v3['sectvarf']
    # now we use the scaled section as the preferred field for setting the
    # color limits of both figures in the case -avl True
    # if in_dict['auto_vlims']:
    #     pinfo.vlims_dict[vn] = pfun.auto_lims(sf)
    
    # PLOTTING
    # map with section line
    
    
    
    plt.close('all')
    
    fs = 14
    pfun.start_plot(fs=fs, figsize=(20,9))
    fig6 = plt.figure()
    
    ax6 = fig6.add_subplot(1, 3, 1)
    cs = pfun.add_map_field(ax6, ds, vn, pinfo.vlims_dict,
            cmap=pinfo.cmap_dict[vn], fac=pinfo.fac_dict[vn], do_mask_edges=True)
    # fig.colorbar(cs, ax=ax) # It is identical to that of the section
    pfun.add_coast(ax6)
    aaf = [min_lon[1], max_lon[1], min_lat[1], max_lat[1]] # focus domain
    ax6.axis(aaf)
    pfun.dar(ax6)
    pfun.add_info(ax6, fn_his, loc='upper_right')
    ax6.set_title('Surface %s %s' % (pinfo.tstr_dict[vn],pinfo.units_dict[vn]))
    ax6.set_xlabel('Longitude')
    ax6.set_ylabel('Latitude')
    # add section track
    ax6.plot(x, y, '-r', linewidth=2)
    ax6.plot(x[idist0], y[idist0], 'or', markersize=5, markerfacecolor='w',
        markeredgecolor='r', markeredgewidth=2)
    #ax6.set_xticks([-125, -124, -123])
    #ax6.set_yticks([47, 48, 49, 50])
    # section
    ax6 = fig6.add_subplot(1, 3, (2, 3))
    ax6.plot(dist, v2['zbot'], '-k', linewidth=2)
    ax6.plot(dist, v2['zeta'], '-b', linewidth=1)
    ax6.set_xlim(dist.min(), dist.max())
    ax6.set_ylim(zdeep, 5)
    # plot section
    svlims = pinfo.vlims_dict[vn]
    cs = ax6.pcolormesh(v3['distf'], v3['zrf'], sf,
                       vmin=svlims[0], vmax=svlims[1], cmap=pinfo.cmap_dict[vn])
    fig6.colorbar(cs, ax=ax6)
    ax6.set_xlabel('Distance (km)')
    ax6.set_ylabel('Z (m)')
    ax6.set_title('Section %s %s' % (pinfo.tstr_dict[vn],pinfo.units_dict[vn]))
    fig6.tight_layout()
    # FINISH
    ds.close()
    pfun.end_plot()
    plt.savefig('/Users/dakotamascarenas/Desktop/pltz/sect_LO_'+mon_str+'.png')

