"""
Test of nearest-neighbor extrapolation to fill missing (masked) values
on a plaid grid.

Test on mac in ipython:
run test_KDTree_DM -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test True

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

Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

dt = pd.Timestamp('2022-11-30 01:30:00')
fn = cfun.get_his_fn_from_dt(Ldir, dt)

#grid info
G, S, T = zrfun.get_basic_info(fn)
Lon = G['lon_rho'][0,:]
Lat = G['lat_rho'][:,0]

# get segment info
vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
v_df = pd.read_pickle(vol_dir / 'volumes.p')
j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
seg_list = list(v_df.index)

info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')

for seg_name in seg_list:
    
    if 'G1' in seg_name:
            
        jjj = j_dict[seg_name]
        iii = i_dict[seg_name]
        
        # i_ = np.unique(iii)
        # j_ = np.unique(jjj)
        
        # m,n = np.meshgrid(i_,j_)
        
        info_df = pd.read_pickle(info_fn)
        
        ij_ = []
        ii_ = []

        for cid in info_df.index:
            
            lon = info_df.loc[cid, 'lon']
            lat = info_df.loc[cid, 'lat']
            
            ij = zfun.find_nearest_ind(Lat, lat)
            ii = zfun.find_nearest_ind(Lon, lon)
            
            if (ii in iii) and (ij in jjj):
                
                ij_.append(ij)
                ii_.append(ii)
                               
        i_cast = list(map(list, zip(*list(set(zip(ii_,ij_))))))
        
        ii_cast = i_cast[0]
        
        ij_cast = i_cast[1]
        
        
xx = np.arange(min(iii), max(iii)+1)
yy = np.arange(min(jjj), max(jjj)+1)
x, y = np.meshgrid(xx, yy)
        
a = np.full([len(yy),len(xx)], -99)
a[jjj-min(jjj),iii-min(iii)] = -1
a = np.ma.masked_array(a,a==-99)

b = a.copy()
for n in range(len(ij_cast)):
    b[ij_cast[n]-min(jjj), ii_cast[n]-min(iii)] = n
    
c = b.copy()
c = np.ma.masked_array(c,c==-1)
    
xy_water = np.array((x[~a.mask],y[~a.mask])).T
xy_land = np.array((x[a.mask],y[a.mask])).T

xy_casts = np.array((x[~c.mask],y[~c.mask])).T

tree = KDTree(xy_casts)

tree_query = tree.query(xy_water)[1]

d = a.copy()
d[~a.mask] = b[~c.mask][tree_query]


#hacky tef segment implementation
sect_df = tfun.get_sect_df('cas6')

#min_lon = sect_df.at['sog5','x0']
#max_lon = sect_df.at['sog1','x1']
#min_lat = sect_df.at['sog1','y0']
#max_lat = sect_df.at['sog5','y0']

min_lat = [48, 48.4]
max_lat = [49, 48.7]
min_lon = [-124, -123.4]
max_lon = [-122.25,-122.4]

# tt0 = time()
# x = []; y = []
# s0 = []; s1 = []
# t0 = []; t1 = []
# z_rho = []
# oxygen = []
# for fn in fn_list:
#     ds = xr.open_dataset(fn)
#     x.append(ds.lon_rho.values)
#     y.append(ds.lat_rho.values)
#     z_rho.append(ds.z_rho.values)
#     oxygen.append(ds.oxygen.values*32/1000) #mg/L
#     s0.append(ds.salt[0].values)
#     t0.append(ds.temp[0].values)
#     s1.append(ds.salt[-1].values)
#     t1.append(ds.temp[-1].values)
#     ds.close()
# print('Took %0.2f sec' % (time()-tt0))

cmap = cm.get_cmap('viridis', len(ii_cast))

plt.close('all')
pfun.start_plot(fs=14, figsize=(20,15))
fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
axes[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d)
for m in range(len(ii_cast)):
    axes[0,0].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
axes[0,0].set_xlim([min_lon[0],max_lon[0]])
axes[0,0].set_ylim([min_lat[0],max_lat[0]])
axes[0,0].tick_params(labelrotation=45)
pfun.add_coast(axes[0,0])
pfun.dar(axes[0,0])
axes[0,1].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d)
for m in range(len(ii_cast)):
    axes[0,1].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
axes[0,1].set_xlim([min_lon[1],max_lon[1]])
axes[0,1].set_ylim([min_lat[1],max_lat[1]])
axes[0,1].tick_params(labelrotation=45)
pfun.add_coast(axes[0,1])
pfun.dar(axes[0,1])
fig.tight_layout()
plt.show()
