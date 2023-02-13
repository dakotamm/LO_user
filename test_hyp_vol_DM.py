"""
Test finding hypoxic depth and volume.

Test on mac in ipython:
run test_hyp_vol_DM -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test True

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

Ldir = exfun.intro() # this handles the argument passing

year_str = str(Ldir['year'])

dt = pd.Timestamp('2022-11-30 01:30:00')
fn = cfun.get_his_fn_from_dt(Ldir, dt)

#grid info
G, S, T = zrfun.get_basic_info(fn)
Lon = G['lon_rho'][0,:]
Lat = G['lat_rho'][:,0]
z_rho_grid, z_w_grid = zrfun.get_z(G['h'], 0*G['h'], S)
dz = np.diff(z_w_grid,axis=0)
dv = dz*G['DX']*G['DY']

# get segment info
vol_dir = Ldir['LOo'] / 'extract' / 'tef' / ('volumes_' + Ldir['gridname'])
v_df = pd.read_pickle(vol_dir / 'volumes.p')
j_dict = pickle.load(open(vol_dir / 'j_dict.p', 'rb'))
i_dict = pickle.load(open(vol_dir / 'i_dict.p', 'rb'))
seg_list = list(v_df.index)

#this is very hacky below - but works for now

info_fn = Ldir['LOo'] / 'obs' / Ldir['source'] / Ldir['otype'] / ('info_' + year_str + '.p')

in_dir = (Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'cast' /
    (Ldir['source'] + '_' + Ldir['otype'] + '_' + year_str))

fn_list = list(in_dir.glob('*.nc'))

foo = xr.open_dataset(fn_list[0])
for vn in foo.data_vars:
    print('%14s: %s' % (vn, str(foo[vn].shape)))
foo.close()

tt0 = time()
x = []; y = []
s0 = []; s1 = []
t0 = []; t1 = []
z_rho = []
oxygen = []
for fn in fn_list:
    ds = xr.open_dataset(fn)
   # x.append(ds.lon_rho.values)
   # y.append(ds.lat_rho.values)
    z_rho.append(ds.z_rho.values)
    oxygen.append(ds.oxygen.values*32/1000) #mg/L
    s0.append(ds.salt[0].values)
    t0.append(ds.temp[0].values)
    s1.append(ds.salt[-1].values)
    t1.append(ds.temp[-1].values)
    ds.close()
print('Took %0.2f sec' % (time()-tt0))

for seg_name in seg_list:
    
    if 'G1' in seg_name:
            
        jjj = j_dict[seg_name]
        iii = i_dict[seg_name]
        
        info_df = pd.read_pickle(info_fn)
        
        ij_ = []
        ii_ = []
        

        for cid in info_df.index:
            
            if info_df.loc[cid,'time'] >= datetime(2019,6,1) and info_df.loc[cid,'time'] <= datetime(2019,8,31):
            
                lon = info_df.loc[cid, 'lon']
                lat = info_df.loc[cid, 'lat']
                
                ij = zfun.find_nearest_ind(Lat, lat)
                ii = zfun.find_nearest_ind(Lon, lon)
                
                if (ii in iii) and (ij in jjj):
                    
                    ij_.append(ij)
                    ii_.append(ii)
                                   
            ii_cast = ii_
            ij_cast = ij_
        
        
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


sect_df = tfun.get_sect_df('cas6')

min_lat = [48, 48.4]
max_lat = [49, 48.7]
min_lon = [-124, -123.4]
max_lon = [-122.25,-122.4]


cmap = cm.get_cmap('viridis', len(ii_cast))


plt.close('all')
pfun.start_plot(fs=14, figsize=(20,15))
fig0, axes0 = plt.subplots(nrows=1, ncols=2, squeeze=False)
axes0[0,0].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d)
for m in range(len(ii_cast)):
    axes0[0,0].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
axes0[0,0].set_xlim([min_lon[0],max_lon[0]])
axes0[0,0].set_ylim([min_lat[0],max_lat[0]])
axes0[0,0].tick_params(labelrotation=45)
pfun.add_coast(axes0[0,0])
pfun.dar(axes0[0,0])
axes0[0,1].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d)
for m in range(len(ii_cast)):
    axes0[0,1].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=cmap(m),markeredgecolor='black', markersize=10)
axes0[0,1].set_xlim([min_lon[1],max_lon[1]])
axes0[0,1].set_ylim([min_lat[1],max_lat[1]])
axes0[0,1].tick_params(labelrotation=45)
pfun.add_coast(axes0[0,1])
pfun.dar(axes0[0,1])
fig0.tight_layout()
plt.show()

pfun.start_plot(fs=14, figsize=(15,10))
fig1, axes1 = plt.subplots(nrows=1, ncols=1, squeeze=False)
for i in range(len(fn_list)):
    axes1[0,0].plot(oxygen[i],z_rho[i],'o',c=cmap(i))
axes1[0,0].set_xlabel('Oxygen Concentration [mg/L]')
axes1[0,0].set_ylabel('Depth [m]')
fig1.tight_layout()
plt.show()

pfun.start_plot(fs=14, figsize=(15,10))
fig2, axes2 = plt.subplots(nrows=1, ncols=1, squeeze=False)
for m in range(len(ii_cast)):
    axes2[0,0].plot(Lon[ii_cast[m]],Lat[ij_cast[m]],'o',c=new_cmap(m),markeredgecolor='black', markersize=10)
axes2[0,0].set_xlim([min_lon[0],max_lon[0]])
axes2[0,0].set_ylim([min_lat[0],max_lat[0]])
axes2[0,0].tick_params(labelrotation=45)
pfun.add_coast(axes2[0,0])
pfun.dar(axes0[0,0])
axes0[0,1].pcolormesh(Lon[np.unique(iii)],Lat[np.unique(jjj)],d)
fig1.tight_layout()
plt.show()

cast_no = list(map(str, np.arange(len(fn_list))))

df0 = pd.DataFrame(oxygen)
df0['cast_no'] = cast_no
df0 = pd.melt(df0, id_vars=['cast_no'], var_name='sigma_level', value_name='DO')

df1 = pd.DataFrame(z_rho)
df1['cast_no'] = cast_no
df1 = pd.melt(df1, id_vars=['cast_no'], var_name='sigma_level', value_name='z_rho')

df = df0.merge(df1,on=['cast_no','sigma_level'])

df_hyp_test = df[df['DO'] < 6.2]

hyp_casts = df_hyp_test['cast_no'].unique()

hyp_casts = [int(l) for l in hyp_casts]

max_z_hyp = []

iii_hyp = []
jjj_hyp = []

for m in range(len(hyp_casts)):
    df_temp = df_hyp_test[df_hyp_test['cast_no']==hyp_casts[m]]
    max_z_hyp.append(df_temp['z_rho'].max())
    d_hyp = np.where(d == hyp_casts[m])
    jjj_hyp.append(d_hyp[0]+min(jjj))
    iii_hyp.append(d_hyp[1]+min(iii))
    
# hacky af
    
# e= d.copy()

# f= d.copy()

# g = d.copy()

# e = d[d != 3 or d != 1] = -1

#f = np.ma.masked_array(f, f !=3)

#g = g[e.mask]

#g = g[f.mask]
    
    
    