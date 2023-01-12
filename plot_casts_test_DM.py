"""
Code to plot the results of a cast extraction.

run plot_casts_test_DM -gtx cas6_v0_live -source dfo -otype ctd -year 2019 -test False
"""

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
from lo_tools import extract_argfun as exfun
Ldir = exfun.intro() # this handles the argument passing

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
import tef_fun as tfun

year_str = str(Ldir['year'])

#hacky tef segment implementation
sect_df = tfun.get_sect_df('cas6')

min_lon = sect_df.at['sog5','x0']
max_lon = sect_df.at['sog1','x1']
min_lat = sect_df.at['sog1','y0']
max_lat = sect_df.at['sog5','y0']

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
    x.append(ds.lon_rho.values)
    y.append(ds.lat_rho.values)
    z_rho.append(ds.z_rho.values)
    oxygen.append(ds.oxygen.values*32/1000) #mg/L
    s0.append(ds.salt[0].values)
    t0.append(ds.temp[0].values)
    s1.append(ds.salt[-1].values)
    t1.append(ds.temp[-1].values)
    ds.close()
print('Took %0.2f sec' % (time()-tt0))

cmap = cm.get_cmap('tab20b', 20)

plt.close('all')
pfun.start_plot(fs=14, figsize=(14,10))
fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)
for i in range(len(fn_list)):
    axes[0,0].plot(x[i],y[i],'o',c=cmap(i))
axes[0,0].set_xlim([min_lon,max_lon])
axes[0,0].set_ylim([min_lat,max_lat])
depth = axes[0,1].plot(s0,t0,'.', c='orange', label ='depth') # FIX!!!!!!!!
surface = axes[0,1].plot(s1,t1,'.', c='dodgerblue', label = 'surface')
axes[0,1].legend(handles = [depth,surface])
pfun.add_coast(axes[0,0])
pfun.dar(axes[0,0])
axes[0,1].set_xlabel('Salinity [g/kg]')
axes[0,1].set_ylabel('Potential Temperature [deg C]')
axes[0,0].set_title(in_dir.name)
for i in range(len(fn_list)):
    axes[0,2].plot(oxygen[i],z_rho[i],'.',c=cmap(i))
axes[0,2].set_xlabel('Oxygen Concentration [mg/L]')
axes[0,2].set_ylabel('Depth [m]')
axes[0,0].set_title(in_dir.name)
fig.tight_layout()
plt.show()
pfun.end_plot()
