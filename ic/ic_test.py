"""
Code to explore an initial condition for LiveOcean.
"""

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
import matplotlib.pyplot as plt
import matplotlib.path as mpth
import xarray as xr
import numpy as np
import pandas as pd

Ldir = Lfun.Lstart(gridname='cas7')

vn_dict = {
    'salt':'SA',
    'temp':'CT',
    'oxygen':'DO (uM)',
    'NO3':'NO3 (uM)'
}

# select variable to plot (ROMS name)
vn = 'NO3'

# ocean_ini from Dakota
fn = (Ldir['LOo'] / 'forcing' / Ldir['gridname'] / 'f2013.01.01' / 'ocn00' / 'ocean_ini.nc')
ds = xr.open_dataset(fn)
v = ds[vn][0,-1,:,:].values
vv = ds[vn][0,:,:,:].values

# grid
fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)
x = dsg.lon_rho.values
y = dsg.lat_rho.values
m = dsg.mask_rho.values
xp, yp = pfun.get_plon_plat(x,y)
h = dsg.h.values
h[m==0] = np.nan

# z array
fngs = Ldir['grid'] / 'S_COORDINATE_INFO.csv'
s_dict = Lfun.csv_to_dict(fngs)
S = zrfun.get_S(s_dict)
zz = zrfun.get_z(h, 0*h, S, only_rho=True)

# polygon
fnp = Ldir['LOo'] / 'section_lines' / 'poly_sog.p'
p = pd.read_pickle(fnp)
xx = p.x.to_numpy()
yy = p.y.to_numpy()
xxyy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
path = mpth.Path(xxyy)
# make Boolean array (npoints) of which are inside
xy = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)
isin = path.contains_points(xy)
isina = isin.reshape(x.shape)

# select ic that is inside path (for map plot)
v[~isina] = np.nan

# observations
odir = Ldir['LOo'] / 'obs' / 'dfo1' / 'bottle'
ii = 0
for year in [2012, 2013]:
    if ii == 0:
        odf = pd.read_pickle( odir / (str(year) + '.p'))
    else:
        this_odf = pd.read_pickle( odir / (str(year) + '.p'))
        odf = pd.concat((odf,this_odf))
    ii += 1

if False:
    # limit time range
    ti = pd.DatetimeIndex(odf.time)
    mo = ti.month
    mo_mask = mo==0 # initialize all false
    for imo in [11,12,1,2]:
        mo_mask = mo_mask | (mo==imo)
    odf = odf.loc[mo_mask,:]
# get lon lat of remaining obs
ox = odf.lon.to_numpy()
oy = odf.lat.to_numpy()
oxoy = np.concatenate((ox.reshape(-1,1),oy.reshape(-1,1)), axis=1)

# get all profiles insde the polygon

# (i) from the ic file
vvv = vv[:,isina]
zzz = zz[:,isina]
xxx = x[isina]
yyy = y[isina]
# Note: using the 2d boolean array "isina" reduces the dimension
# of vv from 3d to 2d, and x from 2d to 1d. No need to flatten.

# (ii) from the observations
oisin = path.contains_points(oxoy)
odfin = odf.loc[oisin,:]
ovn = vn_dict[vn]
ov = odfin[ovn].to_numpy()
oz = odfin.z.to_numpy()
oxin = odfin.lon.to_numpy()
oyin = odfin.lat.to_numpy()

# PLOTTING

plt.close('all')
pfun.start_plot(figsize=(12,8))
fig = plt.figure()

# map
ax = fig.add_subplot(121)
cs = ax.pcolormesh(xp,yp,v)
pfun.add_coast(ax)
pfun.dar(ax)
aa = [-125.5, -122, 47, 50.5]
ax.axis(aa)
fig.colorbar(cs, ax=ax)
ax.set_title(vn)
# polygon
ax.plot(xx,yy,'-*r', linewidth=3)
# ax.plot(xxx,yyy,'.k', alpha = .1)
ax.plot(oxin,oyin,'ow',mec='k')

# profiles
ax = fig.add_subplot(122)
ax.plot(vvv.flatten(), zzz.flatten(),'.b')
ax.plot(ov, oz, '.r',alpha=.2)

plt.show()