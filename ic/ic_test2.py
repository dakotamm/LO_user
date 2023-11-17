"""
Code to explore an initial condition for LiveOcean. This is focused
just on observations, with the goal of coming up with reasonable values
for all tracers by basin/depth.
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

Ldir = Lfun.Lstart(gridname='cas7')

# grid
fng = Ldir['grid'] / 'grid.nc'
dsg = xr.open_dataset(fng)
x = dsg.lon_rho.values
y = dsg.lat_rho.values
m = dsg.mask_rho.values
xp, yp = pfun.get_plon_plat(x,y)
h = dsg.h.values
h[m==0] = np.nan

# polygons
basin_list = ['sog', 'jdf', 'ps','hc']
c_list = ['r','b','g','c'] # colors to associate with basins
c_dict = dict(zip(basin_list,c_list))

path_dict = dict()
xxyy_dict = dict()
for basin in basin_list:
    # polygon
    fnp = Ldir['LOo'] / 'section_lines' / ('poly_'+basin+'.p')
    p = pd.read_pickle(fnp)
    xx = p.x.to_numpy()
    yy = p.y.to_numpy()
    xxyy = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)), axis=1)
    path = mpth.Path(xxyy)
    # store in dicts
    path_dict[basin] = path
    xxyy_dict[basin] = xxyy

# observations
source_list = ['dfo1', 'ecology', 'nceiSalish']
year_list = [2013,2014,2015,2016,2017]
ii = 0
for year in year_list:
    for source in source_list:
        odir = Ldir['LOo'] / 'obs' / source / 'bottle'
        try:
            if ii == 0:
                odf = pd.read_pickle( odir / (str(year) + '.p'))
                # print(odf.columns)
            else:
                this_odf = pd.read_pickle( odir / (str(year) + '.p'))
                # print(this_odf.columns)
                odf = pd.concat((odf,this_odf),ignore_index=True)
            ii += 1
        except FileNotFoundError:
            pass

if True:
    # limit time range
    ti = pd.DatetimeIndex(odf.time)
    mo = ti.month
    mo_mask = mo==0 # initialize all false
    for imo in [9,10,11]:
        mo_mask = mo_mask | (mo==imo)
    odf = odf.loc[mo_mask,:]
    
# get lon lat of (remaining) obs
ox = odf.lon.to_numpy()
oy = odf.lat.to_numpy()
oxoy = np.concatenate((ox.reshape(-1,1),oy.reshape(-1,1)), axis=1)


# get all profiles inside each polygon
odf_dict = dict()
for basin in basin_list:
    path = path_dict[basin]
    oisin = path.contains_points(oxoy)
    odfin = odf.loc[oisin,:]
    odf_dict[basin] = odfin.copy()
    

# PLOTTING

vn_list = ['SA', 'CT', 'DO (uM)', 'NO3 (uM)', 'DIC (uM)', 'TA (uM)']
n_list = [1,2,4,5,7,8]
n_dict = dict(zip(vn_list,n_list))
vn_dict = dict(zip(n_list,vn_list))

plt.close('all')
pfun.start_plot(figsize=(12,8))
fig = plt.figure()

# set up axes for profiles
ax_dict = dict()
for n in n_list:
    ax = fig.add_subplot(3,3,n)
    ax_dict[n] = ax
    ax.text(.05,.05,vn_dict[n],transform=ax.transAxes)

# map
ax = fig.add_subplot(133)
pfun.add_coast(ax)
pfun.dar(ax)
aa = [-125.5, -122, 46.5, 50.5]
ax.axis(aa)
# polygons
for basin in basin_list:
    oo = odf_dict[basin]
    oxin = oo.lon.to_numpy()
    oyin = oo.lat.to_numpy()
    ax.plot(oxin,oyin,'o',color=c_dict[basin],ms=3)
    
    xxyy = xxyy_dict[basin]
    ax.plot(xxyy[:,0],xxyy[:,1],'-*',color=c_dict[basin], linewidth=3)

# profiles
zi_list = [.9,.8,.7,.6] # z position for means
zi_dict = dict(zip(basin_list,zi_list))
for basin in basin_list:
    oo = odf_dict[basin]
    for vn in vn_list:
        try:
            ov = oo[vn].to_numpy()
            oz = oo.z.to_numpy()
            n = n_dict[vn]
            ax = ax_dict[n]
            ax.plot(ov, oz, '.',alpha=.2,color=c_dict[basin])
            # add text averages
            zdiv = -25
            try:
                vtop = np.nanmean(ov[oz>=zdiv])
                vbot = np.nanmean(ov[oz<=zdiv])
                ax.text(.05,zi_dict[basin],'%d/%d' % (int(vtop),int(vbot)),
                    transform=ax.transAxes,color=c_dict[basin])
            except ValueError:
                pass
        except KeyError:
            pass

plt.show()