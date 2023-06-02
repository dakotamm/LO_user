import numpy as np
import xarray as xr
import pickle
from datetime import datetime, timedelta
import pandas as pd
from cmocean import cm
import sys

from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun
#import pinfo
from importlib import reload
reload(pfun)
#reload(pinfo)

import matplotlib.pyplot as plt


fn = '/Users/dakotamascarenas/LO_data/grids/cas6/grid.nc'

ds = xr.open_dataset(fn)

lon = ds.lon_psi.values

lat = ds.lat_psi.values

h1 = ds.h[1:-1,1:-1]

ax0 = plt.pcolormesh(lon,lat,-h1, vmin=-200, cmap = 'jet')

plt.colorbar()
#pfun.add_coast(ax)

plt.show()

#pfun.add_bathy_contours(ax, ds, txt=True)

#pfun.add_coast(ax)
