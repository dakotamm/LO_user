"""
Daily wind over the Penn Cove segments, to test whether it drives the gyre.

The residual circulation across the Penn Cove sections is lateral -- in along
the north shore, out along the south -- and it does NOT follow the tidal prism
(flat to 6% while the gyre doubles) or the along-cove salinity gradient. Wind
is the remaining candidate, and this pulls it out of the model's own forcing so
the comparison is against exactly what the run felt.

Uwind and Vwind sit on the rho grid in the avg files, so this averages them
over the cells of the named segments and writes a daily series.

Also saved rotated into the cove's own frame:
    w_along  positive toward the head, i.e. blowing INTO the cove
    w_cross  positive toward the north shore
The cove axis is taken as the line joining the centroids of the innermost and
outermost segments, so "along" means along the actual basin rather than due
east-west.

Output: LO_output/extract/<gtagex>/tef2/wind_<ds0>_<ds1>_<gctag>.nc, a few kB.

On apogee:
python reduce_wind_cove.py -gtx wb1_t0_xn11abbur00 -ro 2 -ctag pc1 -riv trapsN00 -0 2024.01.01 -1 2025.12.31 -his_num 1
"""
import pickle
import sys
from time import time

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun

Ldir = exfun.intro()

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'

seg_fn = tef2_dir / ('seg_info_dict_' + gctag + '_' + Ldir['riv'] + '.p')
if not seg_fn.is_file():
    alt = sorted(tef2_dir.glob('seg_info_dict_' + gctag + '_*.p'))
    if len(alt) == 0:
        raise FileNotFoundError('no seg_info_dict for %s' % gctag)
    print('using %s' % alt[0].name)
    seg_fn = alt[0]
seg_info = pickle.load(open(seg_fn, 'rb'))

COVE = ['pc_cp_m', 'pc_cp_p', 'pc_lp_m']
ji = np.concatenate([np.array(seg_info[s]['ji_list']) for s in COVE])
jj, ii = ji[:, 0], ji[:, 1]

out_dir = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
Lfun.make_dir(out_dir)
out_fn = out_dir / ('wind_' + Ldir['ds0'] + '_' + Ldir['ds1'] + '_' + gctag + '.nc')

fn_list = Lfun.get_fn_list('hourly', Ldir, Ldir['ds0'], Ldir['ds1'],
                           his_num=Ldir['his_num'])
print('%d files, first %s' % (len(fn_list), fn_list[0]))

# cove axis from the segment centroids, inner to outer
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon = g.lon_rho.values
lat = g.lat_rho.values
g.close()


def centroid(s):
    a = np.array(seg_info[s]['ji_list'])
    return lon[a[:, 0], a[:, 1]].mean(), lat[a[:, 0], a[:, 1]].mean()


x_in, y_in = centroid('pc_cp_m')
x_out, y_out = centroid('pc_lp_m')
# unit vector pointing from the mouth toward the head, in metres
ax = (x_in - x_out) * np.cos(np.deg2rad(y_in)) * 111e3
ay = (y_in - y_out) * 111e3
norm = np.hypot(ax, ay)
ax, ay = ax / norm, ay / norm
print('cove axis (mouth -> head) unit vector: (%.3f, %.3f)' % (ax, ay))

tt0 = time()
ot, U, Vv = [], [], []
for k, fn in enumerate(fn_list):
    ds = xr.open_dataset(fn)
    ot.append(ds.ocean_time.values[0])
    U.append(float(np.nanmean(ds.Uwind.isel(ocean_time=0).values[jj, ii])))
    Vv.append(float(np.nanmean(ds.Vwind.isel(ocean_time=0).values[jj, ii])))
    ds.close()
    if np.mod(k, 2000) == 0 and k > 0:
        el = time() - tt0
        print('  %6d / %d  %.1f min elapsed' % (k, len(fn_list), el / 60))
        sys.stdout.flush()

t = pd.to_datetime(ot)
df = pd.DataFrame({'Uwind': U, 'Vwind': Vv}, index=t)
# positive w_along blows from the mouth toward the head, i.e. into the cove
df['w_along'] = df.Uwind * ax + df.Vwind * ay
df['w_cross'] = -df.Uwind * ay + df.Vwind * ax
daily = df.resample('1D').mean()

print('\ndaily wind over the cove:')
print(daily.describe().round(3).to_string())

out = xr.Dataset.from_dataframe(daily.rename_axis('day'))
out.attrs['note'] = ('daily mean wind over the Penn Cove segments; w_along '
                     'positive INTO the cove (mouth -> head), w_cross '
                     'positive toward the north shore')
out.attrs['cove_axis'] = '(%.4f, %.4f)' % (ax, ay)
out.to_netcdf(out_fn)
print('\nelapsed %.1f min' % ((time() - tt0) / 60))
print('saved %s' % out_fn)
