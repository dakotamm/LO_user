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
# Subsample in time. The target is a DAILY mean to correlate against daily gyre
# strength, and the cost here is dominated by the per-file open, not by the
# bytes. Every 6th hour still gives 4 samples a day, which resolves the diurnal
# sea breeze (4 per cycle, comfortably above Nyquist) at a sixth of the cost.
# Set -nskip 1 if you ever want the full hourly series.
NSKIP = 6
fn_list = fn_list[::NSKIP]
print('%d files (every %dth hour), first %s' % (len(fn_list), NSKIP, fn_list[0]))

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

# Read only a bounding box around the cove instead of the whole field. Indexing
# with .values[jj, ii] pulls the entire 368x272 grid off disk first; slicing
# lets netCDF read just the hyperslab, which is what made extract_segments_SV
# fast (0.054 s/file) while the first version of this script was not.
j0, j1 = jj.min(), jj.max() + 1
i0, i1 = ii.min(), ii.max() + 1
jl, il = jj - j0, ii - i0
sl = dict(eta_rho=slice(j0, j1), xi_rho=slice(i0, i1))
print('bounding box j %d:%d i %d:%d (%d of %d cells)'
      % (j0, j1, i0, i1, (j1 - j0) * (i1 - i0), 368 * 272))

tt0 = time()
ot, U, Vv = [], [], []
for k, fn in enumerate(fn_list):
    # decode_times=False skips the per-file CF/time decoding, which is pure
    # overhead when the cost is dominated by opening thousands of files
    ds = xr.open_dataset(fn, decode_times=False)
    ot.append(float(ds.ocean_time.values[0]))
    U.append(float(np.nanmean(ds.Uwind.isel(ocean_time=0, **sl).values[jl, il])))
    Vv.append(float(np.nanmean(ds.Vwind.isel(ocean_time=0, **sl).values[jl, il])))
    ds.close()
    if np.mod(k, 500) == 0 and k > 0:
        el = time() - tt0
        print('  %5d / %d  %.1f min elapsed, ~%.1f min left'
              % (k, len(fn_list), el / 60, el / 60 * (len(fn_list) - k) / k))
        sys.stdout.flush()

# Lfun.roms_time_units is 'seconds since 1970-01-01', i.e. plain Unix epoch, so
# pandas converts the whole array at once. Lfun.modtime_to_datetime is scalar
# only and would need a loop.
t = pd.to_datetime(np.array(ot), unit='s')
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
