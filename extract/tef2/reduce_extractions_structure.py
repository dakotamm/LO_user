"""
Reduce extractions_avg to the residual circulation across each section.

extract_sections_avg.py stores q(time, z, p) and salt(time, z, p), where z is
the sigma level and p indexes the u/v FACES across the section. Averaging over
a long record kills the tidal oscillation and leaves the residual (subtidal)
circulation resolved in BOTH directions at once:

    qbar(z, p)     mean volume flux per face-cell   [m3 s-1]
    fsbar(z, p)    mean salt flux per face-cell     [g kg-1 m3 s-1]
    sbar(z, p)     mean salinity per face-cell      [g kg-1]

That is what distinguishes the two candidate explanations for Sin = Sout in
Penn Cove:

  vertical exchange -> qbar organises by z: one sign near the bed, the other
                       near the surface, and roughly uniform across p
  lateral exchange  -> qbar organises by p: one sign on one side of the cove,
                       the other on the other side, at all depths

Summing qbar over the other axis gives the two 1-D profiles to compare, and
the variance of each says which axis carries the circulation.

Also saved per month, since the gradients have a strong seasonal cycle.

Face geometry (lon, lat, h, dd) is carried through so the plots can use real
coordinates instead of face index.

Output is small: (z, p) is at most 30 x 32 per section.

On apogee:
python reduce_extractions_structure.py -gtx wb1_t0_xn11abbur00 -ctag pc1 \
    -0 2024.01.01 -1 2025.12.31
"""
import sys
from time import time

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun

Ldir = exfun.intro()

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
in_dir = out_dir0 / ('extractions_avg_' + Ldir['ds0'] + '_' + Ldir['ds1'])
out_fn = out_dir0 / ('structure_' + Ldir['ds0'] + '_' + Ldir['ds1']
                     + '_' + gctag + '.nc')

# the sect_df carries the face coordinates
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
sect_df = pd.read_pickle(tef2_dir / ('sect_df_' + gctag + '.p'))
grid = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon_rho = grid.lon_rho.values
lat_rho = grid.lat_rho.values
grid.close()

sect_list = sorted([f.name.replace('.nc', '') for f in in_dir.glob('*.nc')])
print('sections: ' + ', '.join(sect_list))

tt0 = time()
ds_out = xr.Dataset()
for sn in sect_list:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    t = pd.to_datetime(ds.time.values)
    q = ds.q.values                       # (time, z, p)
    sa = ds.salt.values
    NT, NZ, NP = q.shape

    qbar = np.nanmean(q, axis=0)
    fsbar = np.nanmean(q * sa, axis=0)
    sbar = np.nanmean(sa, axis=0)

    # monthly, to follow the seasonal cycle in the gradients
    qmon = np.full((12, NZ, NP), np.nan)
    for m in range(1, 13):
        sel = t.month == m
        if sel.any():
            qmon[m - 1] = np.nanmean(q[sel], axis=0)

    # DAILY lateral profile: depth-summed, one row per day. Twelve monthly
    # means is far too few points to attribute the gyre to a driver -- daily
    # gives n of order 700 and lets the correlation carry weight. Depth-summed
    # here because the lateral profile is all the gyre metric needs, and it
    # keeps the array small.
    day = pd.Series(np.arange(NT), index=t).groupby(t.floor('D'))
    dates = np.array(sorted(day.groups.keys()))
    qlat_day = np.full((len(dates), NP), np.nan)
    for k, dd in enumerate(dates):
        idx = day.get_group(dd).values
        qlat_day[k] = -np.nansum(np.nanmean(q[idx], axis=0), axis=0)

    # how much of the residual circulation lives on each axis: collapse to a
    # profile along one direction and take its variance
    prof_z = np.nansum(qbar, axis=1)      # (z) -- vertical structure
    prof_p = np.nansum(qbar, axis=0)      # (p) -- lateral structure
    print('  %-10s NZ=%d NP=%d  net=%+8.2f  |  sd(vertical profile)=%7.2f  '
          'sd(lateral profile)=%7.2f  lateral/vertical=%5.2f'
          % (sn, NZ, NP, np.nansum(qbar), np.nanstd(prof_z), np.nanstd(prof_p),
             np.nanstd(prof_p) / np.nanstd(prof_z)))

    d = sect_df[sect_df.sn == sn]
    lon_f = 0.5 * (lon_rho[d.jrp, d.irp] + lon_rho[d.jrm, d.irm])
    lat_f = 0.5 * (lat_rho[d.jrp, d.irp] + lat_rho[d.jrm, d.irm])

    pdim, zdim = sn + '_p', sn + '_z'
    ds_out[sn + '_qlat_day'] = (('day', pdim), qlat_day)
    if 'day' not in ds_out.coords:
        ds_out = ds_out.assign_coords(day=dates)
    ds_out[sn + '_qbar'] = ((zdim, pdim), qbar)
    ds_out[sn + '_fsbar'] = ((zdim, pdim), fsbar)
    ds_out[sn + '_sbar'] = ((zdim, pdim), sbar)
    ds_out[sn + '_qmon'] = (('month', zdim, pdim), qmon)
    ds_out[sn + '_lon'] = ((pdim), lon_f)
    ds_out[sn + '_lat'] = ((pdim), lat_f)
    ds_out[sn + '_h'] = ((pdim), ds.h.values)
    ds_out[sn + '_dd'] = ((pdim), ds.dd.values)
    ds.close()
    sys.stdout.flush()

ds_out = ds_out.assign_coords(month=np.arange(1, 13))
ds_out.attrs['note'] = ('residual (time-mean) circulation per face-cell from '
                        'extractions_avg; z index 0 = bed, -1 = surface; '
                        'p indexes faces across the section')
ds_out.to_netcdf(out_fn)
print('\nelapsed %.1f sec' % (time() - tt0))
print('saved %s' % out_fn)
