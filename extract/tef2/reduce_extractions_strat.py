"""
Reduce extractions_avg to hourly top-minus-bottom salinity at each section.

extract_sections_avg.py saves salt(time, z, p) for each section: 30 sigma
levels by however many u/v faces the section has, hourly. That is the only
place in the tef2 chain where VERTICAL structure survives -- bulk_avg layers
are salinity classes, not depths, and processed_avg has already been binned
into salinity space. So stratification has to come from here.

ROMS sigma index 0 is the bed and -1 is the surface (same convention used by
extract_segments_one_time.py when it grabs salt_surf).

Output per section, hourly:
    s_top    width-weighted salinity of the surface sigma level
    s_bot    width-weighted salinity of the bottom sigma level
    dstrat   s_bot - s_top, positive when stably stratified in salt
    t_top, t_bot   same for temperature, if it was extracted

Faces are weighted by dd, their width, so a section's value is a genuine
cross-section mean rather than a mean over grid cells of unequal size.

Output is a few MB: LO_output/extract/<gtagex>/tef2/strat_<ds0>_<ds1>_<gctag>.nc

On apogee:
python reduce_extractions_strat.py -gtx wb1_t0_xn11abbur00 -ctag pc1 \
    -0 2024.01.01 -1 2025.12.31
"""
import sys
from time import time

import numpy as np
import xarray as xr

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun

Ldir = exfun.intro()

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
in_dir = out_dir0 / ('extractions_avg_' + Ldir['ds0'] + '_' + Ldir['ds1'])
out_fn = out_dir0 / ('strat_' + Ldir['ds0'] + '_' + Ldir['ds1'] + '_' + gctag + '.nc')

sect_list = sorted([f.name.replace('.nc', '') for f in in_dir.glob('*.nc')])
print('sections: ' + ', '.join(sect_list))

tt0 = time()
res = {}
for sn in sect_list:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    if 'time' not in res:
        res['time'] = ds.time.values
    dd = ds.dd.values                       # face widths, (p)
    w = dd / np.nansum(dd)
    d = {}
    for vn, tag in (('salt', 's'), ('temp', 't')):
        if vn not in ds.data_vars:
            continue
        top = ds[vn].values[:, -1, :]       # -1 is the surface
        bot = ds[vn].values[:, 0, :]        # 0 is the bed
        d[tag + '_top'] = np.nansum(top * w[None, :], axis=1)
        d[tag + '_bot'] = np.nansum(bot * w[None, :], axis=1)
    d['dstrat'] = d['s_bot'] - d['s_top']
    print('  %-10s n=%d  mean dstrat = %+.3f g/kg  (max %+.2f)'
          % (sn, len(d['dstrat']), np.nanmean(d['dstrat']), np.nanmax(d['dstrat'])))
    res[sn] = d
    ds.close()
    sys.stdout.flush()

print('\nelapsed %.1f sec' % (time() - tt0))

vlist = sorted(set().union(*[set(res[sn].keys()) for sn in sect_list]))
out = xr.Dataset(coords={'time': res['time'], 'sect': sect_list})
for v in vlist:
    out[v] = (('time', 'sect'),
              np.stack([res[sn].get(v, np.full(len(res['time']), np.nan))
                        for sn in sect_list], axis=1))
out['dstrat'].attrs['long_name'] = 'bottom minus surface salinity'
out['dstrat'].attrs['units'] = 'g kg-1'
out.attrs['note'] = ('hourly, NOT tidally filtered; width-weighted across '
                     'section faces; sigma index 0 = bed, -1 = surface')
out.to_netcdf(out_fn)
print('saved %s' % out_fn)
print(out)
