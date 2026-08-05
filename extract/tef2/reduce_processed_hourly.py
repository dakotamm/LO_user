"""
Reduce processed_avg to small hourly time series, one per section.

process_sections_avg.py writes transport binned into 1000 salinity classes at
every hour, which for a two year run is around 11 GB across five sections --
too big to move off apogee. But the tidal questions only need the integrals
over salinity, which are 1-D in time:

    qnet    net volume transport                sum over bins of q
    Fsalt   net salt flux, sum(q*s)             sum over bins of 'salt'
    Fsalt2  net salinity-variance flux, sum(q*s^2)   sum over bins of 'salt2'
    sflux   flux-weighted salinity, Fsalt/qnet

Those are exact integrals of the binned fields, not approximations, because
process_sections_avg.py already formed q*s and q*s*s per bin before binning.

WHY THIS IS NEEDED AT ALL
bulk_avg is Godin filtered and daily subsampled, so every tidal-timescale
signal has been removed from it by construction. Anything about tidal storage,
tidal pumping, or the phase between transport and salinity has to come from
here instead.

Output: LO_output/extract/<gtagex>/tef2/hourly_flux_<ds0>_<ds1>_<gctag>.nc
with dims (time, sect). A couple of MB, so easy to bring back to the mac.

On apogee:
python reduce_processed_hourly.py -gtx wb1_t0_xn11abbur00 -ctag pc1 \
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
in_dir = out_dir0 / ('processed_avg_' + Ldir['ds0'] + '_' + Ldir['ds1'])
out_fn = out_dir0 / ('hourly_flux_' + Ldir['ds0'] + '_' + Ldir['ds1']
                     + '_' + gctag + '.nc')

sect_list = sorted([item.name.replace('.nc', '') for item in in_dir.glob('*.nc')])
print('sections: ' + ', '.join(sect_list))

# tracers to integrate. salt and salt2 are what the variance work needs; the
# rest are cheap to add here and expensive to come back for.
VN = ['salt', 'salt2']

tt0 = time()
out = {}
for sn in sect_list:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    if 'time' not in out:
        out['time'] = ds.time.values
    d = {}
    # qnet is already 1-D in the file; recompute from the bins as a check that
    # the binning conserved transport
    d['qnet'] = ds.qnet.values
    q_from_bins = ds.q.sum('sbins').values
    err = np.nanmax(np.abs(q_from_bins - d['qnet']))
    scale = np.nanmax(np.abs(d['qnet']))
    print('  %-10s max|sum(bins)-qnet| = %.3e  (%.2e of peak qnet)'
          % (sn, err, err / scale if scale > 0 else np.nan))
    d['ssh'] = ds.ssh.values
    for vn in VN:
        d['F' + vn] = ds[vn].sum('sbins').values
    ds.close()
    d['sflux'] = np.where(d['qnet'] != 0, d['Fsalt'] / d['qnet'], np.nan)
    out[sn] = d
    sys.stdout.flush()

print('\nelapsed %.1f sec' % (time() - tt0))

# ---------------------------------------------------------------- package ---
vlist = ['qnet', 'ssh', 'Fsalt', 'Fsalt2', 'sflux']
ds_out = xr.Dataset(coords={'time': out['time'], 'sect': sect_list})
for v in vlist:
    ds_out[v] = (('time', 'sect'),
                 np.stack([out[sn][v] for sn in sect_list], axis=1))
ds_out['qnet'].attrs['units'] = 'm3 s-1'
ds_out['Fsalt'].attrs['units'] = 'g kg-1 m3 s-1'
ds_out['Fsalt2'].attrs['units'] = '(g kg-1)2 m3 s-1'
ds_out['sflux'].attrs['long_name'] = 'transport weighted salinity, Fsalt/qnet'
ds_out['ssh'].attrs['units'] = 'm'
ds_out.attrs['note'] = ('hourly, NOT tidally filtered -- reduced from '
                        'processed_avg by summing over salinity bins')
ds_out.to_netcdf(out_fn)
print('saved %s' % out_fn)
print(ds_out)
