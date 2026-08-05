"""
Extract the salinity-variance state of each tef2 segment, hourly.

For every segment in seg_info_dict this accumulates, at each hour:
    volume      sum(dV)
    salt_int    sum(s dV)
    salt2_int   sum(s^2 dV)
    area        sum(dx dy)          (constant, but carried for convenience)
    surf_sflux  sum(s_surf * EminusP * dx dy)
    surf_s2flux sum(s_surf^2 * EminusP * dx dy)

salt2_int is the state variable of the salinity-variance budget. Combined with
the salt2 flux that process_sections_avg.py already bins at each section, the
volume-integrated mixing follows as the residual

    M = sum_j F_j - d/dt (salt2_int)

where F_j is the s^2 flux through bounding section j, signed into the segment.

WHY HISTORY FILES AND NOT AVERAGES
This reads ocean_his (instantaneous snapshots), while the section fluxes come
from ocean_avg. That pairing is deliberate. The budget is

    [state(t2) - state(t1)] / (t2 - t1) = mean flux over [t1, t2]

so the state must be instantaneous at the interval ends and the flux must be
the average across it. Taking the state from an avg file would also give
mean(s)^2 rather than mean(s^2), which is a different quantity and would put a
spurious residual straight into M.

NOTE the ji_list of a segment does not depend on the river set -- only its
riv_list does -- so a seg_info_dict built with riv00 gives the same cells as
one built with trapsN00, and is fine for this extraction.

To test on the mac or apogee:
run extract_segments_SV.py -gtx wb1_t0_xn11abbur00 -ro 2 -ctag pc1 -riv riv00 \
    -0 2024.07.01 -1 2024.07.02 -test True

On apogee for real:
python extract_segments_SV.py -gtx wb1_t0_xn11abbur00 -ro 2 -ctag pc1 \
    -riv riv00 -0 2024.01.01 -1 2025.12.31 > pc1_segSV.log &
"""

import pickle
import sys
from time import time

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zrfun
from lo_tools import extract_argfun as exfun

Ldir = exfun.intro()  # this handles the argument passing

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'

seg_fn = tef2_dir / ('seg_info_dict_' + gctag + '_' + Ldir['riv'] + '.p')
seg_info = pickle.load(open(seg_fn, 'rb'))
seg_names = sorted(seg_info.keys())
print('segments: ' + ', '.join(seg_names))

out_dir = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
Lfun.make_dir(out_dir)
out_fn = out_dir / ('segments_SV_' + Ldir['ds0'] + '_' + Ldir['ds1'] + '_'
                    + gctag + '_' + Ldir['riv'] + '.nc')

fn_list = Lfun.get_fn_list('hourly', Ldir, Ldir['ds0'], Ldir['ds1'],
                           his_num=Ldir['his_num'])
if Ldir['testing']:
    fn_list = fn_list[:24]
print('\n%d history files' % len(fn_list))
print('first: %s' % fn_list[0])
print('last:  %s' % fn_list[-1])
print('first exists: %s\n' % fn_list[0].is_file())
sys.stdout.flush()

# ------------------------------------------------------------ index setup ---
# Work on the bounding box of all the segment cells rather than the whole grid.
# For wb1_pc1 that is Penn Cove plus the reach of Saratoga Passage it opens
# onto, a small fraction of the 368 x 272 grid, and it cuts the read per file
# by the same fraction.
allji = np.concatenate([np.array(seg_info[k]['ji_list']) for k in seg_names])
j0, j1 = allji[:, 0].min(), allji[:, 0].max() + 1
i0, i1 = allji[:, 1].min(), allji[:, 1].max() + 1
print('bounding box j %d:%d, i %d:%d  (%d of %d rows, %d of %d cols)'
      % (j0, j1, i0, i1, j1 - j0, 368, i1 - i0, 272))

# per-segment indices, local to the bounding box
seg_idx = {}
for k in seg_names:
    ji = np.array(seg_info[k]['ji_list'])
    seg_idx[k] = (ji[:, 0] - j0, ji[:, 1] - i0)

G, S, T = zrfun.get_basic_info(fn_list[0])

ds0 = xr.open_dataset(fn_list[0])
sl = dict(eta_rho=slice(j0, j1), xi_rho=slice(i0, i1))
h = ds0.h.isel(**sl).values
DA = (1 / ds0.pm.isel(**sl).values) * (1 / ds0.pn.isel(**sl).values)
ds0.close()

area = {k: float(DA[seg_idx[k]].sum()) for k in seg_names}

# ----------------------------------------------------------------- loop ----
NS = len(seg_names)
NT = len(fn_list)
ot = np.zeros(NT, dtype='datetime64[ns]')
V = {v: np.nan * np.ones((NT, NS)) for v in
     ['volume', 'salt_int', 'salt2_int', 'surf_sflux', 'surf_s2flux']}

tt0 = time()
for tt, fn in enumerate(fn_list):
    ds = xr.open_dataset(fn)
    ot[tt] = ds.ocean_time.values[0]
    salt = ds.salt.isel(ocean_time=0, **sl).values          # (N, M, L)
    zeta = ds.zeta.isel(ocean_time=0, **sl).values          # (M, L)
    EmP = ds.EminusP.isel(ocean_time=0, **sl).values        # (M, L)
    ds.close()

    zw = zrfun.get_z(h, zeta, S, only_w=True)
    dV = np.diff(zw, axis=0) * DA                           # (N, M, L)
    s_surf = salt[-1, :, :]

    for kk, k in enumerate(seg_names):
        jj, ii = seg_idx[k]
        dv = dV[:, jj, ii]
        s = salt[:, jj, ii]
        V['volume'][tt, kk] = np.nansum(dv)
        V['salt_int'][tt, kk] = np.nansum(s * dv)
        V['salt2_int'][tt, kk] = np.nansum(s * s * dv)
        # EminusP is a surface volume flux [m s-1]; sign convention follows
        # LO/extract/tef2/tracer_budget.py, which forms salt_surf*area*EminusP
        f = EmP[jj, ii] * DA[jj, ii]
        V['surf_sflux'][tt, kk] = np.nansum(s_surf[jj, ii] * f)
        V['surf_s2flux'][tt, kk] = np.nansum(s_surf[jj, ii] ** 2 * f)

    if np.mod(tt, 240) == 0 and tt > 0:
        el = time() - tt0
        print('  %6d / %d   %.1f min elapsed, ~%.1f min left'
              % (tt, NT, el / 60, el / 60 * (NT - tt) / tt))
        sys.stdout.flush()

print('\nTotal processing time = %0.1f sec' % (time() - tt0))

# ---------------------------------------------------------------- output ---
out = xr.Dataset(coords={'time': ot, 'seg': seg_names})
units = {'volume': 'm3', 'salt_int': 'g kg-1 m3', 'salt2_int': '(g kg-1)2 m3',
         'surf_sflux': 'g kg-1 m3 s-1', 'surf_s2flux': '(g kg-1)2 m3 s-1'}
for v in V.keys():
    out[v] = (('time', 'seg'), V[v], {'units': units[v]})
out['area'] = (('seg'), np.array([area[k] for k in seg_names]), {'units': 'm2'})

# handy derived fields, so the analysis does not have to redo them
out['sbar'] = out.salt_int / out.volume
out['sbar'].attrs['long_name'] = 'volume mean salinity'
out['svar_int'] = out.salt2_int - out.volume * out.sbar ** 2
out['svar_int'].attrs['long_name'] = 'volume integrated salinity variance'
out['svar_int'].attrs['units'] = '(g kg-1)2 m3'

out.to_netcdf(out_fn)
print('saved %s' % out_fn)
print(out)
