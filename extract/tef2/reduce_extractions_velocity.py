"""
Reduce extractions_avg to the VELOCITY field at each section.

extract_sections_avg.py stores the volume flux per face-cell, q(time, z, p)
[m3 s-1], together with the face width dd(p) [m] and the cell thickness
DZ(time, z, p) [m]. Their ratio is the section-normal velocity

    u(time, z, p) = q / (dd * DZ)                              [m s-1]

which is the only place in the tef2 chain where velocity survives at all:
bulk_avg is Godin filtered and binned into salinity classes, hourly_flux is a
section integral, and structure_*.nc carries the time-mean flux but not the
cell thickness needed to turn it into a speed.

NOTE u is the component NORMAL to the section, not the speed. The along-
section component never enters the tef2 extraction, so |u| here is a lower
bound on the true current magnitude.

WHY THE SECTION AVERAGE IS NOT ENOUGH
qnet/area is the barotropic velocity that fills and empties the cove. It is
what a tidal prism argument predicts and it is easy to get from hourly_flux.
But at every Penn Cove section the residual flux changes sign ACROSS the
section (the lateral gyre), so the section average cancels most of the real
motion. Both numbers are saved here, and their ratio is the point:

    ubar   = qnet / area              signed section average, the prism part
    urms   = area-weighted rms of u   the actual local magnitude
    umax   = max |u| over the section the fastest cell at that hour

Saved per section, HOURLY (time, sect), so that the tidal, fortnightly and
seasonal bands can all be recovered afterwards:
    ubar, urms, umax, up95, area, zeta

Saved per FACE-CELL (z, p), so the magnitude can be mapped in the section:
    umean       time-mean u -- the residual velocity, signed
    urms        rms of u about zero -- total magnitude
    utid        rms of the tidal band, u minus its Godin lowpass
    usub        sd of the Godin lowpass about its own mean -- subtidal variability
    up95, umax  95th percentile and maximum of |u|
    umon_mean   (month, z, p) monthly mean u, the seasonal cycle of the residual
    umon_tid    (month, z, p) monthly tidal rms, the seasonal cycle of the tide
    uspr, unea  tidal rms on spring and on neap days
    dzbar       time-mean cell thickness, so cell areas can be rebuilt
    lon, lat, h, dd

Spring and neap are labelled from the daily range of section-mean zeta at
-ref_sect, three-day smoothed and split at its terciles, so every section
carries the SAME label and the sections can be compared on the same days.

Output is a few MB: LO_output/extract/<gtagex>/tef2/velocity_<ds0>_<ds1>_<gctag>.nc

On apogee:
python reduce_extractions_velocity.py -gtx wb1_t0_xn11abbur00 -ctag pc1 \
    -0 2024.01.01 -1 2025.12.31
"""
import sys
from time import time

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import convolve1d

from lo_tools import Lfun, zfun
from lo_tools import extract_argfun as exfun

# exfun.intro() builds its own parser and rejects anything it does not know,
# so -ref_sect is pulled out of sys.argv before it runs
ref_sect = 'pc_lp'
if '-ref_sect' in sys.argv:
    k = sys.argv.index('-ref_sect')
    ref_sect = sys.argv[k + 1]
    del sys.argv[k:k + 2]

Ldir = exfun.intro()

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
in_dir = out_dir0 / ('extractions_avg_' + Ldir['ds0'] + '_' + Ldir['ds1'])
if not in_dir.is_dir():
    # the older his-based extractions carry vel and DZ too, which is what the
    # small test runs on the mac have
    in_dir = out_dir0 / ('extractions_' + Ldir['ds0'] + '_' + Ldir['ds1'])
    print('no extractions_avg -- falling back to %s' % in_dir.name)
out_fn = out_dir0 / ('velocity_' + Ldir['ds0'] + '_' + Ldir['ds1']
                     + '_' + gctag + '.nc')

# face coordinates, same source as reduce_extractions_structure.py
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
sect_df = pd.read_pickle(tef2_dir / ('sect_df_' + gctag + '.p'))
grid = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon_rho = grid.lon_rho.values
lat_rho = grid.lat_rho.values
grid.close()

sect_list = sorted([f.name.replace('.nc', '') for f in in_dir.glob('*.nc')])
print('sections: ' + ', '.join(sect_list))

GODIN = zfun.godin_shape()
NPAD = len(GODIN) // 2


def godin_nd(a):
    """Godin lowpass along axis 0 of an ND array, NaN at the ends.

    zfun.lowpass flattens the whole array before convolving, which lets one
    column's last hours bleed into the next column's first hours. Here the
    filter is applied strictly along the time axis instead, so every cell is
    filtered independently and only the true record ends are padded.
    """
    out = convolve1d(a, GODIN, axis=0, mode='nearest')
    out[:NPAD] = np.nan
    out[-NPAD:] = np.nan
    return out


# ---------------------------------------------- pass 1: spring/neap labels ---
# cheap: zeta is (time, p) only
fn_ref = in_dir / (ref_sect + '.nc')
if not fn_ref.is_file():
    fn_ref = in_dir / (sect_list[0] + '.nc')
    print('-ref_sect %s not found, using %s for spring/neap' % (ref_sect, fn_ref.stem))
    ref_sect = fn_ref.stem
dsr = xr.open_dataset(fn_ref)
t_all = pd.to_datetime(dsr.time.values)
zeta_ref = pd.Series(np.nanmean(dsr.zeta.values, axis=1), index=t_all)
dsr.close()

day = zeta_ref.index.floor('D')
rng = zeta_ref.groupby(day).max() - zeta_ref.groupby(day).min()
rng_sm = rng.rolling(3, center=True, min_periods=1).mean()
q33, q67 = rng_sm.quantile([1 / 3, 2 / 3]).values
phase_day = pd.Series(np.where(rng_sm >= q67, 'spring',
                               np.where(rng_sm <= q33, 'neap', 'transition')),
                      index=rng_sm.index)
phase = phase_day.reindex(day).values
is_spr = phase == 'spring'
is_nea = phase == 'neap'
month = t_all.month.values
print('spring/neap from %s zeta: terciles of the 3-day smoothed daily range at '
      '%.2f and %.2f m (%d spring, %d neap, %d transition hours)'
      % (ref_sect, q33, q67, is_spr.sum(), is_nea.sum(),
         len(phase) - is_spr.sum() - is_nea.sum()))

# ------------------------------------------------------ pass 2: velocities ---
tt0 = time()
ds_out = xr.Dataset()
hourly = {vn: [] for vn in ['ubar', 'urms', 'umax', 'up95', 'area', 'zeta']}
summary = []

for sn in sect_list:
    ds = xr.open_dataset(in_dir / (sn + '.nc'))
    DZ = ds.DZ.values                                 # (time, z, p)
    dd = ds.dd.values                                 # (p)
    h = ds.h.values
    zeta = ds.zeta.values                             # (time, p)
    area = DZ * dd[np.newaxis, np.newaxis, :]         # cell area (time, z, p)

    # extractions_avg (Huon/Hvom) stores the flux q; the older his-based
    # extractions store the velocity itself. Either way both are wanted.
    with np.errstate(divide='ignore', invalid='ignore'):
        if 'q' in ds.data_vars:
            q = ds.q.values                           # (time, z, p)
            u = np.where(area > 0, q / area, np.nan)
        else:
            u = ds.vel.values
            q = u * area
    NT, NZ, NP = u.shape

    # ---- hourly section-level series
    A_tot = np.nansum(area, axis=(1, 2))
    ubar = np.nansum(q, axis=(1, 2)) / A_tot
    urms = np.sqrt(np.nansum(area * u ** 2, axis=(1, 2)) / A_tot)
    au = np.abs(u)
    umax = np.nanmax(au, axis=(1, 2))
    up95 = np.nanpercentile(au, 95, axis=(1, 2))
    hourly['ubar'].append(ubar)
    hourly['urms'].append(urms)
    hourly['umax'].append(umax)
    hourly['up95'].append(up95)
    hourly['area'].append(A_tot)
    hourly['zeta'].append(np.nanmean(zeta, axis=1))

    # ---- per-cell statistics
    ulp = godin_nd(u)                                 # subtidal
    utd = u - ulp                                     # tidal band
    ok = np.isfinite(ulp[:, 0, 0])                    # rows the filter kept

    umean = np.nanmean(u, axis=0)
    urms_c = np.sqrt(np.nanmean(u ** 2, axis=0))
    utid_c = np.sqrt(np.nanmean(utd[ok] ** 2, axis=0))
    usub_c = np.nanstd(ulp[ok], axis=0)
    up95_c = np.nanpercentile(au, 95, axis=0)
    umax_c = np.nanmax(au, axis=0)

    umon_mean = np.full((12, NZ, NP), np.nan)
    umon_tid = np.full((12, NZ, NP), np.nan)
    for m in range(1, 13):
        sel = month == m
        if sel.any():
            umon_mean[m - 1] = np.nanmean(u[sel], axis=0)
            selo = sel & ok
            if selo.any():
                umon_tid[m - 1] = np.sqrt(np.nanmean(utd[selo] ** 2, axis=0))
    uspr = (np.sqrt(np.nanmean(utd[is_spr & ok] ** 2, axis=0))
            if (is_spr & ok).any() else np.full((NZ, NP), np.nan))
    unea = (np.sqrt(np.nanmean(utd[is_nea & ok] ** 2, axis=0))
            if (is_nea & ok).any() else np.full((NZ, NP), np.nan))

    d = sect_df[sect_df.sn == sn]
    lon_f = 0.5 * (lon_rho[d.jrp, d.irp] + lon_rho[d.jrm, d.irm])
    lat_f = 0.5 * (lat_rho[d.jrp, d.irp] + lat_rho[d.jrm, d.irm])

    pdim, zdim = sn + '_p', sn + '_z'
    for nm, arr in [('umean', umean), ('urms', urms_c), ('utid', utid_c),
                    ('usub', usub_c), ('up95', up95_c), ('umax', umax_c),
                    ('uspr', uspr), ('unea', unea),
                    ('dzbar', np.nanmean(DZ, axis=0))]:
        ds_out['%s_%s' % (sn, nm)] = ((zdim, pdim), arr)
    ds_out[sn + '_umon_mean'] = (('month', zdim, pdim), umon_mean)
    ds_out[sn + '_umon_tid'] = (('month', zdim, pdim), umon_tid)
    ds_out[sn + '_lon'] = ((pdim), lon_f)
    ds_out[sn + '_lat'] = ((pdim), lat_f)
    ds_out[sn + '_h'] = ((pdim), h)
    ds_out[sn + '_dd'] = ((pdim), dd)

    # the headline comparison: how much of the local speed the section average
    # sees. A ratio near 1 is a section moving as a slab; well below 1 means
    # the flux is cancelling within the section.
    print('  %-10s NZ=%d NP=%2d  A=%8.0f m2 | section avg rms %.4f  local rms '
          '%.4f  (avg/local %.2f)  p95 %.3f  max %.3f m/s | residual |umean| '
          'mean %.4f max %.4f'
          % (sn, NZ, NP, np.nanmean(A_tot), np.sqrt(np.nanmean(ubar ** 2)),
             np.sqrt(np.nanmean(urms ** 2)),
             np.sqrt(np.nanmean(ubar ** 2)) / np.sqrt(np.nanmean(urms ** 2)),
             np.nanmean(up95), np.nanmax(umax), np.nanmean(np.abs(umean)),
             np.nanmax(np.abs(umean))))
    summary.append(dict(sect=sn, NZ=NZ, NP=NP, area_m2=np.nanmean(A_tot),
                        ubar_rms=np.sqrt(np.nanmean(ubar ** 2)),
                        u_rms=np.sqrt(np.nanmean(urms ** 2)),
                        u_p95=np.nanmean(up95), u_max=np.nanmax(umax),
                        umean_abs_mean=np.nanmean(np.abs(umean)),
                        umean_abs_max=np.nanmax(np.abs(umean)),
                        utid_rms=np.nanmean(utid_c), usub_rms=np.nanmean(usub_c)))
    ds.close()
    del q, DZ, u, ulp, utd, au, area
    sys.stdout.flush()

for vn, lst in hourly.items():
    ds_out[vn] = (('time', 'sect'), np.array(lst).T)
ds_out = ds_out.assign_coords(time=t_all.values, sect=sect_list,
                              month=np.arange(1, 13))
ds_out['phase'] = (('time',), phase.astype('U10'))
ds_out.attrs['note'] = (
    'section-NORMAL velocity u = q/(dd*DZ) from extractions_avg. Hourly and '
    'NOT tidally filtered. ubar = qnet/area (section average, signed); urms = '
    'area-weighted rms of the local u; per-cell fields on (z, p) with z index '
    '0 = bed, -1 = surface. spring/neap labelled from ' + ref_sect + ' zeta.')
ds_out.to_netcdf(out_fn)

pd.DataFrame(summary).to_csv(str(out_fn).replace('.nc', '_summary.csv'),
                             index=False, float_format='%.5f')
print('\nelapsed %.1f sec' % (time() - tt0))
print('saved %s' % out_fn)
print('saved %s' % str(out_fn).replace('.nc', '_summary.csv'))
