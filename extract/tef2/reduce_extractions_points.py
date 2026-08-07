"""
Pull the FULL-DEPTH hourly record at a few named faces of one tef2 section.

This is the velocity companion to the mouth salinity work. The two Penn Cove
mouth points -- pc_lp p=2 (north, lat 48.2438) and p=11 (south, lat 48.2275),
depth-matched at h = 19.8 and 20.0 m -- already have their salinity record in
extractions_avg. The same file holds the volume flux q(time, z, p) and the cell
thickness DZ(time, z, p), so the section-normal VELOCITY at those points

    u(time, z) = q / (dd * DZ)                                  [m s-1]

needs no new extraction either. It just has to be sliced out and brought back.

WHY ALL 30 LEVELS AND NOT JUST TOP AND BOTTOM
Two points by thirty levels by two years is about 4 MB per variable -- small
enough that there is no reason to throw depth away at this stage. Top and
bottom are the first thing to look at, so they are also saved pre-sliced as
u_top / u_bot, but the full profile is there when the question becomes where
in the water column the tidal and residual flow actually live.

DEPTHS ARE CARRIED, NOT ASSUMED
Sigma levels move with the tide, so "level 5" is not a fixed depth. z_rho and
z_w are rebuilt here by integrating DZ up from -h, which makes them exactly
consistent with the cell thicknesses the velocity was divided by. Every level
therefore comes with its true depth at every hour.

SIGN
q is positive from the minus side to the plus side of the section. At pc_lp
that is eastward, which is OUT of Penn Cove. The script does not assume this:
it correlates the section-summed q against d(zeta)/dt and reports which sign
is flood, storing it as the flood_sign attribute. u_in = flood_sign * u is
then positive into the cove.

OUTPUT  LO_output/extract/<gtagex>/tef2/points_<ds0>_<ds1>_<gctag>_<sect>.nc
    u, q, DZ, z_rho, salt, temp   (time, z, point)
    u_top, u_bot, zeta            (time, point)
    lon, lat, h, dd               (point)
dims are (time, z, point) with z index 0 = bed, -1 = surface.

On apogee:
python reduce_extractions_points.py -gtx wb1_t0_xn11abbur00 -ctag pc1 \
    -0 2024.01.01 -1 2025.12.31 -sect pc_lp -lats 48.2438,48.2275 \
    -names north,south

or name the faces directly, which skips the lat lookup:
python reduce_extractions_points.py -gtx wb1_t0_xn11abbur00 -ctag pc1 \
    -0 2024.01.01 -1 2025.12.31 -sect pc_lp -faces 2,11 -names north,south
"""
import sys
from time import time

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun


# exfun.intro() builds its own parser and rejects anything it does not know,
# so these are pulled out of sys.argv before it runs
def pop_arg(flag, default):
    if flag in sys.argv:
        k = sys.argv.index(flag)
        val = sys.argv[k + 1]
        del sys.argv[k:k + 2]
        return val
    return default


sect = pop_arg('-sect', 'pc_lp')
lats_str = pop_arg('-lats', '48.2438,48.2275')
faces_str = pop_arg('-faces', '')
names_str = pop_arg('-names', 'north,south')

Ldir = exfun.intro()

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
# prefer extractions_uv: it is the only one with the tangential component, so
# it is the only one that can give a real speed rather than a normal component
for tag in ['extractions_uv_', 'extractions_avg_', 'extractions_']:
    in_dir = out_dir0 / (tag + Ldir['ds0'] + '_' + Ldir['ds1'])
    if in_dir.is_dir():
        break
print('reading from %s' % in_dir.name)
out_fn = out_dir0 / ('points_' + Ldir['ds0'] + '_' + Ldir['ds1'] + '_'
                     + gctag + '_' + sect + '.nc')

fn_in = in_dir / (sect + '.nc')
if not fn_in.is_file():
    raise SystemExit('%s not found' % fn_in)

# face coordinates live in sect_df, NOT in the extraction
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
sect_df = pd.read_pickle(tef2_dir / ('sect_df_' + gctag + '.p'))
grid = xr.open_dataset(Ldir['grid'] / 'grid.nc')
d = sect_df[sect_df.sn == sect]
lon_f = 0.5 * (grid.lon_rho.values[d.jrp, d.irp] + grid.lon_rho.values[d.jrm, d.irm])
lat_f = 0.5 * (grid.lat_rho.values[d.jrp, d.irp] + grid.lat_rho.values[d.jrm, d.irm])
grid.close()

tt0 = time()
ds = xr.open_dataset(fn_in)
dd_all = ds.dd.values
h_all = ds.h.values
NP = len(dd_all)
if len(lat_f) != NP:
    raise SystemExit('sect_df gives %d faces but %s.nc has %d'
                     % (len(lat_f), sect, NP))

# ------------------------------------------------------------ pick faces ---
if faces_str:
    pk = [int(v) for v in faces_str.split(',')]
else:
    want = [float(v) for v in lats_str.split(',')]
    pk = [int(np.argmin(np.abs(lat_f - w))) for w in want]
names = [s.strip() for s in names_str.split(',')]
if len(names) != len(pk):
    names = ['p%d' % k for k in pk]

print('%s has %d faces; taking:' % (fn_in.name, NP))
for nm, k in zip(names, pk):
    print('  %-8s p=%2d  lat %.4f  lon %.4f  h %5.1f m  dd %6.1f m'
          % (nm, k, lat_f[k], lon_f[k], h_all[k], dd_all[k]))

# --------------------------------------------------------- slice and load ---
t = pd.to_datetime(ds.time.values)
DZ = ds.DZ.values[:, :, pk]                            # (time, z, npt)
dd = dd_all[pk]
h = h_all[pk]
zeta = ds.zeta.values[:, pk]
area = DZ * dd[np.newaxis, np.newaxis, :]

u_tan = None
with np.errstate(divide='ignore', invalid='ignore'):
    if 'u_norm' in ds.data_vars:
        # extractions_uv: both components were extracted directly
        u = ds.u_norm.values[:, :, pk]
        u_tan = ds.u_tan.values[:, :, pk]
        q = ds.q.values[:, :, pk]
    elif 'q' in ds.data_vars:
        q = ds.q.values[:, :, pk]
        u = np.where(area > 0, q / area, np.nan)
    else:
        u = ds.vel.values[:, :, pk]                    # his-based extractions
        q = u * area
NT, NZ, NPT = u.shape

# depths, integrated up from the bed so they match DZ exactly
z_w = np.concatenate([-h[np.newaxis, np.newaxis, :] * np.ones((NT, 1, NPT)),
                      -h[np.newaxis, np.newaxis, :] + np.cumsum(DZ, axis=1)],
                     axis=1)                           # (time, z+1, npt)
z_rho = 0.5 * (z_w[:, :-1, :] + z_w[:, 1:, :])
# check: the top w level must land on zeta
err = np.nanmax(np.abs(z_w[:, -1, :] - zeta))
print('z rebuilt from DZ; max |z_w(top) - zeta| = %.2e m' % err)

# ---------------------------------------------------------- which is flood ---
# section-summed q against the rate of change of section-mean sea level: on a
# rising tide water must be entering the volume behind the section
qnet = np.nansum(ds.q.values, axis=(1, 2)) if 'q' in ds.data_vars else \
    np.nansum(ds.vel.values * (ds.DZ.values * dd_all[np.newaxis, np.newaxis, :]),
              axis=(1, 2))
zbar = np.nanmean(ds.zeta.values, axis=1)
r_check = np.corrcoef(qnet, np.gradient(zbar))[0, 1]
flood_sign = -1.0 if r_check < 0 else 1.0
print('corr(qnet, d(zeta)/dt) = %+.2f  ->  flood is q %s 0, so u_in = %+.0f * u'
      % (r_check, '<' if flood_sign < 0 else '>', flood_sign))

# ------------------------------------------------------------------ write ---
out = xr.Dataset()
dims3 = ('time', 'z', 'point')
out['u'] = (dims3, u.astype('float32'))
out['q'] = (dims3, q.astype('float32'))
if u_tan is not None:
    out['u_tan'] = (dims3, u_tan.astype('float32'))
    out['speed'] = (dims3, np.hypot(u, u_tan).astype('float32'))
    out['u_tan_top'] = (('time', 'point'), u_tan[:, -1, :].astype('float32'))
    out['u_tan_bot'] = (('time', 'point'), u_tan[:, 0, :].astype('float32'))
out['DZ'] = (dims3, DZ.astype('float32'))
out['z_rho'] = (dims3, z_rho.astype('float32'))
for vn in ['salt', 'temp', 'oxygen']:
    if vn in ds.data_vars:
        out[vn] = (dims3, ds[vn].values[:, :, pk].astype('float32'))
# the first thing to look at, pre-sliced: z index 0 is the bed, -1 the surface
out['u_top'] = (('time', 'point'), u[:, -1, :].astype('float32'))
out['u_bot'] = (('time', 'point'), u[:, 0, :].astype('float32'))
out['zeta'] = (('time', 'point'), zeta.astype('float32'))
out['lon'] = (('point',), lon_f[pk])
out['lat'] = (('point',), lat_f[pk])
out['h'] = (('point',), h)
out['dd'] = (('point',), dd)
out['face'] = (('point',), np.array(pk))
out = out.assign_coords(time=ds.time.values, point=np.array(names))

out.u.attrs['units'] = 'm s-1'
out.u.attrs['long_name'] = ('velocity normal to section %s, positive from the '
                            'minus side to the plus side' % sect)
out.z_rho.attrs['units'] = 'm, negative down, relative to instantaneous free surface datum'
out.attrs['note'] = (
    'full-depth hourly record at selected faces of %s, sliced from %s. '
    'u = q/(dd*DZ) is the section-NORMAL velocity, hourly and NOT tidally '
    'filtered. z index 0 = bed, -1 = surface. z_rho and DZ are consistent by '
    'construction. u_in = flood_sign * u is positive INTO the basin behind '
    'the section.' % (sect, in_dir.name))
out.attrs['flood_sign'] = flood_sign
out.attrs['corr_qnet_dzeta_dt'] = r_check
out.attrs['sect'] = sect
out.attrs['faces'] = str(pk)
ds.close()
out.to_netcdf(out_fn)

print('\n%d hourly steps, %s to %s (UTC)' % (NT, t[0], t[-1]))
print('velocity magnitude, section-normal [m s-1]:')
print('  %-8s %9s %9s %9s %9s %9s' % ('point', 'lev', 'mean', 'rms', 'p95|u|', 'max|u|'))
for j, nm in enumerate(names):
    for lab, k in [('surface', -1), ('bed', 0)]:
        v = u[:, k, j]
        print('  %-8s %9s %+9.4f %9.4f %9.4f %9.4f'
              % (nm, lab, np.nanmean(v), np.sqrt(np.nanmean(v ** 2)),
                 np.nanpercentile(np.abs(v), 95), np.nanmax(np.abs(v))))
    v = u[:, :, j]
    print('  %-8s %9s %+9.4f %9.4f %9.4f %9.4f'
          % (nm, 'all z', np.nanmean(v), np.sqrt(np.nanmean(v ** 2)),
             np.nanpercentile(np.abs(v), 95), np.nanmax(np.abs(v))))

print('\nelapsed %.1f sec' % (time() - tt0))
print('saved %s  (%.1f MB)' % (out_fn, out_fn.stat().st_size / 1e6))
