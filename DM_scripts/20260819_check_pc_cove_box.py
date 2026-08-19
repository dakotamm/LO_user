"""
Sanity-check a pc_cove box extraction before trusting it (or before spending a
transfer on it). Reads metadata and a few slices only -- it does NOT load the
whole file, so it is fast enough to run on apogee straight after the extraction.

run 20260819_check_pc_cove_box.py
run 20260819_check_pc_cove_box.py -gtx wb1_t0_xn11abbur00 -0 2024.01.01 -1 2025.12.31

Checks, in order of how badly each would hurt:
 1. NATIVE GRIDS. If eta_v is missing the box was made with -uv_to_rho, which
    leaves a NaN ring on the outermost row/column -- and the east edge is the
    mouth, so v at pc_lp would be all NaN. Fatal; re-extract.
 2. pc_lp u-faces present. xi_u must reach the pc_lp face, which is the whole
    reason the east edge is rho column 68.
 3. w present and finite over water. Without it there is no vertical velocity.
 4. v finite at the pc_lp rho column, which is the column most likely to be
    damaged by an edge effect.
 5. Time axis complete and hourly, no gaps or duplicates.
 6. Grid self-consistency: h/mask taken from the RUN, compared with grid.nc.
    For the t0 family these agree; for r0 they do NOT (h differs at 5197 cells,
    up to 20 m) -- see [[wb1-grid-vs-run-mask]]. A mismatch here is only a
    problem if you then mix in grid.nc values downstream.
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
p.add_argument('-job', default='pc_cove', type=str)
p.add_argument('-0', '--ds0', default='2024.01.01', type=str)
p.add_argument('-1', '--ds1', default='2025.12.31', type=str)
args = p.parse_args()

warnings.simplefilter('ignore')
Ldir = Lfun.Lstart(gridname='wb1')
fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'box' /
      ('%s_%s_%s.nc' % (args.job, args.ds0, args.ds1)))
PC_LP_LON = -122.6534
fails, warns = [], []


def ok(c, msg):
    print(('  PASS  ' if c else '  FAIL  ') + msg)
    if not c:
        fails.append(msg)


print('checking %s' % fn)
if not fn.is_file():
    sys.exit('  FAIL  file does not exist')
print('  size %.2f GB' % (fn.stat().st_size / 1e9))
ds = xr.open_dataset(fn)
print('  dims %s' % {k: int(v) for k, v in ds.sizes.items()})

print('\n1. native staggered grids (no -uv_to_rho)')
native = ('eta_v' in ds.sizes) and ('xi_u' in ds.sizes)
ok(native, 'eta_v and xi_u present')
if not native:
    sys.exit('\nFATAL: made with -uv_to_rho. Re-extract without it.')
NR, NC = ds.sizes['eta_rho'], ds.sizes['xi_rho']
ok(ds.sizes['xi_u'] == NC - 1 and ds.sizes['eta_v'] == NR - 1,
   'xi_u == xi_rho-1 (%d) and eta_v == eta_rho-1 (%d)' % (ds.sizes['xi_u'], ds.sizes['eta_v']))
ok(ds.sizes['xi_v'] == NC, 'xi_v spans the full rho range (%d) so v exists at the mouth'
   % ds.sizes['xi_v'])

print('\n2. pc_lp u-faces inside the box')
lon_u = ds.lon_u.values[0, :]
du = np.abs(lon_u - PC_LP_LON).min()
ok(du < 6e-4, 'a u-face sits at the pc_lp longitude (closest is %.5f deg = %.0f m away)'
   % (du, du * 74000))

print('\n3. w present and usable')
ok('w' in ds.data_vars, "'w' is in the file")
if 'w' in ds.data_vars:
    ok(ds.w.dims == ('ocean_time', 's_w', 'eta_rho', 'xi_rho'), 'w dims %s' % (ds.w.dims,))
    wsnap = ds.w.isel(ocean_time=slice(0, 3)).values
    mask = ds.mask_rho.values.astype(bool)
    fin = np.isfinite(wsnap[:, :, mask]).mean()
    ok(fin > 0.99, 'w finite over %.1f%% of water cells' % (100 * fin))

print('\n4. v at the pc_lp rho column (the edge most at risk)')
lon_r = ds.lon_rho.values[0, :]
i_lp = int(np.argmin(np.abs(lon_r - PC_LP_LON)))
mask = ds.mask_rho.values.astype(bool)
wet = np.flatnonzero(mask[:, i_lp])
vcol = ds.v.isel(xi_v=i_lp, ocean_time=slice(0, 24)).values
kk = np.arange(wet.min(), wet.max())
fin = np.isfinite(vcol[:, :, kk]).mean()
ok(fin > 0.99, 'v finite at %.1f%% of the %d interior v-faces at rho col %d'
   % (100 * fin, len(kk), i_lp))
ok(np.nanstd(vcol[:, :, kk]) > 1e-5, 'v there is not identically zero (std %.2e m/s)'
   % np.nanstd(vcol[:, :, kk]))

print('\n5. time axis')
tt = pd.to_datetime(ds.ocean_time.values)
dt = np.diff(tt.values).astype('timedelta64[m]').astype(float)
ok(len(tt) > 0 and (dt == 60).all(), '%d times, all exactly hourly' % len(tt))
ok(not tt.duplicated().any(), 'no duplicate timestamps')
print('        %s to %s' % (tt[0], tt[-1]))
exp = int(round((pd.Timestamp(args.ds1.replace('.', '-')) + pd.Timedelta(days=1)
                 - pd.Timestamp(args.ds0.replace('.', '-'))) / pd.Timedelta(hours=1)))
if len(tt) not in (exp, exp + 1):
    warns.append('expected ~%d hourly times for %s..%s, found %d' % (exp, args.ds0, args.ds1, len(tt)))

print('\n6. grid provenance (h/mask come from the RUN, not grid.nc)')
g = xr.open_dataset(Ldir['data'] / 'grids' / 'wb1' / 'grid.nc')
i0 = int(np.argmin(np.abs(g.lon_rho.values[0, :] - lon_r[0])))
j0 = int(np.argmin(np.abs(g.lat_rho.values[:, 0] - ds.lat_rho.values[0, 0])))
gh = g.h.values[j0:j0 + NR, i0:i0 + NC]
gm = g.mask_rho.values[j0:j0 + NR, i0:i0 + NC]
nh = int((np.abs(ds.h.values - gh) > 1e-6).sum())
nm = int((ds.mask_rho.values != gm).sum())
print('        h differs from grid.nc at %d cells, mask at %d' % (nh, nm))
if nh or nm:
    warns.append('this run does NOT share grid.nc bathymetry -- never mix grid.nc h/pm/pn '
                 'into analysis of this box (see wb1-grid-vs-run-mask)')
else:
    print('  PASS  run grid matches grid.nc, so either source is safe')

print('\ncove vs box: %d wet cells, %d in rho cols 0..%d (the cove), %d Saratoga'
      % (mask.sum(), mask[:, :NC - 1].sum(), NC - 2, mask[:, NC - 1].sum()))
print('\n' + '=' * 62)
for w in warns:
    print('  WARN  ' + w)
print(('FAILED %d check(s) -- do not transfer yet' % len(fails)) if fails
      else 'ALL CHECKS PASSED -- safe to transfer and analyse')
