"""
Reduce pcmap tracks to TIDAL-DAY-MEAN particle paths, for the gyre / eddy
trapping analysis (Lagrangian residual velocity, looping, time spent in a
gyre) -- small enough to analyse the whole year on the mac without the tracks.

Runs on apogee next to the tracks (reads only lon, lat, cs). Rerunning skips
releases already reduced unless -clobber.

FILTER: a running 25-h boxcar (one lunar day, 24.84 h) of each particle's
hourly lon, lat and cs, centred on hours 12, 12 + step, ... up to the last
centre whose window fits inside -days. This removes the tidal back-and-forth
(the excursion), leaving the residual path. A window with any missing hour is
NaN. The boxcar passes almost nothing shorter than a day, so -step 12 samples
the smoothed path at its Nyquist rate; -step 6 for smoother plots.

Particles are kept, and labelled, as in 20261005_pcmap_reduce.py: only those
whose INITIAL rho cell is in the cove (pc_cp_m + pc_cp_p + pc_lp_m), quadrant
from pcmap_regions.py (pc_ew N/S split), surface half if the initial cs >= -0.5.
Particle order is the track file's order after that drop, the same as the
reduce, so the two can be joined.

Per release (pickle):
  meta    file, dir, t0, set (E/F from the dir name), nf, days, step,
          n_dropped
  t_h     (nt,) window-centre hours since release
  lon, lat, cs   (nt, np) float32 tidal-day-mean positions
  in_h    (nt, np) uint8, hours of the 25-h window spent inside the cove
  lon0, lat0, cs0, i0, j0, quad0   (np,) the raw starting position, rho cell
          and quadrant code (0 inner-N, 1 inner-S, 2 outer-N, 3 outer-S)

Output: LO_output/DM_outs/20261013_pcmap_tidalday_reduce/<gtx>/<dir>__<release>.p

run 20261013_pcmap_tidalday_reduce.py
run 20261013_pcmap_tidalday_reduce.py -step 6 -clobber
run 20261013_pcmap_tidalday_reduce.py -dir_glob pcret_3d    (mac test)
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from pcmap_regions import regions

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-dir_glob', default='pcmap_3d*')
p.add_argument('-days', type=float, default=14.0, help='common record length')
p.add_argument('-step', type=int, default=12, help='hours between window centres')
p.add_argument('-win', type=int, default=25, help='boxcar length [h], odd')
p.add_argument('-clobber', action='store_true')
p.add_argument('-done_only', action='store_true',
               help='only releases the launcher logged as finished (returncode 0)')
args = p.parse_args()
assert args.win % 2 == 1

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcmap_tidalday_reduce' / args.gtx
Lfun.make_dir(out_dir)

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape
QUAD = regions(Ldir, lon, lat)['QUAD']

fns = sorted(f for dd in sorted(trk.glob(args.dir_glob)) if dd.is_dir()
             for f in dd.glob('release_*.nc'))
if args.done_only:
    T = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_launch' / 'pcmap_timing.csv')
    ok = set(T.log[T.returncode == 0].str.replace('.log', '', regex=False))
    fns = [f for f in fns if f.parent.name in ok]
print('%d release files under %s/%s%s' % (len(fns), trk.name, args.dir_glob,
                                        ' (finished only)' if args.done_only else ''))
nf = int(round(args.days * 24)) + 1
hw = args.win // 2
tc = np.arange(hw, nf - hw, args.step) # window centres [h]


def boxcar(a):
    """Running mean over args.win frames along axis 0, at the centres tc; NaN if any frame is NaN."""
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])
    return ((c[tc + hw + 1] - c[tc - hw]) / args.win).astype(np.float32)


for fn in fns:
    out_fn = out_dir / ('%s__%s.p' % (fn.parent.name, fn.stem))
    if out_fn.is_file() and not args.clobber:
        continue
    d = xr.open_dataset(fn)
    if d.sizes['Time'] < nf:
        print('SKIP %s: %d frames < %d' % (out_fn.name, d.sizes['Time'], nf))
        d.close()
        continue
    plon = d.lon.values[:nf]; plat = d.lat.values[:nf]; cs = d.cs.values[:nf]
    t0 = pd.Timestamp(d.ot.values[0])
    d.close()

    okp = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int); j = np.zeros(plon.shape, dtype=int)
    i[okp] = np.clip(np.round((plon[okp] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[okp] = np.clip(np.round((plat[okp] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    q = np.where(okp, QUAD[j, i], -1)
    keep = q[0] >= 0
    plon, plat, cs, q, i, j = plon[:, keep], plat[:, keep], cs[:, keep], q[:, keep], i[:, keep], j[:, keep]

    inside = (q >= 0).astype(float)
    c_in = np.concatenate([np.zeros((1, inside.shape[1])), np.cumsum(inside, axis=0)])
    out = dict(
        meta=dict(file=str(fn.relative_to(trk)), dir=fn.parent.name, t0=t0,
                  set=next((s for s in ('E', 'F') if '_%s_' % s in fn.parent.name), ''),
                  nf=nf, days=args.days, step=args.step, win=args.win, n_dropped=int((~keep).sum())),
        t_h=tc.astype(np.float32),
        lon=boxcar(plon), lat=boxcar(plat), cs=boxcar(cs),
        in_h=(c_in[tc + hw + 1] - c_in[tc - hw]).astype(np.uint8),
        lon0=plon[0], lat0=plat[0], cs0=cs[0].astype(np.float32),
        i0=i[0].astype(np.int16), j0=j[0].astype(np.int16), quad0=q[0].astype(np.int8))
    pickle.dump(out, open(out_fn, 'wb'))
    # residual drift between the first two windows a day apart, as a sanity line
    k = 24 // args.step
    dx = (out['lon'][k] - out['lon'][0]) * 111.32e3 * np.cos(np.deg2rad(out['lat'][0]))
    dy = (out['lat'][k] - out['lat'][0]) * 111.32e3
    print('%-50s t0 %s  np %d  nt %d  median residual speed day 1: %.1f cm/s  fully inside last window: %.2f'
          % (out_fn.name, t0, keep.sum(), len(tc), 100 * np.nanmedian(np.hypot(dx, dy)) / (24 * 3600),
             (out['in_h'][-1] == args.win).mean()))
