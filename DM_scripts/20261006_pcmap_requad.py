"""
Switch existing pcmap reduced files (20261005_pcmap_reduce.py output) to the
pc_ew quadrant definition of pcmap_regions.py, in place, without the tracks.

What it can redo exactly from what the reduce stored:
  P.quad0                         from each particle's starting cell (j0, i0)
  curves[<quadrant>], curves[<quadrant>-surf / -bot]
                                  "still inside" and "never left" curves, from
                                  the per-particle hourly inside-the-cove
                                  record (inside_bits); 'cove' is unchanged
What it cannot redo (needs the tracks), so it sets them to NaN:
  P.quad_exit_h, P.reg3_exit_h    time to leave the particle's own quadrant /
                                  3-D region (not used in any figure so far;
                                  rerun the reduce on apogee if needed)

The old values are kept under quad0_colmean, curves_colmean, quad_exit_h_colmean,
reg3_exit_h_colmean, so the change is reversible. meta['quad_def'] = 'pc_ew'
marks a converted file, and converted files are skipped unless -clobber.

Files freshly made by the updated 20261005_pcmap_reduce.py already use pc_ew
and are marked the same way, so this is only for files reduced before
2026-10-06.

run 20261006_pcmap_requad.py
"""
import argparse
import pickle

import numpy as np
import xarray as xr

from lo_tools import Lfun
from pcmap_regions import regions, QNAMES

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-glob', default='pcmap_3d*')
p.add_argument('-clobber', action='store_true')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
REG = regions(Ldir, g.lon_rho.values, g.lat_rho.values)
g.close()
QUAD = REG['QUAD']

n_done = n_skip = 0
moved = []
for fn in sorted(red_dir.glob(args.glob + '.p')):
    D = pickle.load(open(fn, 'rb'))
    if D['meta'].get('quad_def') == 'pc_ew' and not args.clobber:
        n_skip += 1
        continue
    P = D['P']
    q_old = P['quad0_colmean'].values if 'quad0_colmean' in P else P.quad0.values
    q_new = QUAD[P.j0.values, P.i0.values]
    if (q_new < 0).any():
        raise ValueError('%s: %d particles start outside the cove under pcmap_regions'
                         % (fn.name, (q_new < 0).sum()))
    nf = D['meta']['nf']
    ins = np.unpackbits(D['inside_bits'], axis=0)[:nf].astype(bool)
    if ins.shape[1] != len(P):
        raise ValueError('%s: inside_bits has %d particles, P has %d' % (fn.name, ins.shape[1], len(P)))
    if not np.allclose(ins.mean(axis=1), D['curves']['cove']['still'][:nf]):
        raise ValueError('%s: inside_bits does not reproduce the stored cove curve' % fn.name)
    nev = np.minimum.accumulate(ins, axis=0)

    if 'quad0_colmean' not in P:
        P['quad0_colmean'] = q_old
        for c in ['quad_exit_h', 'reg3_exit_h']:
            if c in P:
                P[c + '_colmean'] = P[c].values
    P['quad0'] = q_new
    for c in ['quad_exit_h', 'reg3_exit_h']:
        if c in P:
            P[c] = np.nan
    if 'curves_colmean' not in D:
        D['curves_colmean'] = D['curves']
    s0 = P.surf0.values
    curves = {'cove': D['curves_colmean']['cove']}
    for k, qn in enumerate(QNAMES):
        for gname, m in [(qn, q_new == k), (qn + '-surf', (q_new == k) & s0), (qn + '-bot', (q_new == k) & ~s0)]:
            if m.sum():
                curves[gname] = dict(n=int(m.sum()), still=ins[:, m].mean(axis=1), never=nev[:, m].mean(axis=1))
    D['curves'] = curves
    D['P'] = P
    D['meta']['quad_def'] = 'pc_ew'
    pickle.dump(D, open(fn, 'wb'))
    moved.append((q_new != q_old).sum())
    n_done += 1

print('converted %d files, skipped %d already on pc_ew' % (n_done, n_skip))
if moved:
    print('particles changing quadrant per release: %d to %d (median %d)'
          % (min(moved), max(moved), int(np.median(moved))))
