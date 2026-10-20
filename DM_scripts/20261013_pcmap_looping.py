"""
Do pcmap particles recirculate (loop) inside Penn Cove before leaving?
Per-particle looping metrics from the tidal-day-mean paths
(20261013_pcmap_tidalday_reduce.py: 25-h boxcar positions every 12 h).

SEGMENT: each particle's tidal-day-mean path from the first window (centred
  12 h after release) up to, not including, the first window whose mean
  position is east of pc_lp (out of the cove), or the end of the 14 d. The
  cove opens only at pc_lp, so this is the tidally-averaged residence segment.
METRICS (local x/y in metres)
  turn      signed cumulative turning angle of the residual path [rad], + =
            counterclockwise. Headings come from 12-h steps of at least -dmin
            m (shorter steps have no meaningful direction and are skipped;
            the turn is taken between consecutive kept steps). loops = turn/2pi.
            A single in-north / out-south pass through the cove's CCW
            recirculation is a U-turn, |turn| ~ pi; a closed loop is 2pi.
  looper    |loops| >= 1
  returned  came back within -rret m of an earlier mean position at least 2 d
            later, having travelled at least -lret m of residual path in
            between (so sitting still does not count)
  path_km, net_km, res_speed_cm_s   residual path length, start-to-end
            distance, path length / segment duration
  seg_h     segment duration [h]; seg_censored = still inside at 14 d
Joined from 20261005_pcmap_reduce.py: first_exit_h, censored (hourly track),
plus d0_km (along-water distance from the starting cell to pc_lp), quad0,
surf0, and the release classes.

Output: LO_output/DM_outs/20261013_pcmap_looping/
  pcmap_looping_particles.p     per-particle table (all 218 releases)
  pcmap_looping_maps.png        by starting cell, rows surface / bottom
      starters: fraction loopers | mean signed loops | fraction returned
  pcmap_looping_paths_<rel>.png residual paths of one release, surface and
      bottom starters, loopers coloured by sense (CCW blue, CW red)
  pcmap_looping_summary.csv     fractions by class x starting half x quadrant

  python 20261013_pcmap_looping.py
  python 20261013_pcmap_looping.py -paths E_2025.07.10,F_2025.07.17
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import cmocean

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
from pcmap_regions import regions, QNAMES
from pcmap_tidalday import load, dist_to_mouth, FAM

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-dmin', type=float, default=100, help='min 12-h residual step for a heading [m]')
p.add_argument('-rret', type=float, default=250, help='return radius [m]')
p.add_argument('-lret', type=float, default=1500, help='min path between the two visits [m]')
p.add_argument('-paths', default='E_2025.07.10,F_2025.07.10,F_2025.07.17,F_2025.08.21,F_2025.08.28',
               help='releases (sub_tag) for the path figures')
p.add_argument('-npath', type=int, default=300, help='particles drawn per panel')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcmap_looping'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
M_PER_DEG = 111.32e3
LAT0 = 48.23

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat = g.lon_rho.values, g.lat_rho.values
DX, DY = 1 / g.pm.values, 1 / g.pn.values
g.close()
NR, NC = lon.shape
REG = regions(Ldir, lon, lat)
cove = REG['cove']
dist = dist_to_mouth(cove, DX, DY, np.flatnonzero(cove.any(axis=0))[-1])
LON_MOUTH = pd.read_pickle(sect_dir / 'pc_lp.p').x.mean()

REL, T = load(Ldir, args.gtx)
print('%d releases' % len(REL))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def metrics(x, y, nseg, step_h):
    """Looping metrics for one particle; x, y [m] over its segment (nseg points)."""
    x, y = x[:nseg], y[:nseg]
    dx, dy = np.diff(x), np.diff(y)
    ds = np.hypot(dx, dy)
    keep = ds >= args.dmin
    hd = np.arctan2(dy[keep], dx[keep])
    turn = wrap(np.diff(hd)).sum() if hd.size > 1 else 0.0
    cum = np.concatenate([[0], np.cumsum(ds)])
    lag = int(round(48 / step_h))
    ret = False
    for a in range(nseg - lag):
        b = np.arange(a + lag, nseg)
        if np.any((np.hypot(x[b] - x[a], y[b] - y[a]) < args.rret) & (cum[b] - cum[a] >= args.lret)):
            ret = True
            break
    dur = max(nseg - 1, 1) * step_h * 3600
    return turn, ret, cum[-1], np.hypot(x[-1] - x[0], y[-1] - y[0]), cum[-1] / dur


rows = []
PATHS = {}
for D in REL:
    step = float(D['t_h'][1] - D['t_h'][0])
    X = (D['lon'].astype(float) - LON_MOUTH) * M_PER_DEG * np.cos(np.deg2rad(LAT0))
    Y = (D['lat'].astype(float) - LAT0) * M_PER_DEG
    out = D['lon'] > LON_MOUTH
    nseg = np.where(out.any(axis=0), out.argmax(axis=0), out.shape[0])
    sub = next((s for s in args.paths.split(',') if D['name'].endswith(s)), None)
    if sub:
        PATHS[sub] = (D, X, Y, nseg)
    for n in range(X.shape[1]):
        if nseg[n] < 2:
            turn, ret, pl, net, rs = 0.0, False, 0.0, 0.0, np.nan
        else:
            turn, ret, pl, net, rs = metrics(X[:, n], Y[:, n], nseg[n], step)
        rows.append((D['name'], D['meta']['set'], D['meta']['t0'], D['season'], D['strat'], D['springneap'], D['wind'], D['tideF'],
                     n, int(D['quad0'][n]), bool(D['cs0'][n] >= -0.5), dist[D['j0'][n], D['i0'][n]] / 1e3,
                     int(D['i0'][n]), int(D['j0'][n]), turn, ret, pl / 1e3, net / 1e3, 100 * rs,
                     float(D['t_h'][max(nseg[n] - 1, 0)]), bool(nseg[n] == X.shape[0]),
                     float(D['first_exit_h'][n]), bool(D['censored'][n])))
A = pd.DataFrame(rows, columns=['release', 'set', 't0', 'season', 'strat', 'springneap', 'wind', 'tideF', 'p', 'quad0', 'surf0',
                                'd0_km', 'i0', 'j0', 'turn', 'returned', 'path_km', 'net_km', 'res_speed_cm_s', 'seg_h',
                                'seg_censored', 'first_exit_h', 'censored'])
A['loops'] = A.turn / (2 * np.pi)
A['looper'] = A.loops.abs() >= 1
A.to_pickle(out_dir / 'pcmap_looping_particles.p')
print('%d particles; loopers %.3f (CCW %.3f, CW %.3f), returned %.3f, both %.3f'
      % (len(A), A.looper.mean(), (A.loops >= 1).mean(), (A.loops <= -1).mean(), A.returned.mean(), (A.looper & A.returned).mean()))

# --------------------------------------------------------------- summary ---
srow = []
for fam, col, cls in FAM:
    for c in cls:
        a = A if col is None else A[A[col] == c]
        for half, hs in [('surf', True), ('bot', False)]:
            for qn, qq in [(-9, 'cove')] + list(enumerate(QNAMES)):
                b = a[(a.surf0 == hs) & ((a.quad0 == qn) if qn >= 0 else True)]
                srow.append(dict(family=fam, cls=c, half=half, quad=qq, n=len(b), frac_looper=b.looper.mean(),
                                 frac_ccw=(b.loops >= 1).mean(), frac_cw=(b.loops <= -1).mean(), mean_loops=b.loops.mean(),
                                 frac_returned=b.returned.mean(), med_res_speed_cm_s=b.res_speed_cm_s.median(),
                                 med_first_exit_d=b.first_exit_h.median() / 24))
Sm = pd.DataFrame(srow)
Sm.to_csv(out_dir / 'pcmap_looping_summary.csv', index=False, float_format='%.4g')
pd.set_option('display.width', 220)
print(Sm[(Sm.family == 'all')].round(3).to_string(index=False))
print(Sm[(Sm.family != 'all') & (Sm.quad == 'cove')].round(3).to_string(index=False))

# ------------------------------------------------------------------- maps ---
plon, plat = pfun.get_plon_plat(lon, lat)
ci, cj = np.flatnonzero(cove.any(axis=0)), np.flatnonzero(cove.any(axis=1))
aa = [plon[0, ci[0]], plon[0, ci[-1] + 1], plat[cj[0], 0], plat[cj[-1] + 1, 0]]
sec = {k: pd.read_pickle(sect_dir / (k + '.p')) for k in ['pc_cp', 'pc_lp', 'pc_ew']}


def setup(ax, a=aa):
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    for k, ls in [('pc_cp', '-'), ('pc_lp', '-'), ('pc_ew', ':')]:
        ax.plot(sec[k].x, sec[k].y, color='0.3', linestyle=ls, linewidth=0.8)
    ax.axis(a)
    pfun.dar(ax)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis='x', labelrotation=30)


def cellmap(b, col):
    s, n = np.zeros((NR, NC)), np.zeros((NR, NC))
    np.add.at(s, (b.j0.values, b.i0.values), b[col].values.astype(float))
    np.add.at(n, (b.j0.values, b.i0.values), 1)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


fig, axs = plt.subplots(2, 3, figsize=(20, 7.5), layout='constrained')
for r, (hs, lab) in enumerate([(True, 'surface-half starters'), (False, 'bottom-half starters')]):
    b = A[A.surf0 == hs]
    for k, (col, cmap, v0, v1, ttl) in enumerate([('looper', cmocean.cm.amp, 0, None, 'Fraction loopers (|loops| >= 1)'),
                                                   ('loops', cmocean.cm.balance, -0.4, 0.4, 'Mean signed loops (+ = counterclockwise)'),
                                                   ('returned', cmocean.cm.amp, 0, None, 'Fraction returned (within %d m, >= 2 d later)' % args.rret)]):
        f = cellmap(b, col)
        if v1 is None:
            v1 = np.nanpercentile(cellmap(A, col), 99)
        h = axs[r, k].pcolormesh(plon, plat, f, cmap=cmap, vmin=v0, vmax=v1, shading='flat')
        fig.colorbar(h, ax=axs[r, k], shrink=0.85)
        setup(axs[r, k])
        if r == 0:
            axs[r, k].set_title(ttl, fontsize=11)
    axs[r, 0].set_ylabel(lab, fontsize=11)
fig.suptitle('Looping of tidal-day-mean particle paths before leaving the cove, by starting cell (%d releases)' % len(REL))
fn = out_dir / 'pcmap_looping_maps.png'
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))

# ----------------------------------------------------------------- paths ---
ab = [aa[0], aa[1] + 0.02, aa[2] - 0.004, aa[3] + 0.008]
rng = np.random.default_rng(0)
for sub, (D, X, Y, nseg) in PATHS.items():
    a = A[A.release == D['name']]
    fig, axs = plt.subplots(1, 2, figsize=(16, 4.6), layout='constrained')
    for k, (hs, lab) in enumerate([(True, 'surface-half starters'), (False, 'bottom-half starters')]):
        ax = axs[k]
        idx = np.flatnonzero((D['cs0'] >= -0.5) == hs)
        idx = rng.choice(idx, min(args.npath, idx.size), replace=False)
        for colr, sel, lw, z in [('0.75', lambda L: abs(L) < 1, 0.5, 2), ('tab:blue', lambda L: L >= 1, 1.0, 3), ('tab:red', lambda L: L <= -1, 1.0, 3)]:
            segs = []
            for n in idx:
                L = a.loops.values[n]
                if sel(L):
                    m = min(nseg[n] + 1, X.shape[0]) # include the first point outside
                    segs.append(np.column_stack([D['lon'][:m, n], D['lat'][:m, n]]))
            ax.add_collection(LineCollection(segs, colors=colr, linewidths=lw, zorder=z))
        ax.plot(D['lon'][0, idx], D['lat'][0, idx], 'k.', ms=1.5, zorder=4)
        setup(ax, ab)
        b = a[a.surf0 == hs]
        ax.set_title('%s: loopers %.0f%% (CCW %.0f%%, CW %.0f%%), returned %.0f%%'
                     % (lab, 100 * b.looper.mean(), 100 * (b.loops >= 1).mean(), 100 * (b.loops <= -1).mean(), 100 * b.returned.mean()), fontsize=10)
    fig.suptitle('%s (%s): tidal-day-mean paths until first exit, %d of each; blue = CCW looper, red = CW looper, grey = |loops| < 1'
                 % (sub, D['meta']['t0'].strftime('%Y-%m-%d %H:%M'), args.npath), fontsize=11)
    fn = out_dir / ('pcmap_looping_paths_%s.png' % sub)
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))
