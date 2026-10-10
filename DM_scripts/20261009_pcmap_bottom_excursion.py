"""
Bottom velocity and bottom tidal excursion in Penn Cove, composited over the
same release classes as the pcmap residence-time results.

Two trapping processes in the inner cove: (1) recirculating eddies, (2) small
velocities. This script is for (2): where is the bottom flow too weak for
water to reach the mouth in a tidal cycle?

PER RELEASE (the 218 pcmap releases of 2025, E + F, every 3rd lunar day), over
the first 3 lunar days after t0 (3 x 25 hourly samples; the class window in
20261006_pcmap_retention_bulk.py is the first 3 d):
  speed      mean |u_bot| over the 75 h [m/s]: bottom layer (s_rho index 0)
             of the pc_cove box, C-grid faces averaged to rho (masked faces =
             0, the wall value)
  excursion  for each 25-h block: subtract the block-mean velocity, integrate
             hourly -> displacement path X(t); excursion = the largest
             distance between any two points of that path (the full ebb or
             flood stroke, the big one on a mixed-tide day). Mean of the 3
             blocks [m]. Eulerian: the excursion at a fixed point, not a
             particle's path.
DISTANCE TO MOUTH: shortest along-water path from each cove cell to the pc_lp
  u-face (8-connected cells, no corner cutting across land, dx = 1/pm,
  dy = 1/pn), so it goes around headlands, not straight through them.
RATIO  excursion / distance to mouth. >= 1: one bottom stroke can carry water
  from the cell to the mouth.

Classes are read from the per-release CSVs that 20261006_pcmap_retention_bulk.py
wrote (season, spring/neap, strat, wind, tide form; year-wide terciles).

Output: LO_output/DM_outs/20261009_pcmap_bottom_excursion/
  pcmap_bottom_excursion_<gtx>_releases.p      per-release maps (cache)
  pcmap_bottom_excursion_distance.png          distance to mouth
  pcmap_bottom_excursion_<family>.png          rows = classes; columns =
      mean bottom speed | excursion (+ contour where excursion = distance to
      mouth) | ratio. Same colour scales in every figure.
  pcmap_bottom_excursion_summary.csv           per class: inner/outer means and
      the fraction of cells with ratio >= 1

  python 20261009_pcmap_bottom_excursion.py
  python 20261009_pcmap_bottom_excursion.py -redo True
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import cmocean
from matplotlib.colors import LogNorm

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-box', default='pc_cove_2024.01.01_2025.12.31', type=str)
parser.add_argument('-ndays', default=3, type=int) # lunar days per release window
parser.add_argument('-redo', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
box_fn = Ldir['LOo'] / 'extract' / args.gtagex / 'box' / (args.box + '.nc')
cls_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_retention_bulk' / args.gtagex
out_dir = Ldir['LOo'] / 'DM_outs' / '20261009_pcmap_bottom_excursion'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
NB = 25 # hours per block (one lunar day, 24.84 h)
NH = NB * args.ndays

# ---------------------------------------------------------------- classes ---
T = pd.read_csv(cls_dir / 'pcmap_retention_bulk_cove_strat_releases.csv')[['t0', 'set', 'season', 'strat']]
for fn, col, new in [('tri_SN', 'springneap', 'springneap'), ('wind', 'class_year', 'wind'), ('tideF', 'class_year', 'tideF')]:
    d = pd.read_csv(cls_dir / ('pcmap_retention_bulk_cove_%s_releases.csv' % fn))
    T = T.merge(d[['t0', 'set', col]].rename(columns={col: new}), on=['t0', 'set'])
T['t0'] = pd.to_datetime(T.t0)
T = T.sort_values(['t0', 'set']).reset_index(drop=True)
print('%d releases' % len(T))

# ------------------------------------------------------------------- grid ---
ds = xr.open_dataset(box_fn)
lon, lat = ds.lon_rho.values, ds.lat_rho.values
NR, NC = lon.shape
DX, DY = 1 / ds.pm.values, 1 / ds.pn.values
iu_mouth = int(np.argmin(np.abs(ds.lon_u.values[0] - pd.read_pickle(sect_dir / 'pc_lp.p').x.mean())))
cove = (ds.mask_rho.values == 1) & (np.arange(NC)[None, :] <= iu_mouth) # rho col iu is just west of u-face iu
print('%d cove cells, pc_lp = u col %d (lon %.4f)' % (cove.sum(), iu_mouth, ds.lon_u.values[0, iu_mouth]))

# ------------------------------------------------- distance to the mouth ---
idx = -np.ones((NR, NC), dtype=int)
idx[cove] = np.arange(cove.sum())
N = cove.sum()
r0, c0, w0 = [], [], []
for dj, di in [(0, 1), (1, 0), (1, 1), (1, -1)]:
    for j in range(NR):
        for i in range(NC):
            j2, i2 = j + dj, i + di
            if not (0 <= j2 < NR and 0 <= i2 < NC and cove[j, i] and cove[j2, i2]):
                continue
            if dj and di and not (cove[j, i2] and cove[j2, i]): # no corner cutting
                continue
            ddx = 0.5 * (DX[j, i] + DX[j2, i2]) * di
            ddy = 0.5 * (DY[j, i] + DY[j2, i2]) * dj
            r0.append(idx[j, i]); c0.append(idx[j2, i2]); w0.append(np.hypot(ddx, ddy))
# virtual source node N joined to the mouth column at half a cell
jm = np.flatnonzero(cove[:, iu_mouth])
r0 += [N] * len(jm); c0 += list(idx[jm, iu_mouth]); w0 += list(0.5 * DX[jm, iu_mouth])
G = coo_matrix((w0, (r0, c0)), shape=(N + 1, N + 1)).tocsr()
dist = np.full((NR, NC), np.nan)
dist[cove] = dijkstra(G, directed=False, indices=N)[:N]
print('distance to mouth: max %.2f km' % (np.nanmax(dist) / 1e3))

# ------------------------------------------------------ per-release maps ---
cache = out_dir / ('pcmap_bottom_excursion_%s_releases.p' % args.gtagex)
if cache.is_file() and not args.redo:
    S, E = pickle.load(open(cache, 'rb'))
else:
    ot = pd.to_datetime(ds.ocean_time.values)
    it0 = np.searchsorted(ot, T.t0.min())
    it1 = np.searchsorted(ot, T.t0.max()) + NH + 1
    print('reading bottom u, v: %s to %s' % (ot[it0], ot[it1 - 1]))
    u = ds.u.isel(s_rho=0, ocean_time=slice(it0, it1)).values
    v = ds.v.isel(s_rho=0, ocean_time=slice(it0, it1)).values
    ot = ot[it0:it1]
    u, v = np.nan_to_num(u), np.nan_to_num(v)
    ur = np.full((len(ot), NR, NC), np.nan, dtype=np.float32)
    vr = np.full((len(ot), NR, NC), np.nan, dtype=np.float32)
    ur[:, :, 1:-1] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    vr[:, 1:-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    ur[:, ~cove], vr[:, ~cove] = np.nan, np.nan
    assert np.isfinite(ur[:, cove]).all() and np.isfinite(vr[:, cove]).all(), 'cove cell on the box edge'
    S = np.full((len(T), NR, NC), np.nan)
    E = np.full((len(T), NR, NC), np.nan)
    for n, t0 in enumerate(T.t0):
        k = np.searchsorted(ot, t0)
        assert ot[k] == t0
        a, b = ur[k:k + NH], vr[k:k + NH]
        S[n] = np.hypot(a, b).mean(axis=0)
        ex = []
        for m in range(args.ndays):
            aa = a[m * NB:(m + 1) * NB] - a[m * NB:(m + 1) * NB].mean(axis=0)
            bb = b[m * NB:(m + 1) * NB] - b[m * NB:(m + 1) * NB].mean(axis=0)
            X = np.concatenate([np.zeros((1, NR, NC)), np.cumsum(aa, axis=0) * 3600])
            Y = np.concatenate([np.zeros((1, NR, NC)), np.cumsum(bb, axis=0) * 3600])
            ex.append(np.hypot(X[:, None] - X[None, :], Y[:, None] - Y[None, :]).max(axis=(0, 1)))
        E[n] = np.mean(ex, axis=0)
    pickle.dump((S, E), open(cache, 'wb'))
    print('saved ' + str(cache))
R = E / dist[None]

# ---------------------------------------------------------------- figures ---
plon, plat = pfun.get_plon_plat(lon, lat)
sec = {k: pd.read_pickle(sect_dir / (k + '.p')) for k in ['pc_cp', 'pc_lp', 'pc_ew']}
ci = np.flatnonzero(cove.any(axis=0))
cj = np.flatnonzero(cove.any(axis=1))
aa = [plon[0, ci[0]], plon[0, ci[-1] + 1], plat[cj[0], 0], plat[cj[-1] + 1, 0]]
inner = cove & (lon < sec['pc_cp'].x.mean())
FAM = [('all', None, ['all']),
       ('season', 'season', ['Dec-Mar', 'Apr-Jul', 'Aug-Nov']),
       ('springneap', 'springneap', ['neap', 'mid', 'spring']),
       ('strat', 'strat', ['weak', 'mid', 'strong']),
       ('wind', 'wind', ['down-cove', 'mid', 'up-cove']),
       ('tideF', 'tideF', ['semidiurnal', 'mid', 'diurnal'])]
FAM_T = {'all': 'All releases', 'season': 'Season', 'springneap': 'Spring/neap (qprism terciles)',
         'strat': 'Stratification (cove drho terciles)', 'wind': 'Along-cove wind (terciles)',
         'tideF': 'Tide form (diurnal/semidiurnal terciles)'}

def comp(sel):
    return np.nanmean(S[sel], axis=0), np.nanmean(E[sel], axis=0), np.nanmean(R[sel], axis=0)

COMP = {}
for fam, col, classes in FAM:
    for c in classes:
        sel = np.ones(len(T), bool) if col is None else (T[col] == c).values
        COMP[(fam, c)] = (sel.sum(),) + comp(sel)
smax = 100 * np.nanpercentile(np.stack([v[1] for v in COMP.values()])[:, cove], 99)
emax = np.nanpercentile(np.stack([v[2] for v in COMP.values()])[:, cove], 99) / 1e3

def setup(ax):
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    for k, ls in [('pc_cp', '-'), ('pc_lp', '-'), ('pc_ew', ':')]:
        ax.plot(sec[k].x, sec[k].y, color='0.3', linestyle=ls, linewidth=0.8)
    ax.axis(aa)
    pfun.dar(ax)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis='x', labelrotation=30)

def masked(f):
    return np.where(cove, f, np.nan)

# distance to mouth
fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
cs = ax.pcolormesh(plon, plat, masked(dist / 1e3), cmap=cmocean.cm.deep, shading='flat')
ax.contour(lon, lat, masked(dist / 1e3), levels=np.arange(1, 8), colors='w', linewidths=0.6)
setup(ax)
fig.colorbar(cs, ax=ax, shrink=0.85, label='Along-water distance to pc_lp [km]')
ax.set_title('Distance to the mouth (white contours every 1 km)', fontsize=10)
fn = out_dir / 'pcmap_bottom_excursion_distance.png'
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))

rows = []
for fam, col, classes in FAM:
    nr = len(classes)
    fig, axs = plt.subplots(nr, 3, figsize=(16, 2.9 * nr + 0.8), layout='constrained', squeeze=False)
    for r, c in enumerate(classes):
        n, s, e, q = COMP[(fam, c)]
        ax = axs[r, 0]
        h0 = ax.pcolormesh(plon, plat, masked(100 * s), cmap=cmocean.cm.speed, vmin=0, vmax=smax, shading='flat')
        ax.set_ylabel('%s (n=%d)' % (c, n), fontsize=11)
        ax = axs[r, 1]
        h1 = ax.pcolormesh(plon, plat, masked(e / 1e3), cmap=cmocean.cm.amp, vmin=0, vmax=emax, shading='flat')
        ax.contour(lon, lat, masked(q), levels=[1], colors='k', linewidths=1.5)
        ax.contour(lon, lat, masked(q), levels=[0.5], colors='k', linewidths=1, linestyles='--')
        ax = axs[r, 2]
        h2 = ax.pcolormesh(plon, plat, masked(q), cmap=cmocean.cm.balance, norm=LogNorm(0.1, 10), shading='flat')
        ax.contour(lon, lat, masked(q), levels=[1], colors='k', linewidths=1.5)
        for ax in axs[r]:
            setup(ax)
        for reg, m in [('inner', inner), ('outer', cove & ~inner), ('cove', cove)]:
            rows.append(dict(family=fam, cls=c, n=n, region=reg, speed_cm_s=100 * s[m].mean(), excursion_km=e[m].mean() / 1e3,
                             dist_km=dist[m].mean() / 1e3, frac_ratio_ge1=(q[m] >= 1).mean(), frac_ratio_ge05=(q[m] >= 0.5).mean()))
    axs[0, 0].set_title('Mean bottom speed', fontsize=11)
    axs[0, 1].set_title('Bottom tidal excursion\n(solid = distance to mouth, dashed = half)', fontsize=11)
    axs[0, 2].set_title('Excursion / distance to mouth', fontsize=11)
    fig.colorbar(h0, ax=axs[:, 0], shrink=0.8, label='[cm/s]')
    fig.colorbar(h1, ax=axs[:, 1], shrink=0.8, label='[km]')
    fig.colorbar(h2, ax=axs[:, 2], shrink=0.8, label='ratio')
    fig.suptitle('%s: bottom layer, first %d lunar days after each release (2025)' % (FAM_T[fam], args.ndays))
    fn = out_dir / ('pcmap_bottom_excursion_%s.png' % fam)
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))

D = pd.DataFrame(rows)
D.to_csv(out_dir / 'pcmap_bottom_excursion_summary.csv', index=False, float_format='%.4g')
pd.set_option('display.width', 200)
print(D.round(3).to_string(index=False))
