"""
Lagrangian residual velocity in Penn Cove from the pcmap tidal-day-mean
particle paths (20261013_pcmap_tidalday_reduce.py): where are the gyres?

VELOCITY: centred difference of each particle's 25-h-mean position,
  u_k = (x_{k+1} - x_{k-1}) / 24 h, placed at x_k (the mean position at the
  window centre t_k), binned to the nearest rho cell. Includes Stokes drift and
  everything else a water parcel feels, unlike the Eulerian time mean.
  Split by the particle's CURRENT tidal-day-mean height: surface half if
  cs_k >= -0.5. Only samples with t_k <= -tmax (default 72 h: the first 3 d,
  the window the release classes were defined on; the 24-h centred difference
  then spans hours 0-96). Cells with fewer than -nmin samples are blank.

Each sample is one particle at one time, and every release has 2616
particles, so releases carry roughly equal weight.

Output: LO_output/DM_outs/20261013_pcmap_lagr_residual/
  pcmap_lagr_residual_<family>.png   rows = classes; columns = surface /
      bottom half: residual speed (colour) + arrows
  pcmap_lagr_residual_summary.csv    mean u, v [cm/s] (+ = east/out, north)
      per class, layer and quadrant of the CURRENT position

  python 20261013_pcmap_lagr_residual.py
  python 20261013_pcmap_lagr_residual.py -tmax 312
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cmocean

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
from pcmap_regions import regions, QNAMES
from pcmap_tidalday import load, FAM, FAM_T

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-tmax', type=float, default=72, help='last window centre used [h]')
p.add_argument('-nmin', type=int, default=20, help='min samples per cell')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcmap_lagr_residual'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
M_PER_DEG = 111.32e3

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, mask = g.lon_rho.values, g.lat_rho.values, g.mask_rho.values == 1
g.close()
lon_ax, lat_ax = lon[0], lat[:, 0]
NR, NC = lon.shape
REG = regions(Ldir, lon, lat)
cove, QUAD = REG['cove'], REG['QUAD']

REL, T = load(Ldir, args.gtx)
print('%d releases' % len(REL))

# -------------------------------------------------- residual velocity samples ---
S = [] # per release: (j, i, u, v, surf) of every sample
for D in REL:
    t = D['t_h']
    k = np.flatnonzero((np.arange(len(t)) >= 1) & (np.arange(len(t)) <= len(t) - 2) & (t <= args.tmax))
    dt = (t[k + 1] - t[k - 1]) * 3600
    la = D['lat'][k].astype(float)
    u = (D['lon'][k + 1] - D['lon'][k - 1]) * M_PER_DEG * np.cos(np.deg2rad(la)) / dt[:, None]
    v = (D['lat'][k + 1] - D['lat'][k - 1]) * M_PER_DEG / dt[:, None]
    i = np.round((D['lon'][k] - lon_ax[0]) / (lon_ax[1] - lon_ax[0])).astype(int)
    j = np.round((la - lat_ax[0]) / (lat_ax[1] - lat_ax[0])).astype(int)
    ok = np.isfinite(u) & np.isfinite(v) & (i >= 0) & (i < NC) & (j >= 0) & (j < NR)
    S.append((j[ok], i[ok], u[ok], v[ok], (D['cs'][k] >= -0.5)[ok]))


def binned(sel):
    """(u, v, n) per rho cell and layer over the releases in sel."""
    out = {}
    for L, want in [('surf', True), ('bot', False)]:
        su, sv, n = np.zeros((NR, NC)), np.zeros((NR, NC)), np.zeros((NR, NC))
        for r in np.flatnonzero(sel):
            j, i, u, v, s = S[r]
            m = s == want
            np.add.at(su, (j[m], i[m]), u[m]); np.add.at(sv, (j[m], i[m]), v[m]); np.add.at(n, (j[m], i[m]), 1)
        good = (n >= args.nmin) & mask
        out[L] = (np.where(good, su / np.maximum(n, 1), np.nan), np.where(good, sv / np.maximum(n, 1), np.nan), n)
    return out


# ------------------------------------------------------------------- figures ---
plon, plat = pfun.get_plon_plat(lon, lat)
ci, cj = np.flatnonzero(cove.any(axis=0)), np.flatnonzero(cove.any(axis=1))
aa = [plon[0, ci[0]], plon[0, ci[-1] + 1] + 0.03, plat[cj[0], 0] - 0.005, plat[cj[-1] + 1, 0] + 0.012]
win = (lon >= aa[0]) & (lon <= aa[1]) & (lat >= aa[2]) & (lat <= aa[3])
sec = {k: pd.read_pickle(sect_dir / (k + '.p')) for k in ['pc_cp', 'pc_lp', 'pc_ew']}
lat0 = 0.5 * (aa[2] + aa[3])

COMP = {}
for fam, col, cls in FAM:
    for c in cls:
        sel = np.ones(len(REL), bool) if col is None else np.array([D[col] == c for D in REL])
        COMP[(fam, c)] = (sel.sum(), binned(sel))
vmax = 100 * np.nanpercentile(np.concatenate([np.hypot(*B[L][:2])[win].ravel() for _, B in COMP.values() for L in B]), 98)
SCALE = vmax / 100 / 0.25 # m/s per inch of arrow


def setup(ax):
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    for k, ls in [('pc_cp', '-'), ('pc_lp', '-'), ('pc_ew', ':')]:
        ax.plot(sec[k].x, sec[k].y, color='0.3', linestyle=ls, linewidth=0.8)
    ax.axis(aa)
    pfun.dar(ax)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis='x', labelrotation=30)


rows = []
for fam, col, cls in FAM:
    nr = len(cls)
    fig, axs = plt.subplots(nr, 2, figsize=(14, 3.3 * nr + 0.8), layout='constrained', squeeze=False)
    for r, c in enumerate(cls):
        n, B = COMP[(fam, c)]
        for k, (L, lab) in enumerate([('surf', 'surface half'), ('bot', 'bottom half')]):
            ax = axs[r, k]
            u, v, cnt = B[L]
            h = ax.pcolormesh(plon, plat, 100 * np.hypot(u, v), cmap=cmocean.cm.speed, vmin=0, vmax=vmax, shading='flat')
            jj, ii = np.nonzero(np.isfinite(u) & win)
            q = ax.quiver(lon[jj, ii], lat[jj, ii], u[jj, ii], v[jj, ii], scale=SCALE, scale_units='inches', width=0.0025, headwidth=4, color='k', zorder=3)
            setup(ax)
            if r == 0:
                ax.set_title(lab, fontsize=11)
            for qn, qq in enumerate(QNAMES):
                m = (QUAD == qn) & np.isfinite(u)
                w = cnt[m]
                rows.append(dict(family=fam, cls=c, n_rel=n, layer=L, quad=qq, u_cm_s=100 * np.sum(u[m] * w) / w.sum(),
                                 v_cm_s=100 * np.sum(v[m] * w) / w.sum(), n_samples=int(w.sum())))
        axs[r, 0].set_ylabel('%s (n=%d)' % (c, n), fontsize=11)
    ax = axs[0, 1]
    ref = float('%.0g' % (vmax / 100 / 2))
    ax.quiverkey(q, 0.85, 1.04, ref, '%g cm/s' % (100 * ref), labelpos='E', coordinates='axes', fontproperties={'size': 9})
    fig.colorbar(h, ax=axs, shrink=0.6, label='Lagrangian residual speed [cm/s]')
    fig.suptitle('%s: Lagrangian residual velocity (tidal-day-mean particle drift), first %g h' % (FAM_T[fam], args.tmax))
    fn = out_dir / ('pcmap_lagr_residual_%s.png' % fam)
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))

R = pd.DataFrame(rows)
R.to_csv(out_dir / 'pcmap_lagr_residual_summary.csv', index=False, float_format='%.4g')
pd.set_option('display.width', 200)
print(R[R.family.isin(['all', 'strat'])].round(2).to_string(index=False))
