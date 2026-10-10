"""
Bottom speed, tidal excursion and excursion / distance-to-mouth for single
pcmap releases (the spaghetti-tidal ones), over the same window the spaghetti
plots show: the first tidal day after release (hourly frames 0-25 h).
The per-release version of 20261009_pcmap_bottom_excursion.py (which averages
over release classes).

EULERIAN (needs the pc_cove box; mac, or apogee if the box is there)
  bottom layer (s_rho index 0), C-grid faces -> rho (masked faces = 0)
  speed      mean |u_bot| over the 25 hourly velocities
  excursion  subtract the 25-h mean velocity, integrate hourly -> displacement
             path; excursion = largest distance between any two points of it
PARTICLES (needs the track files; apogee)
  the release's particles in the -half of the column at release (cs0 < -0.5 =
  bottom half, the spaghetti split), only those starting in the cove, binned
  by starting cell (nearest rho point, as in the spaghetti scripts)
  speed      path length over the 26 hourly positions / 25 h
  excursion  largest distance between any two of its 26 hourly positions
             (the particle's own path: net drift is NOT removed)
  crossed    fraction of particles from the cell that were east of pc_lp
             (outside the cove) at any hour of the window
DISTANCE TO MOUTH: shortest along-water path from each cove cell to the pc_lp
  u-face (8-connected, no corner cutting, dx = 1/pm, dy = 1/pn).
RATIO  excursion / distance to mouth of the (starting) cell.

The particle reduce is saved per release as a small pickle, so the figures can
be remade on the mac after copying the pickles over (the Eulerian contour is
then added to the particle figure too).

Output: LO_output/DM_outs/20261009_pcmap_release_excursion/
  <out_name>_pex.p                            per-particle reduce
  release_excursion_eulerian.png              rows = releases: speed |
      excursion (solid: = distance to mouth, dashed: = half) | ratio
  release_excursion_particles_<half>.png      rows = releases: the same three
      from the particles + fraction that crossed pc_lp; grey dash-dot on the
      excursion panel = Eulerian excursion = distance to mouth, same window
  release_excursion_summary.csv               inner/outer/cove means per release

apogee: python 20261009_pcmap_release_excursion.py -half bot
apogee: python 20261009_pcmap_release_excursion.py -half surf
mac:    python 20261009_pcmap_release_excursion.py
mac test: python 20261009_pcmap_release_excursion.py -dir pcret_3d -file release_2024.02.15.nc
"""

import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
import cmocean

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun
from pcmap_regions import regions

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-rels', default='E_2025.07.10,F_2025.07.10,F_2025.07.17,F_2025.08.21,F_2025.08.28',
               help='launcher sub_tags (the spaghetti-tidal releases)')
p.add_argument('-dir', default='', help='tracks2 output dir instead of -rels (test)')
p.add_argument('-file', default='', help='release file in -dir')
p.add_argument('-hours', type=int, default=25, help='window [h]: hourly frames 0..hours')
p.add_argument('-half', default='bot', choices=['bot', 'surf', 'all'])
p.add_argument('-box', default='pc_cove_2024.01.01_2025.12.31')
args = p.parse_args()

gridname, tag, ex_name = args.gtx.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261009_pcmap_release_excursion'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
box_fn = Ldir['LOo'] / 'extract' / args.gtx / 'box' / (args.box + '.nc')
rt_fn = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times' / 'pcmap_release_times_2025.csv'
NH = args.hours
M_PER_DEG = 111.32e3


def read_sec(k):
    return pd.read_pickle(sect_dir / (k + '.p'))


LON_MOUTH = read_sec('pc_lp').x.mean()


def dist_to_mouth(cove, DX, DY, imouth):
    """Along-water distance [m] from each cove cell to the u-face east of rho column imouth."""
    NR, NC = cove.shape
    idx = -np.ones((NR, NC), dtype=int)
    N = cove.sum()
    idx[cove] = np.arange(N)
    r0, c0, w0 = [], [], []
    for dj, di in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        for j, i in zip(*np.nonzero(cove)):
            j2, i2 = j + dj, i + di
            if not (0 <= j2 < NR and 0 <= i2 < NC and cove[j2, i2]):
                continue
            if dj and di and not (cove[j, i2] and cove[j2, i]): # no corner cutting
                continue
            r0.append(idx[j, i]); c0.append(idx[j2, i2])
            w0.append(np.hypot(0.5 * (DX[j, i] + DX[j2, i2]) * di, 0.5 * (DY[j, i] + DY[j2, i2]) * dj))
    jm = np.flatnonzero(cove[:, imouth]) # virtual source joined to the mouth column at half a cell
    r0 += [N] * len(jm); c0 += list(idx[jm, imouth]); w0 += list(0.5 * DX[jm, imouth])
    G = coo_matrix((w0, (r0, c0)), shape=(N + 1, N + 1)).tocsr()
    d = np.full((NR, NC), np.nan)
    d[cove] = dijkstra(G, directed=False, indices=N)[:N]
    return d


def stroke(X, Y):
    """Largest distance between any two points of paths X, Y (time first)."""
    return np.hypot(X[:, None] - X[None, :], Y[:, None] - Y[None, :]).max(axis=(0, 1))


# ------------------------------------------------------------- releases ---
if args.dir:
    tdir = trk / args.dir
    REL = [dict(name=args.dir + '_' + args.file.replace('release_', '').replace('.nc', ''), fn=tdir / args.file)]
else:
    rt = pd.read_csv(rt_fn) if rt_fn.is_file() else None
    REL = []
    for r in args.rels.split(','):
        dirs = sorted(d for d in trk.glob('pcmap_3d*_' + r) if d.is_dir())
        fns = sorted(dirs[0].glob('release_*.nc')) if len(dirs) == 1 else []
        name = dirs[0].name if len(dirs) == 1 else (rt.loc[rt.sub_tag == r, 'out_name'].iloc[0] if rt is not None else 'pcmap_3d_' + r)
        REL.append(dict(name=name, fn=fns[0] if len(fns) == 1 else None, sub_tag=r))
        if rt is not None and (rt.sub_tag == r).any():
            REL[-1]['t0'] = pd.Timestamp(rt.loc[rt.sub_tag == r, 't_release'].iloc[0])

# --------------------------------------------- full grid (particle side) ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
glon, glat = g.lon_rho.values, g.lat_rho.values
gDX, gDY = 1 / g.pm.values, 1 / g.pn.values
g.close()
lon_ax, lat_ax = glon[0], glat[:, 0]
gcove = regions(Ldir, glon, glat)['cove']
gimouth = np.flatnonzero(gcove.any(axis=0))[-1]
gdist = dist_to_mouth(gcove, gDX, gDY, gimouth)
ginner = gcove & (glon < read_sec('pc_cp').x.mean())

# ---------------------------------------------------- particle reduce ---
for R in REL:
    pfn = out_dir / (R['name'] + '_pex.p')
    if R['fn'] is not None and R['fn'].is_file():
        d = xr.open_dataset(R['fn'])
        lon = d.lon.values[:NH + 1]; lat = d.lat.values[:NH + 1]; cs0 = d.cs.values[0]
        R['t0'] = pd.Timestamp(d.ot.values[0])
        d.close()
        i0 = np.clip(np.round((lon[0] - lon_ax[0]) / (lon_ax[1] - lon_ax[0])), 0, len(lon_ax) - 1).astype(int)
        j0 = np.clip(np.round((lat[0] - lat_ax[0]) / (lat_ax[1] - lat_ax[0])), 0, len(lat_ax) - 1).astype(int)
        ok = np.isfinite(lon[0]) & gcove[j0, i0]
        lon, lat, cs0, i0, j0 = lon[:, ok], lat[:, ok], cs0[ok], i0[ok], j0[ok]
        X = (lon - lon[0]) * M_PER_DEG * np.cos(np.deg2rad(lat[0]))
        Y = (lat - lat[0]) * M_PER_DEG
        P = dict(t0=R['t0'], i0=i0, j0=j0, cs0=cs0,
                 speed=np.nansum(np.hypot(np.diff(X, axis=0), np.diff(Y, axis=0)), axis=0) / (NH * 3600),
                 exc=np.nanmax(np.hypot(X[:, None] - X[None, :], Y[:, None] - Y[None, :]), axis=(0, 1)),
                 crossed=(lon > LON_MOUTH).any(axis=0), hours=NH)
        pickle.dump(P, open(pfn, 'wb'))
        print('saved %s (%d particles)' % (pfn, ok.sum()))
    if pfn.is_file():
        R['P'] = pickle.load(open(pfn, 'rb'))
        R['t0'] = R['P']['t0']
        if R['P']['hours'] != NH:
            raise SystemExit('%s was reduced with -hours %d' % (pfn, R['P']['hours']))
    if 't0' not in R:
        raise SystemExit('no track file, pickle or release time for ' + R['name'])
HAVE_P = all('P' in R for R in REL)


def pmaps(P):
    """Per starting cell means: speed, excursion, crossed fraction (selected half)."""
    sel = {'bot': P['cs0'] < -0.5, 'surf': P['cs0'] >= -0.5, 'all': np.ones(len(P['cs0']), bool)}[args.half]
    out = {}
    cnt = np.zeros(glon.shape)
    np.add.at(cnt, (P['j0'][sel], P['i0'][sel]), 1)
    for k in ['speed', 'exc', 'crossed']:
        s = np.zeros(glon.shape)
        np.add.at(s, (P['j0'][sel], P['i0'][sel]), P[k][sel].astype(float))
        out[k] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    out['ratio'] = out['exc'] / gdist
    out['n'] = sel.sum()
    return out


# ------------------------------------------------------------ Eulerian ---
HAVE_E = box_fn.is_file()
if HAVE_E:
    ds = xr.open_dataset(box_fn)
    blon, blat = ds.lon_rho.values, ds.lat_rho.values
    NR, NC = blon.shape
    bimouth = int(np.argmin(np.abs(ds.lon_u.values[0] - LON_MOUTH)))
    bcove = (ds.mask_rho.values == 1) & (np.arange(NC)[None, :] <= bimouth)
    bdist = dist_to_mouth(bcove, 1 / ds.pm.values, 1 / ds.pn.values, bimouth)
    binner = bcove & (blon < read_sec('pc_cp').x.mean())
    ot = pd.to_datetime(ds.ocean_time.values)
    for R in REL:
        k = np.searchsorted(ot, R['t0'])
        assert ot[k] == R['t0']
        u = np.nan_to_num(ds.u.isel(s_rho=0, ocean_time=slice(k, k + NH)).values)
        v = np.nan_to_num(ds.v.isel(s_rho=0, ocean_time=slice(k, k + NH)).values)
        ur = np.full((NH, NR, NC), np.nan); vr = np.full((NH, NR, NC), np.nan)
        ur[:, :, 1:-1] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
        vr[:, 1:-1, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        ur[:, ~bcove], vr[:, ~bcove] = np.nan, np.nan
        a, b = ur - ur.mean(axis=0), vr - vr.mean(axis=0)
        X = np.concatenate([np.zeros((1, NR, NC)), np.cumsum(a, axis=0) * 3600])
        Y = np.concatenate([np.zeros((1, NR, NC)), np.cumsum(b, axis=0) * 3600])
        R['E'] = dict(speed=np.hypot(ur, vr).mean(axis=0), exc=stroke(X, Y))
        R['E']['ratio'] = R['E']['exc'] / bdist
else:
    print('no box file (%s): particle figure only' % box_fn)

# ------------------------------------------------------------- figures ---
if HAVE_P:
    for R in REL:
        R['PM'] = pmaps(R['P'])
gplon, gplat = pfun.get_plon_plat(glon, glat)
ci, cj = np.flatnonzero(gcove.any(axis=0)), np.flatnonzero(gcove.any(axis=1))
aa = [gplon[0, ci[0]], gplon[0, ci[-1] + 1], gplat[cj[0], 0], gplat[cj[-1] + 1, 0]]
sec = {k: read_sec(k) for k in ['pc_cp', 'pc_lp', 'pc_ew']}
allspd = [R[m]['speed'] for R in REL for m in ['E', 'PM'] if m in R]
allexc = [R[m]['exc'] for R in REL for m in ['E', 'PM'] if m in R]
smax = 100 * np.nanpercentile(np.concatenate([a[np.isfinite(a)] for a in allspd]), 99)
emax = np.nanpercentile(np.concatenate([a[np.isfinite(a)] for a in allexc]), 99) / 1e3


def setup(ax):
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    for k, ls in [('pc_cp', '-'), ('pc_lp', '-'), ('pc_ew', ':')]:
        ax.plot(sec[k].x, sec[k].y, color='0.3', linestyle=ls, linewidth=0.8)
    ax.axis(aa)
    pfun.dar(ax)
    ax.tick_params(labelsize=8)
    ax.tick_params(axis='x', labelrotation=30)


def row_label(R):
    s = R.get('sub_tag', R['name'])
    ph = {'E': ' (peak ebb)', 'F': ' (peak flood)'}.get(s[0], '')
    return '%s%s\n%s' % (s, ph, R['t0'].strftime('%Y-%m-%d %H:%M'))


def three_cols(fig, axs, rows, plon, plat, lon, lat, cover, extra=None):
    """rows: list of (label, speed, exc, ratio). Returns the mappables."""
    for r, (lab, s, e, q) in enumerate(rows):
        h0 = axs[r, 0].pcolormesh(plon, plat, np.where(cover, 100 * s, np.nan), cmap=cmocean.cm.speed, vmin=0, vmax=smax, shading='flat')
        axs[r, 0].set_ylabel(lab, fontsize=10)
        h1 = axs[r, 1].pcolormesh(plon, plat, np.where(cover, e / 1e3, np.nan), cmap=cmocean.cm.amp, vmin=0, vmax=emax, shading='flat')
        qq = np.where(cover, q, np.nan)
        if np.isfinite(qq).sum() > 3:
            axs[r, 1].contour(lon, lat, np.nan_to_num(qq, nan=0), levels=[1], colors='k', linewidths=1.5)
            axs[r, 1].contour(lon, lat, np.nan_to_num(qq, nan=0), levels=[0.5], colors='k', linewidths=1, linestyles='--')
        if extra is not None:
            extra(r, axs[r, 1])
        h2 = axs[r, 2].pcolormesh(plon, plat, qq, cmap=cmocean.cm.balance, norm=LogNorm(0.1, 10), shading='flat')
        for ax in axs[r]:
            setup(ax)
    fig.colorbar(h0, ax=axs[:, 0], shrink=0.8, label='[cm/s]')
    fig.colorbar(h1, ax=axs[:, 1], shrink=0.8, label='[km]')
    fig.colorbar(h2, ax=axs[:, 2], shrink=0.8, label='ratio')
    axs[0, 1].set_title('Tidal excursion\n(solid = distance to mouth, dashed = half)', fontsize=11)
    axs[0, 2].set_title('Excursion / distance to mouth', fontsize=11)


rows_out = []
nr = len(REL)
if HAVE_E:
    bplon, bplat = pfun.get_plon_plat(blon, blat)
    fig, axs = plt.subplots(nr, 3, figsize=(16, 2.9 * nr + 0.8), layout='constrained', squeeze=False)
    three_cols(fig, axs, [(row_label(R), R['E']['speed'], R['E']['exc'], R['E']['ratio']) for R in REL], bplon, bplat, blon, blat, bcove)
    axs[0, 0].set_title('Mean bottom speed', fontsize=11)
    fig.suptitle('Eulerian, bottom layer: first %d h after each release' % NH)
    fn = out_dir / 'release_excursion_eulerian.png'
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))
    for R in REL:
        for reg, m in [('inner', binner), ('outer', bcove & ~binner), ('cove', bcove)]:
            rows_out.append(dict(release=R.get('sub_tag', R['name']), method='eulerian_bottom', region=reg, speed_cm_s=100 * np.nanmean(R['E']['speed'][m]),
                                 excursion_km=np.nanmean(R['E']['exc'][m]) / 1e3, frac_cells_ratio_ge1=np.mean(R['E']['ratio'][m] >= 1)))

if HAVE_P:
    def eul_contour(r, ax):
        if HAVE_E:
            ax.contour(blon, blat, np.nan_to_num(np.where(bcove, REL[r]['E']['ratio'], np.nan), nan=0), levels=[1], colors='0.5', linewidths=1.2, linestyles='-.')
    fig, axs = plt.subplots(nr, 4, figsize=(21, 2.9 * nr + 0.8), layout='constrained', squeeze=False)
    three_cols(fig, axs[:, :3], [('%s\nn=%d' % (row_label(R), R['PM']['n']), R['PM']['speed'], R['PM']['exc'], R['PM']['ratio']) for R in REL],
               gplon, gplat, glon, glat, gcove, extra=eul_contour)
    for r, R in enumerate(REL):
        h3 = axs[r, 3].pcolormesh(gplon, gplat, np.where(gcove, R['PM']['crossed'], np.nan), cmap=cmocean.cm.tempo, vmin=0, vmax=1, shading='flat')
        setup(axs[r, 3])
    fig.colorbar(h3, ax=axs[:, 3], shrink=0.8, label='fraction')
    axs[0, 0].set_title('Mean particle speed (hourly path)', fontsize=11)
    axs[0, 1].set_title('Particle excursion\n(solid = dist. to mouth, dashed = half%s)' % (', grey = Eulerian' if HAVE_E else ''), fontsize=11)
    axs[0, 3].set_title('Fraction that crossed pc_lp', fontsize=11)
    half_t = {'bot': 'bottom half (cs0 < -0.5)', 'surf': 'surface half (cs0 >= -0.5)', 'all': 'whole column'}[args.half]
    fig.suptitle('Particles, %s, mapped to starting cell: first %d h after each release' % (half_t, NH))
    fn = out_dir / ('release_excursion_particles_%s.png' % args.half)
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))
    for R in REL:
        P = R['P']
        sel = {'bot': P['cs0'] < -0.5, 'surf': P['cs0'] >= -0.5, 'all': np.ones(len(P['cs0']), bool)}[args.half]
        inn = ginner[P['j0'], P['i0']]
        d0 = gdist[P['j0'], P['i0']]
        for reg, m in [('inner', sel & inn), ('outer', sel & ~inn), ('cove', sel)]:
            rows_out.append(dict(release=R.get('sub_tag', R['name']), method='particles_' + args.half, region=reg, n=m.sum(), speed_cm_s=100 * P['speed'][m].mean(),
                                 excursion_km=P['exc'][m].mean() / 1e3, frac_particles_ratio_ge1=np.mean(P['exc'][m] >= d0[m]), frac_crossed=P['crossed'][m].mean()))

D = pd.DataFrame(rows_out)
D.to_csv(out_dir / ('release_excursion_summary_%s.csv' % args.half), index=False, float_format='%.4g')
pd.set_option('display.width', 220)
print(D.round(3).to_string(index=False))
