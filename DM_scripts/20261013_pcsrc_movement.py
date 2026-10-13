"""
How pcsrc particles move, one release at a time, from its reduced file
(20261013_pcsrc_reduce.py) -- no tracks needed, so it runs on the Mac.

  static  paths of the upper-Saratoga particles that reach Penn Cove within
          -days, coloured by STARTING DEPTH (start = dot), and per depth band
          the cumulative fraction of that band that has entered the cove
  tracks  map only: every upper-Saratoga particle that enters Penn Cove within
          -days, drawn from its start to the hour it FIRST enters the cove
          (the route in); -full draws the whole record instead. Grouped and
          coloured by the DEPTH AT ENTRY (-band_on entry, default: Z at the first
          hourly frame inside the cove, so up to an hour after the crossing) or
          by starting depth (-band_on start) or by DO at entry (-band_on do_entry,
          bands -do_bands 2,5 mg/L); -by_band puts each group in its own panel.
          -color DO colours every track by the DO along its path, with
          everything below 2 mg/L one crimson so the hypoxic threshold is a hard
          break (needs 20261013_pcsrc_do_path.py output). Prints the start x
          entry depth table, and with DO the entry depth x entry DO table.
  movie   every particle as a dot, hourly: upper Saratoga coloured by starting
          depth, the cove cohort (pcmap twin) grey; below the map the cove-mean
          sea level and the fraction of cove volume that is upper-Saratoga water
          (labelled volume / tef2 cove volume, particle volumes at the release
          tide as in 20261013_pcsrc_source.py, hourly, unsmoothed)

POSITIONS are the reduced cell centres (about 200 m). Particles that share a
cell would sit on one point, so for DISPLAY ONLY each particle gets a fixed
random offset of up to +-half a cell. Nothing is computed from the offsets.

Starting-depth bands (z0, m below the surface at release): -bands, default
0,5,15,30 -> 0-5, 5-15, 15-30, >30 m. The cove mouth (pc_lp) is 16-27 m deep.

Output: LO_output/DM_outs/20261013_pcsrc_movement/<gtx>/<sub_tag>_<mode>.png/.mp4

run 20261013_pcsrc_movement.py -rel F_2025.08.15
run 20261013_pcsrc_movement.py -rel F_2025.08.15 -mode movie -days 7
run 20261013_pcsrc_movement.py -rel F_2025.08.15 -mode tracks [-full] [-by_band]
run 20261013_pcsrc_movement.py -rel F_2025.08.15 -mode tracks -color DO -band_on do_entry -by_band
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, Normalize
import cmocean

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

import pcsrc_regions as pr

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-rel', default='F_2025.08.15', help='sub_tag, e.g. F_2025.08.15')
p.add_argument('-mode', default='static', choices=['static', 'movie', 'both', 'tracks'])
p.add_argument('-full', action='store_true',
               help='tracks: draw the whole -days record, not just up to entry')
p.add_argument('-by_band', action='store_true',
               help='tracks: one panel per depth band')
p.add_argument('-band_on', default='entry', choices=['entry', 'start', 'do_entry'],
               help='tracks: group by depth at first entry to the cove, depth at the start, '
                    'or DO at first entry (-do_bands)')
p.add_argument('-color', default='band', choices=['band', 'DO'],
               help='tracks: one colour per group, or DO along the path (needs 20261013_pcsrc_do_path.py)')
p.add_argument('-do_bands', default='2,5', help='tracks: DO-at-entry band edges [mg/L]')
p.add_argument('-do_max', type=float, default=10.0, help='tracks: top of the DO colour scale [mg/L]')
p.add_argument('-entry_bands', default='0,5,10,15',
               help='tracks: entry-depth band edges [m]; the mouth is 16-27 m deep')
p.add_argument('-days', type=float, default=7.0)
p.add_argument('-bands', default='0,5,15,30', help='starting-depth band edges [m]')
p.add_argument('-step', type=int, default=1, help='movie: hours per frame')
p.add_argument('-fps', type=int, default=12)
p.add_argument('-nmax', type=int, default=250, help='static: max paths drawn (random sample)')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
red_fn = Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_reduce' / args.gtx / ('%s.p' % args.rel)
out_dir = Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_movement' / args.gtx
Lfun.make_dir(out_dir)

D = pickle.load(open(red_fn, 'rb'))
meta, P, H = D['meta'], D['P'], D['H']
nf = min(meta['nf'], int(round(args.days * 24)) + 1)
t = pd.DatetimeIndex(meta['ot'][:nf])

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h = g.lon_rho.values, g.lat_rho.values, g.h.values
A = 1 / (g.pm.values * g.pn.values)
mask = g.mask_rho.values
g.close()
wet = mask == 1
R = pr.regions(Ldir, lon, lat, wet)
dlon = float(np.diff(lon[0, :]).mean()); dlat = float(np.diff(lat[:, 0]).mean())

J = H['J'][:nf].astype(int); I = H['I'][:nf].astype(int)
ok = J >= 0
Jc, Ic = np.where(ok, J, 0), np.where(ok, I, 0)
rng = np.random.default_rng(0)
jx = rng.uniform(-0.5, 0.5, J.shape[1]) * dlon
jy = rng.uniform(-0.5, 0.5, J.shape[1]) * dlat
X = np.where(ok, lon[Jc, Ic] + jx, np.nan)
Y = np.where(ok, lat[Jc, Ic] + jy, np.nan)
in_cove = ok & R['cove'][Jc, Ic]

spu = (P.cohort == 'spu').values
z0 = -P.z0.values                                   # m below surface at release
edges = [float(e) for e in args.bands.split(',')] + [np.inf]
band = np.digitize(z0, edges[1:-1])                 # 0 .. nb-1
nb = len(edges) - 1
BLAB = ['%g-%g m' % (edges[k], edges[k + 1]) if np.isfinite(edges[k + 1]) else '>%g m' % edges[k]
        for k in range(nb)]
BCOL = [cmocean.cm.deep(x) for x in np.linspace(0.25, 0.9, nb)]
COVE_C = '#8c8a85'

# cove volume, cove-mean sea level and the upper-Saratoga fraction of the cove
segds = xr.open_dataset(Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
                        / 'segments_2024.01.01_2025.12.31_wb1_pc1_trapsN00.nc')
Vseg = segds.volume.to_pandas()
segds.close()
V_cove = Vseg[pr.COVE_SEGS].sum(axis=1)
mc = R['cove']
zeta_c = ((V_cove - (h[mc] * A[mc]).sum()) / A[mc].sum()).reindex(t, method='nearest').values
ms = R['spu']
zeta_s0 = ((Vseg['pc_lp_p'].reindex([t[0]], method='nearest').values[0]
            - (h[ms] * A[ms]).sum()) / A[ms].sum())
hh = h[P.j0.values, P.i0.values]
w = P.vol.values * np.maximum(hh + np.where(spu, zeta_s0, zeta_c[0]), 0) / hh
f_spu = (in_cove[:, spu] * w[spu]).sum(axis=1) / V_cove.reindex(t, method='nearest').values

entered = spu & in_cove.any(axis=0)
k_enter = np.where(in_cove.any(axis=0), np.argmax(in_cove, axis=0), -1)
print('%s: %d upper-SP particles, %d (%.0f%%) reach the cove within %.1f d'
      % (args.rel, spu.sum(), entered.sum(), 100 * entered.sum() / spu.sum(), args.days))
for k in range(nb):
    m = spu & (band == k)
    print('  start %-8s %5d particles, %5.1f%% reach the cove' % (BLAB[k], m.sum(),
                                                               100 * entered[m].sum() / max(m.sum(), 1)))

# ---------------------------------------------------------------- map ---
xs = [lon[ms].min(), lon[ms].max(), lon[mc].min()]
ys = [lat[ms].min(), lat[ms].max(), lat[mc].min(), lat[mc].max()]
aa = [min(xs) - 4 * dlon, max(xs) + 4 * dlon, min(ys) - 4 * dlat, max(ys) + 4 * dlat]
hm = np.ma.masked_where(mask == 0, h)
sect_dir = Ldir['LOo'] / 'extract' / 'tef2' / 'sections_wb1_pc1'


def base_map(ax):
    # light depth shading so the darkest (deep-start) dots stay visible
    ax.pcolormesh(lon, lat, hm, cmap='Greys', vmin=0, vmax=400, shading='nearest', zorder=1)
    pfun.add_coast(ax, color='k', linewidth=0.6)
    for sn in ['pc_lp', 'sp_mid', 'skagit_sp']:
        s = pickle.load(open(sect_dir / (sn + '.p'), 'rb'))
        ax.plot(s.x.values.astype(float), s.y.values.astype(float), '-', color='k', lw=2, zorder=7)
    ax.contour(lon, lat, mc.astype(float), [0.5], colors='k', linewidths=1.0, zorder=7)
    pfun.dar(ax)
    ax.axis(aa)
    ax.tick_params(labelsize=9)


def depth_legend(ax, cove=True):
    hd = [Line2D([], [], marker='o', ls='none', ms=6, mfc=BCOL[k], mec='none',
                 label='upper SP, start ' + BLAB[k]) for k in range(nb)]
    if cove:
        hd.append(Line2D([], [], marker='o', ls='none', ms=6, mfc=COVE_C, mec='none',
                         label='Penn Cove at release'))
    ax.legend(handles=hd, loc='lower left', fontsize=8, framealpha=0.92)


title = '%s  release %s UTC' % (args.rel, t[0].strftime('%Y-%m-%d %H:%M'))

# ------------------------------------------------------------- static ---
if args.mode in ['static', 'both']:
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1], hspace=0.3, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    base_map(ax)
    idx = np.where(entered)[0]
    if len(idx) > args.nmax:
        idx = np.sort(rng.choice(idx, args.nmax, replace=False))
    for k in range(nb):
        kk = idx[band[idx] == k]
        segs = [np.column_stack([X[:, n], Y[:, n]]) for n in kk]
        ax.add_collection(LineCollection(segs, colors=[BCOL[k]], linewidths=0.6, alpha=0.55, zorder=8))
        ax.scatter(X[0, kk], Y[0, kk], s=22, color=BCOL[k], zorder=9, edgecolors='w', linewidths=0.6)
    depth_legend(ax, cove=False)
    ax.set_title('%s\nupper-SP particles that reach Penn Cove within %.0f d: %d of %d '
                 '(%d paths drawn; dot = start)'
                 % (title, args.days, entered.sum(), spu.sum(), len(idx)), fontsize=10, loc='left')
    days = np.arange(nf) / 24
    axc = fig.add_subplot(gs[0, 1])
    for k in range(nb):
        m = spu & (band == k)
        cum = np.array([((k_enter[m] >= 0) & (k_enter[m] <= f)).mean() for f in range(nf)])
        axc.plot(days, cum, color=BCOL[k], lw=2, label=BLAB[k])
    axc.set_xlim(0, days[-1]); axc.set_ylim(0, None)
    axc.set_ylabel('fraction of band that\nhas entered the cove'); axc.grid(lw=0.3, alpha=0.5)
    axc.legend(fontsize=8, title='start depth', title_fontsize=8)
    axf = fig.add_subplot(gs[1, 1])
    axf.plot(days, f_spu, color='#2a78d6', lw=1.2)
    axf.set_xlim(0, days[-1]); axf.set_ylim(0, 1)
    axf.set_ylabel('upper-SP fraction\nof cove volume'); axf.set_xlabel('days since release')
    axf.grid(lw=0.3, alpha=0.5)
    fn_out = out_dir / ('%s_static.png' % args.rel)
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    print('wrote %s' % fn_out)

# ------------------------------------------------------------- tracks ---
if args.mode == 'tracks':
    idx = np.where(entered)[0]
    # route in: start to the first hour inside the cove (inclusive)
    stop = {n: (nf - 1 if args.full else k_enter[n]) for n in idx}
    allx = np.concatenate([X[:stop[n] + 1, n] for n in idx])
    ally = np.concatenate([Y[:stop[n] + 1, n] for n in idx])
    aa = [min(np.nanpercentile(allx, 0.2), lon[mc].min()) - 4 * dlon,
          max(np.nanpercentile(allx, 99.8), lon[mc].max()) + 4 * dlon,
          min(np.nanpercentile(ally, 0.2), lat[mc].min()) - 4 * dlat,
          max(np.nanpercentile(ally, 99.8), lat[mc].max()) + 4 * dlat]
    route = 'whole record' if args.full else 'drawn up to first entry'
    sfx = ('_full' if args.full else '') + ('_colDO' if args.color == 'DO' else '')

    # depth (and DO) at first entry
    zin = np.full(len(spu), np.nan)
    zin[idx] = -H['Z'][k_enter[idx], idx].astype(float)        # m below surface at entry
    eedges = [float(e) for e in args.entry_bands.split(',')] + [np.inf]
    ELAB = ['%g-%g m' % (eedges[k], eedges[k + 1]) if np.isfinite(eedges[k + 1])
            else '>%g m' % eedges[k] for k in range(len(eedges) - 1)]
    eband = np.full(len(spu), -1)
    eband[idx] = np.digitize(zin[idx], eedges[1:-1])
    need_do = args.color == 'DO' or args.band_on == 'do_entry'
    if need_do:
        do_fn = (Ldir['LOo'] / 'DM_outs' / '20261013_pcsrc_do_path' / args.gtx
                 / ('%s_DO.p' % args.rel))
        if not do_fn.is_file():
            raise SystemExit('no DO along the paths for %s: run 20261013_pcsrc_do_path.py on '
                             'apogee and copy %s back' % (args.rel, do_fn.name))
        DOa = pickle.load(open(do_fn, 'rb'))['DO'][:nf].astype(float)
        doin = np.full(len(spu), np.nan)
        doin[idx] = DOa[k_enter[idx], idx]
        dedges = [-np.inf] + [float(e) for e in args.do_bands.split(',')] + [np.inf]
        DLAB = (['<%g mg/L' % dedges[1]]
                + ['%g-%g mg/L' % (dedges[k], dedges[k + 1]) for k in range(1, len(dedges) - 2)]
                + ['>=%g mg/L' % dedges[-2]])
        dband = np.full(len(spu), -1)
        dband[idx] = np.digitize(doin[idx], dedges[1:-1])

    # DO colour scale: everything below 2 mg/L is one crimson, so the hypoxic
    # threshold is a hard break; above it deep_r from dark (2) to light (-do_max)
    CRIMSON = '#e8000d'
    vv = np.linspace(0, args.do_max, 512)
    DOCMAP = ListedColormap([CRIMSON if v < 2 else
                             cmocean.cm.deep_r(0.85 * (v - 2) / (args.do_max - 2)) for v in vv])
    DONORM = Normalize(0, args.do_max)

    if args.band_on == 'entry':
        gband, GLAB, gword = eband, ELAB, 'enter at'
    elif args.band_on == 'do_entry':
        gband, GLAB, gword = dband, DLAB, 'enter with DO'
    else:
        gband, GLAB, gword = band, BLAB, 'start'
    ng = len(GLAB)
    if args.band_on == 'do_entry':
        GCOL = [CRIMSON] + [cmocean.cm.deep_r(x) for x in np.linspace(0.1, 0.7, ng - 1)]
    else:
        GCOL = [cmocean.cm.deep(x) for x in np.linspace(0.25, 0.9, ng)]

    def table(rows, rlab, rname, cols, clab, cname):
        tb = pd.crosstab(pd.Categorical([rlab[b] for b in rows[idx]], rlab),
                         pd.Categorical([clab[b] for b in cols[idx]], clab), dropna=False)
        tb.index.name = rname; tb.columns.name = cname
        return tb.to_string()

    print('\nentrants by start depth (rows) and entry depth (columns):')
    print(table(band, BLAB, 'start depth', eband, ELAB, 'entry depth'))
    print('entry depth of all entrants: median %.1f m, 10-90%% %.1f-%.1f m'
          % (np.median(zin[idx]), np.percentile(zin[idx], 10), np.percentile(zin[idx], 90)))
    if need_do:
        print('\nentrants by entry depth (rows) and DO at entry (columns):')
        print(table(eband, ELAB, 'entry depth', dband, DLAB, 'DO at entry'))
        print('DO at entry: median %.2f mg/L, %.0f%% below 2 mg/L'
              % (np.nanmedian(doin[idx]), 100 * np.nanmean(doin[idx] < 2)))

    def draw_band(ax, k):
        kk = idx[gband[idx] == k]
        if args.color == 'DO':
            segs, vals = [], []
            for n in kk:
                xy = np.column_stack([X[:stop[n] + 1, n], Y[:stop[n] + 1, n]])
                segs.extend(np.stack([xy[:-1], xy[1:]], axis=1))
                vals.extend(DOa[1:stop[n] + 1, n])            # DO at the end of each hour
            ax.add_collection(LineCollection(segs, array=np.array(vals), cmap=DOCMAP, norm=DONORM,
                                             linewidths=0.7, alpha=0.8, zorder=8))
            ax.scatter(X[0, kk], Y[0, kk], s=14, c=DOa[0, kk], cmap=DOCMAP, norm=DONORM,
                       zorder=9, edgecolors='k', linewidths=0.3)
        else:
            segs = [np.column_stack([X[:stop[n] + 1, n], Y[:stop[n] + 1, n]]) for n in kk]
            ax.add_collection(LineCollection(segs, colors=[GCOL[k]], linewidths=0.6, alpha=0.6, zorder=8))
            ax.scatter(X[0, kk], Y[0, kk], s=14, color=GCOL[k], zorder=9, edgecolors='w', linewidths=0.5)
        return len(kk)

    def do_colorbar(fig, axes):
        sm = plt.cm.ScalarMappable(norm=DONORM, cmap=DOCMAP)
        cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02,
                          ticks=[0, 2] + list(range(4, int(args.do_max) + 1, 2)))
        cb.set_label('DO along the path [mg/L]')
        cb.ax.axhline(2, color=CRIMSON, lw=3)              # the hypoxic threshold

    if args.by_band:
        # one panel per band, same extent so they compare directly
        ncol = 3 if ng == 3 else 2
        nrow = int(np.ceil(ng / ncol))
        # height from the map aspect so the panels fill the figure
        asp = (aa[3] - aa[2]) / ((aa[1] - aa[0]) * np.cos(np.deg2rad(np.mean(aa[2:]))))
        fig, axs = plt.subplots(nrow, ncol, figsize=(8 * ncol, (8 * asp + 0.6) * nrow + 0.6),
                                squeeze=False, layout='constrained')
        for k, ax in enumerate(axs.flat[:ng]):
            base_map(ax)
            ne = draw_band(ax, k)
            if args.band_on == 'start':
                nt = int((spu & (band == k)).sum())
                ttl_k = 'start %s: %d of %d enter (%.0f%%)' % (GLAB[k], ne, nt, 100 * ne / max(nt, 1))
            else:
                ttl_k = '%s %s: %d of %d entrants (%.0f%%)' % (gword, GLAB[k], ne, len(idx),
                                                             100 * ne / max(len(idx), 1))
            ax.set_title(ttl_k, fontsize=11, loc='left')
        for ax in axs.flat[ng:]:
            ax.axis('off')
        fig.suptitle('%s   upper-SP particles that enter Penn Cove within %.0f d, by %s; '
                     '%s; dot = start' % (title, args.days, gword.replace('enter', 'entry').replace('entry at', 'depth at entry').replace('entry with DO', 'DO at entry'),
                                         route), fontsize=12, x=0.02, ha='left')
        if args.color == 'DO':
            do_colorbar(fig, axs.ravel().tolist())
        fn_out = out_dir / ('%s_tracks%s_by%s_%gd.png' % (args.rel, sfx, args.band_on, args.days))
    else:
        fig, ax = plt.subplots(figsize=(12, 9))
        base_map(ax)
        hd = []
        for k in range(ng):
            ne = draw_band(ax, k)
            hd.append(Line2D([], [], color=GCOL[k], lw=2, label='%s %s: %d' % (gword, GLAB[k], ne)))
        if args.color == 'DO':
            do_colorbar(fig, ax)
        else:
            ax.legend(handles=hd, loc='lower left', fontsize=9, framealpha=0.92)
        ax.set_title('%s\nupper-SP particles that enter Penn Cove within %.0f d (%d of %d); '
                     '%s; dot = start' % (title, args.days, len(idx), spu.sum(), route),
                     fontsize=10, loc='left')
        fn_out = out_dir / ('%s_tracks%s_%s_%gd.png' % (args.rel, sfx, args.band_on, args.days))
    fig.savefig(fn_out, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % fn_out)

# -------------------------------------------------------------- movie ---
if args.mode in ['movie', 'both']:
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=[5, 1, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0])
    base_map(ax)
    depth_legend(ax)
    cols = np.array([COVE_C if not s else matplotlib.colors.to_hex(BCOL[b])
                     for s, b in zip(spu, band)])
    order = np.argsort(spu)                       # cove cohort drawn first, under
    dots = ax.scatter(X[0, order], Y[0, order], s=3, c=cols[order], zorder=9, edgecolors='none')
    ttl = ax.set_title('', fontsize=10, loc='left')
    days = np.arange(nf) / 24
    axz = fig.add_subplot(gs[1])
    axz.plot(days, zeta_c, color='#0b0b0b', lw=1)
    axz.set_xlim(0, days[-1]); axz.set_ylabel('cove sea\nlevel [m]', fontsize=9)
    axz.grid(lw=0.3, alpha=0.5); axz.tick_params(labelsize=8)
    axf = fig.add_subplot(gs[2])
    axf.plot(days, f_spu, color='#2a78d6', lw=1)
    axf.set_xlim(0, days[-1]); axf.set_ylim(0, 1)
    axf.set_ylabel('upper-SP fraction\nof cove volume', fontsize=9)
    axf.set_xlabel('days since release'); axf.grid(lw=0.3, alpha=0.5); axf.tick_params(labelsize=8)
    marks = [a.axvline(0, color='#e34948', lw=1.5) for a in (axz, axf)]
    frames = np.arange(0, nf, args.step)

    def update(fi):
        dots.set_offsets(np.column_stack([X[fi, order], Y[fi, order]]))
        ttl.set_text('%s\n%s UTC   day %.2f   upper-SP water = %.0f%% of the cove'
                     % (title, t[fi].strftime('%Y-%m-%d %H:%M'), fi / 24, 100 * f_spu[fi]))
        for mk in marks:
            mk.set_xdata([fi / 24, fi / 24])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    fn_out = out_dir / ('%s_movie.mp4' % args.rel)
    anim.save(fn_out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=3000), dpi=130)
    plt.close(fig)
    print('wrote %s (%d frames)' % (fn_out, len(frames)))
