"""
Spaghetti plots and animations of pcmap particle trajectories, one release at
a time, straight from the track file.

  static  map of every selected particle's path over the first -days (lines,
          start = dot, end = x), plus fractional height in the column vs time
          for the same particles as a median and interquartile band per colour
          group (individual vertical paths are an unreadable random walk)
  movie   particles as dots with fading tails of the last -tail hours, the
          fraction of the selection still inside the cove, and ssh at pc_lp
          with a marker, so the motion can be read against the tide

Particles can be filtered and coloured by where they STARTED (quadrant,
surface/bottom half, from the initial cell exactly as in
20261005_pcmap_reduce.py) or by the DO they started in. DO comes from the
reduced file of that release, so it needs 20261005_pcmap_reduce.py to have
been run on it (on apogee, without -no_do).

Selecting the release: -rel is the sub_tag the launcher used, e.g.
E_2025.01.04; the output directory is found by globbing pcmap_3d*_<rel>. Any
other tracker output can be given with -dir/-file instead.

Output: LO_output/DM_outs/20261005_pcmap_spaghetti/<release>_<mode>_<colour>[...].png/.mp4
(.gif if ffmpeg is not available)

run 20261005_pcmap_spaghetti.py -rel E_2025.01.04
run 20261005_pcmap_spaghetti.py -rel E_2025.01.04 -mode movie -days 7
run 20261005_pcmap_spaghetti.py -rel F_2025.08.10 -color DO -n 300
run 20261005_pcmap_spaghetti.py -rel F_2025.08.10 -quad inner-N,inner-S -half bottom
run 20261005_pcmap_spaghetti.py -dir pcret_3d -file release_2024.02.15.nc -mode both   (mac test)
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

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-rel', default='', help='launcher sub_tag, e.g. E_2025.01.04')
p.add_argument('-dir', default='', help='tracks2 output dir (instead of -rel)')
p.add_argument('-file', default='', help='release file in -dir; default the only one')
p.add_argument('-mode', default='static', choices=['static', 'movie', 'both'])
p.add_argument('-days', type=float, default=7.0, help='how much of the track to show')
p.add_argument('-n', type=int, default=200, help='particles to draw (random, seeded); 0 = all')
p.add_argument('-quad', default='', help='keep only these starting quadrants, e.g. inner-N,inner-S')
p.add_argument('-half', default='', choices=['', 'surface', 'bottom'])
p.add_argument('-do_max', type=float, default=np.nan, help='keep only DO0 <= this [mg/L]')
p.add_argument('-color', default='quad', choices=['quad', 'half', 'DO', 'none'])
p.add_argument('-extent', default='auto', choices=['auto', 'cove'],
               help='auto = fit the drawn paths; cove = Penn Cove and the mouth')
p.add_argument('-step', type=int, default=1, help='movie: hours between frames')
p.add_argument('-tail', type=int, default=12, help='movie: tail length [h]')
p.add_argument('-fps', type=int, default=12)
p.add_argument('-seed', type=int, default=0)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_spaghetti'
Lfun.make_dir(out_dir)
QNAMES = np.array(['inner-N', 'inner-S', 'outer-N', 'outer-S'])
QCOL = dict(zip(QNAMES, ['#e8455e', '#f0a04b', '#4565e8', '#45b0a8']))
HCOL = {'surface': '#f0a04b', 'bottom': '#3b0f70'}

# ---------------------------------------------------------- the release ---
if args.rel:
    dirs = sorted(d for d in trk.glob('pcmap_3d*_' + args.rel) if d.is_dir())
    if len(dirs) != 1:
        raise SystemExit('found %d dirs matching pcmap_3d*_%s in %s' % (len(dirs), args.rel, trk))
    rdir = dirs[0]
elif args.dir:
    rdir = trk / args.dir
else:
    raise SystemExit('give -rel or -dir')
fns = [rdir / args.file] if args.file else sorted(rdir.glob('release_*.nc'))
if len(fns) != 1 or not fns[0].is_file():
    raise SystemExit('need exactly one release file in %s (use -file)' % rdir)
fn = fns[0]
name = rdir.name + ('' if args.rel else '_' + fn.stem)

nf_show = int(round(args.days * 24)) + 1
d = xr.open_dataset(fn)
nf_show = min(nf_show, d.sizes['Time'])
plon = d.lon.values[:nf_show]; plat = d.lat.values[:nf_show]; cs = d.cs.values[:nf_show]
ot = pd.to_datetime(d.ot.values[:nf_show])
d.close()

# ------------------------------------------------- regions, as the reduce ---
g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h, mask = g.lon_rho.values, g.lat_rho.values, g.h.values, g.mask_rho.values
g.close()
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
NR, NC = lon.shape
seg = pickle.load(open(sorted((Ldir['LOo'] / 'extract' / 'tef2').glob(
    'seg_info_dict_wb1_pc1_*.p'))[0], 'rb'))


def seg_mask(names):
    m = np.zeros((NR, NC), dtype=bool)
    for s in names:
        a = np.array(seg[s]['ji_list'])
        m[a[:, 0], a[:, 1]] = True
    return m


# cove, inner/outer and the pc_ew north/south split: the shared definition
from pcmap_regions import regions
REG = regions(Ldir, lon, lat)
cove, inner, north = REG['cove'], REG['inner'], REG['north']
QUAD = REG['QUAD'].copy()

okp = np.isfinite(plon) & np.isfinite(plat)
ip = np.zeros(plon.shape, dtype=int); jp = np.zeros(plon.shape, dtype=int)
ip[okp] = np.clip(np.round((plon[okp] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
jp[okp] = np.clip(np.round((plat[okp] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
q = np.where(okp, QUAD[jp, ip], -1)
inside = q >= 0
keep = q[0] >= 0                      # same particles, same order, as the reduce

# ----------------------------------------------------- choose particles ---
sel = keep.copy()
quad0 = np.where(keep, QNAMES[np.clip(q[0], 0, 3)], '')
half0 = np.where(cs[0] >= -0.5, 'surface', 'bottom')
DO0 = np.full(len(sel), np.nan)
if args.color == 'DO' or np.isfinite(args.do_max):
    red = (Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_reduce' / args.gtx
           / ('%s__%s.p' % (rdir.name, fn.stem)))
    if not red.is_file():
        raise SystemExit('DO needs the reduced file %s -- run 20261005_pcmap_reduce.py first' % red)
    P = pickle.load(open(red, 'rb'))['P']
    if len(P) != keep.sum() or not (P.i0.values == ip[0, keep]).all():
        raise SystemExit('reduced file does not line up with the track file particles')
    DO0[keep] = P.DO0.values
    if np.isnan(DO0[keep]).all():
        raise SystemExit('reduced file has no DO (reduced with -no_do)')
if args.quad:
    sel &= np.isin(quad0, args.quad.split(','))
if args.half:
    sel &= half0 == args.half
if np.isfinite(args.do_max):
    sel &= DO0 <= args.do_max
idx = np.where(sel)[0]
if len(idx) == 0:
    raise SystemExit('no particles left after the filters')
if args.n and len(idx) > args.n:
    idx = np.sort(np.random.default_rng(args.seed).choice(idx, args.n, replace=False))
print('%s: %d particles in the cove, %d after filters, drawing %d, %.1f d from %s'
      % (name, keep.sum(), sel.sum(), len(idx), (nf_show - 1) / 24, ot[0]))

X, Y, CS, INS = plon[:, idx], plat[:, idx], cs[:, idx], inside[:, idx]
if args.color == 'quad':
    cols = [QCOL[k] for k in quad0[idx]]
    legend = [(k, QCOL[k]) for k in QNAMES if k in quad0[idx]]
elif args.color == 'half':
    cols = [HCOL[k] for k in half0[idx]]
    legend = [(k, HCOL[k]) for k in ['surface', 'bottom'] if k in half0[idx]]
elif args.color == 'DO':
    norm = plt.Normalize(0, max(10, np.nanmax(DO0[idx])))
    cmap = plt.get_cmap('RdYlBu')
    cols = [cmap(norm(v)) for v in DO0[idx]]
    legend = None
else:
    cols = ['0.2'] * len(idx)
    legend = None
cols = np.array([matplotlib.colors.to_rgba(c) for c in cols])

# --------------------------------------------------------- ssh at pc_lp ---
ssh = None
hf_fn = Ldir['LOo'] / 'extract' / args.gtx / 'tef2' / 'hourly_flux_2024.01.01_2025.12.31_wb1_pc1.nc'
if hf_fn.is_file():
    hf = xr.open_dataset(hf_fn)
    s_all = pd.Series(hf.ssh.sel(sect='pc_lp').values, index=pd.to_datetime(hf.time.values))
    hf.close()
    # hourly_flux is hour-centred (:30); interpolate onto the tracker clock
    ssh = np.interp(ot.values.astype('int64'), s_all.index.values.astype('int64'), s_all.values)

# ------------------------------------------------------------- the map ---
if args.extent == 'cove':
    aa = [lon[cove].min() - 0.01, lon[cove].max() + 0.09,
          lat[cove].min() - 0.02, lat[cove].max() + 0.03]
else:
    xx, yy = X[np.isfinite(X)], Y[np.isfinite(Y)]
    aa = [np.percentile(xx, 0.5) - 0.01, np.percentile(xx, 99.5) + 0.01,
          np.percentile(yy, 0.5) - 0.01, np.percentile(yy, 99.5) + 0.01]
    aa = [min(aa[0], lon[cove].min() - 0.01), max(aa[1], lon[cove].max() + 0.02),
          min(aa[2], lat[cove].min() - 0.01), max(aa[3], lat[cove].max() + 0.01)]
hm = np.ma.masked_where(mask == 0, h)
days = np.arange(nf_show) / 24


def base_map(ax):
    ax.pcolormesh(lon, lat, hm, cmap='Blues', vmin=0, vmax=80, shading='nearest',
                  alpha=0.6, zorder=1)
    pfun.add_coast(ax, color='k', linewidth=0.6)
    ax.contour(lon, lat, cove.astype(float), [0.5], colors='k', linewidths=1.2, zorder=6)
    pfun.dar(ax)
    ax.axis(aa)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')


def add_legend(ax):
    if legend:
        for lab, c in legend:
            ax.plot([], [], '-', color=c, lw=2, label=lab)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)


filt = ' '.join(t for t in [args.quad, args.half,
                            'DO<=%g' % args.do_max if np.isfinite(args.do_max) else ''] if t)
sfx = '_%s' % args.color + ('_' + filt.replace(' ', '_').replace(',', '-').replace('<=', 'le')
                             if filt else '')
title = '%s  release %s UTC, %d particles%s' % (name, ot[0].strftime('%Y-%m-%d %H:%M'),
                                                len(idx), ('  [' + filt + ']') if filt else '')


def colorbar_do(fig, ax):
    if args.color == 'DO':
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, shrink=0.8, label='DO at release [mg/L]')


# ------------------------------------------------------------ static ---
if args.mode in ['static', 'both']:
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    ax = fig.add_subplot(gs[0]); axz = fig.add_subplot(gs[1])
    base_map(ax)
    segs = [np.column_stack([X[:, k], Y[:, k]]) for k in range(len(idx))]
    lc = LineCollection(segs, colors=cols, linewidths=0.6, alpha=0.6, zorder=8)
    ax.add_collection(lc)
    ax.scatter(X[0], Y[0], s=6, c=cols, zorder=9, edgecolors='none')
    ax.scatter(X[-1], Y[-1], s=12, c=cols, marker='x', linewidths=0.8, zorder=9)
    add_legend(ax)
    colorbar_do(fig, ax)
    ax.set_title(title + '\npaths over the first %.1f d (dot = start, x = end)'
                 % ((nf_show - 1) / 24), fontsize=10)
    if legend:
        zgroups = [(lab, c, np.array([tuple(cc) == tuple(matplotlib.colors.to_rgba(c))
                                      for cc in cols])) for lab, c in legend]
    else:
        zgroups = [('all', '0.2', np.ones(len(idx), dtype=bool))]
    for lab, c, m in zgroups:
        hz = CS[:, m] + 1
        axz.fill_between(days, np.nanpercentile(hz, 25, axis=1), np.nanpercentile(hz, 75, axis=1),
                         color=c, alpha=0.2, lw=0)
        axz.plot(days, np.nanmedian(hz, axis=1), color=c, lw=1.4, label=lab)
    axz.set_xlim(0, days[-1]); axz.set_ylim(0, 1)
    axz.set_ylabel('height in column\n[0 bed, 1 surface]\nmedian, 25-75%'); axz.set_xlabel('days from release')
    axz.grid(color='lightgray', linestyle='--', alpha=0.5)
    axz2 = axz.twinx()
    axz2.plot(days, INS.mean(axis=1), color='k', lw=1.6)
    axz2.set_ylim(0, 1.02); axz2.set_ylabel('fraction still in cove (black)')
    fn_out = out_dir / ('%s_static%s.png' % (name, sfx))
    fig.savefig(fn_out, dpi=200, transparent=True)
    plt.close(fig)
    print('wrote %s' % fn_out)

# ------------------------------------------------------------- movie ---
if args.mode in ['movie', 'both']:
    fig = plt.figure(figsize=(11, 10))
    nrow = 2 if ssh is not None else 1
    gs = fig.add_gridspec(nrow, 1, height_ratios=[4, 1][:nrow], hspace=0.25)
    ax = fig.add_subplot(gs[0])
    base_map(ax)
    add_legend(ax)
    colorbar_do(fig, ax)
    tails = LineCollection([], linewidths=0.8, zorder=8)
    ax.add_collection(tails)
    dots = ax.scatter(X[0], Y[0], s=10, c=cols, zorder=9, edgecolors='none')
    ttl = ax.set_title('', fontsize=10)
    if ssh is not None:
        axs = fig.add_subplot(gs[1])
        axs.plot(days, ssh, color='#3b0f70', lw=1.2)
        axs.set_xlim(0, days[-1]); axs.set_ylabel('ssh at pc_lp [m]')
        axs.set_xlabel('days from release')
        axs.grid(color='lightgray', linestyle='--', alpha=0.5)
        mark = axs.axvline(0, color='k', lw=1.5)
    frames = np.arange(0, nf_show, args.step)

    def update(fi):
        k0 = max(0, fi - args.tail)
        segs, cc = [], []
        for k in range(len(idx)):
            pts = np.column_stack([X[k0:fi + 1, k], Y[k0:fi + 1, k]])
            if len(pts) < 2:
                continue
            seg2 = np.stack([pts[:-1], pts[1:]], axis=1)
            a = np.linspace(0.05, 0.7, len(seg2))          # fade toward the tail end
            c = np.repeat(cols[k][None, :], len(seg2), axis=0)
            c[:, 3] = a
            segs.extend(seg2); cc.extend(c)
        tails.set_segments(segs)
        tails.set_color(cc if cc else 'none')
        dots.set_offsets(np.column_stack([X[fi], Y[fi]]))
        ttl.set_text('%s\n%s UTC   day %.2f   %.0f%% of these still in the cove'
                     % (title, ot[fi].strftime('%Y-%m-%d %H:%M'), fi / 24, 100 * INS[fi].mean()))
        if ssh is not None:
            mark.set_xdata([fi / 24, fi / 24])
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)
    if animation.writers.is_available('ffmpeg'):
        fn_out = out_dir / ('%s_movie%s.mp4' % (name, sfx))
        anim.save(fn_out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
    else:
        fn_out = out_dir / ('%s_movie%s.gif' % (name, sfx))
        anim.save(fn_out, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print('wrote %s (%d frames)' % (fn_out, len(frames)))
