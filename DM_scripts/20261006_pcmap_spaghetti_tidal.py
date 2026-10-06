"""
Static spaghetti plots of one pcmap release over ONE TIDAL DAY, split into
the eight starting classes: the four quadrants (inner-N, inner-S, outer-N,
outer-S) x the surface / bottom half of the column (initial cs >= -0.5), the
same definitions as 20261005_pcmap_reduce.py.

One figure per release: rows surface / bottom, columns the four quadrants.
Each panel draws -n particles of that class (random, seeded; 0 = all), their
paths over the first -hours (default 24.84 h, one lunar / tidal day, i.e. both
semidiurnal cycles of this mixed tide; the track is hourly, so that is the 26
frames 0-25 h; -hours 12.42 gives one M2 cycle), coloured by hours since release so the
direction of travel through the tide reads off the colour. Dot = start,
x = end. Map framed on the cove and its mouth. The release starts at peak ebb
(E) or peak flood (F).

Below the maps, a tidal time series over the same window: ssh at pc_lp (tef2
hourly_flux, hour-centred, interpolated onto the track clock) coloured by
hours since release on the same scale as the paths, and the net transport at
pc_lp (positive = out of the cove, ebb; sign taken from the data as in
20261005_pcmap_release_times.py) on the right axis. In the movie a vertical
line marks the current time.

-mode movie (or both) also animates the same window: the same eight panels,
paths growing behind each particle (coloured by hours since release) and the
particles as dots, with ssh at pc_lp and the hour in the title. Positions are
LINEARLY INTERPOLATED between the hourly track frames (-sub frames per hour)
for a smooth movie only; nothing finer than hourly is real.

Reads the track file directly, so it runs on apogee.

Output: LO_output/DM_outs/20261006_pcmap_spaghetti_tidal/<release dir>_tidal.png / _tidal.mp4

run 20261006_pcmap_spaghetti_tidal.py -rel E_2025.07.10
run 20261006_pcmap_spaghetti_tidal.py -rel F_2025.07.10 -n 0
run 20261006_pcmap_spaghetti_tidal.py -rel E_2025.07.10 -mode both
run 20261006_pcmap_spaghetti_tidal.py -dir pcret_3d -file release_2024.02.15.nc   (mac test)
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-rel', default='', help='launcher sub_tag, e.g. E_2025.07.10')
p.add_argument('-dir', default='', help='tracks2 output dir (instead of -rel)')
p.add_argument('-file', default='', help='release file in -dir; default the only one')
p.add_argument('-hours', type=float, default=24.84, help='length of the window to draw [h]; 24.84 = one tidal day')
p.add_argument('-n', type=int, default=150, help='particles per class (random, seeded); 0 = all')
p.add_argument('-seed', type=int, default=0)
p.add_argument('-mode', default='static', choices=['static', 'movie', 'both'])
p.add_argument('-sub', type=int, default=6, help='movie frames per hour (linear interpolation)')
p.add_argument('-fps', type=int, default=12)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
out_dir = Ldir['LOo'] / 'DM_outs' / '20261006_pcmap_spaghetti_tidal'
Lfun.make_dir(out_dir)
QNAMES = ['inner-N', 'inner-S', 'outer-N', 'outer-S']
HALVES = [('surf', 'surface half', True), ('bot', 'bottom half', False)]

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

nfr = int(round(args.hours)) + 1                # hourly frames covering the cycle
d = xr.open_dataset(fn)
nfr = min(nfr, d.sizes['Time'])
plon = d.lon.values[:nfr]; plat = d.lat.values[:nfr]; cs0 = d.cs.values[0]
ot = pd.to_datetime(d.ot.values[:nfr])
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


cove = seg_mask(['pc_cp_m', 'pc_cp_p', 'pc_lp_m'])
inner = seg_mask(['pc_cp_m'])
jj, ii = np.where(cove)
north = np.zeros((NR, NC), dtype=bool)
for i in np.unique(ii):
    north[jj[(ii == i) & (jj > jj[ii == i].mean())], i] = True
QUAD = np.full((NR, NC), -1, dtype=int)
QUAD[cove] = (2 * (~inner) + (~north))[cove]

i0 = np.clip(np.round((plon[0] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
j0 = np.clip(np.round((plat[0] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
q0 = np.where(np.isfinite(plon[0]), QUAD[j0, i0], -1)
surf0 = cs0 >= -0.5

# ------------------------------------------------------------- the map ---
aa = [lon[cove].min() - 0.005, lon[cove].max() + 0.07, lat[cove].min() - 0.03, lat[cove].max() + 0.025]
# bathymetry cropped to the panel frame: every movie frame redraws all eight
# panels, and drawing the full 368 x 272 grid eight times per frame is most of
# the run time
jb = np.where((lat_ax >= aa[2] - 2 * dlat) & (lat_ax <= aa[3] + 2 * dlat))[0]
ib = np.where((lon_ax >= aa[0] - 2 * dlon) & (lon_ax <= aa[1] + 2 * dlon))[0]
bs = (slice(jb[0], jb[-1] + 1), slice(ib[0], ib[-1] + 1))
hm = np.ma.masked_where(mask[bs] == 0, h[bs])
hrs = (ot - ot[0]) / pd.Timedelta(hours=1)
norm = plt.Normalize(0, hrs[-1])
cmap = plt.get_cmap('viridis')
rng = np.random.default_rng(args.seed)

SEL = {}
for r, (hk, hlab, hv) in enumerate(HALVES):
    for c, qn in enumerate(QNAMES):
        idx = np.where((q0 == c) & (surf0 == hv))[0]
        n_all = len(idx)
        if args.n and n_all > args.n:
            idx = np.sort(rng.choice(idx, args.n, replace=False))
        SEL[(r, c)] = (idx, n_all, '%s, %s' % (qn, hlab))
phase = {'E': 'peak of the strongest ebb', 'F': 'peak of the strongest flood'}.get(
    rdir.name.split('_')[-2], 'release')


WINDOW = 'one tidal day' if args.hours > 20 else 'one tidal cycle'

# tide over the window, every 10 min, from the unfiltered tef2 hourly series
TIDE = None
hf_fn = Ldir['LOo'] / 'extract' / args.gtx / 'tef2' / 'hourly_flux_2024.01.01_2025.12.31_wb1_pc1.nc'
if hf_fn.is_file():
    hf = xr.open_dataset(hf_fn)
    th_ = pd.to_datetime(hf.time.values).values.astype('int64')
    ssh_s = hf.ssh.sel(sect='pc_lp').values
    q_s = hf.qnet.sel(sect='pc_lp').values
    hf.close()
    sgn = -np.sign(np.corrcoef(q_s, np.gradient(ssh_s))[0, 1])   # + = out of the cove (ebb)
    t_ts = np.linspace(0, hrs[-1], int(round(hrs[-1] * 6)) + 1)
    ta = (ot[0] + pd.to_timedelta(t_ts, unit='h')).values.astype('datetime64[ns]').astype('int64')
    TIDE = dict(t=t_ts, ssh=np.interp(ta, th_, ssh_s), qout=sgn * np.interp(ta, th_, q_s))
else:
    print('no %s -- tide panel skipped' % hf_fn.name)


def tide_value(t):
    return np.interp(t, TIDE['t'], TIDE['ssh']) if TIDE is not None else np.nan


def base_panels(fig_w=18, fig_h=9.0):
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.55] if TIDE is not None else [1, 1, 0.001],
                          hspace=0.35)
    axs = np.empty((2, 4), dtype=object)
    for r in range(2):
        for c in range(4):
            axs[r, c] = fig.add_subplot(gs[r, c], sharex=axs[0, 0] if (r or c) else None,
                                        sharey=axs[0, 0] if (r or c) else None)
    for (r, c), (idx, n_all, lab) in SEL.items():
        ax = axs[r, c]
        ax.pcolormesh(lon[bs], lat[bs], hm, cmap='Greys', vmin=0, vmax=120, shading='nearest', alpha=0.35)
        pfun.add_coast(ax, color='k', linewidth=0.6)
        ax.contour(lon[bs], lat[bs], cove[bs].astype(float), [0.5], colors='k', linewidths=1.0)
        pfun.dar(ax)
        ax.axis(aa)
        ax.locator_params(axis='x', nbins=4)
        ax.ticklabel_format(useOffset=False)
        ax.set_title('%s (%d of %d)' % (lab, len(idx), n_all), fontsize=10)
        if r == 1:
            ax.set_xlabel('Longitude')
        else:
            ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel('Latitude')
        else:
            ax.tick_params(labelleft=False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axs.ravel()), shrink=0.85, pad=0.01,
                 label='hours since release')
    axt = None
    if TIDE is not None:
        axt = fig.add_subplot(gs[2, :])
        pts = np.column_stack([TIDE['t'], TIDE['ssh']])
        axt.add_collection(LineCollection(np.stack([pts[:-1], pts[1:]], axis=1),
                                          colors=cmap(norm(TIDE['t'][:-1])), linewidths=2.5))
        axt.set_xlim(0, hrs[-1])
        pad = 0.1 * np.ptp(TIDE['ssh'])
        axt.set_ylim(TIDE['ssh'].min() - pad, TIDE['ssh'].max() + pad)
        axt.set_ylabel('ssh at pc_lp [m]\n(coloured by time)')
        axt.set_xlabel('hours since release')
        axt.grid(color='lightgray', linestyle='--', alpha=0.5)
        ax2 = axt.twinx()
        ax2.plot(TIDE['t'], TIDE['qout'], color='0.45', lw=1.0, ls='--')
        ax2.axhline(0, color='0.6', lw=0.6)
        lim = 1.1 * np.abs(TIDE['qout']).max()
        ax2.set_ylim(-lim, lim)
        ax2.set_ylabel('transport out of cove\n[m$^3$ s$^{-1}$], dashed (+ ebb)')
        # the maps keep their aspect (and lose width to the colorbar), so line
        # the tide panel up with the left edge of the first map column and the
        # right edge of the last
        for ax in axs.ravel():
            ax.apply_aspect()
        p0, p3, pt = axs[1, 0].get_position(), axs[1, 3].get_position(), axt.get_position()
        box = [p0.x0, pt.y0, p3.x1 - p0.x0, pt.height]
        axt.set_position(box)
        ax2.set_position(box)
    return fig, axs, axt


def path_segments(idx, x_all, y_all, t_all):
    segs, cols = [], []
    for k in idx:
        x, y = x_all[:, k], y_all[:, k]
        ok = np.isfinite(x) & np.isfinite(y)
        pts = np.column_stack([x[ok], y[ok]])
        if len(pts) < 2:
            continue
        segs.extend(np.stack([pts[:-1], pts[1:]], axis=1))
        cols.extend(cmap(norm(t_all[ok][:-1])))
    return segs, cols


# ------------------------------------------------------------- static ---
if args.mode in ['static', 'both']:
    fig, axs, axt = base_panels()
    for (r, c), (idx, n_all, lab) in SEL.items():
        ax = axs[r, c]
        segs, cols = path_segments(idx, plon, plat, hrs)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.7, alpha=0.8))
        ax.scatter(plon[0, idx], plat[0, idx], s=5, c='k', zorder=5, edgecolors='none')
        ax.scatter(plon[-1, idx], plat[-1, idx], s=12, marker='x', c='crimson', linewidths=0.7, zorder=6)
    fig.suptitle('%s: released %s UTC at the %s; paths over %.0f h (%s); dot = start, x = end'
                 % (name, ot[0].strftime('%Y-%m-%d %H:%M'), phase, hrs[-1], WINDOW), fontsize=12)
    fn_out = out_dir / ('%s_tidal.png' % name)
    fig.savefig(fn_out, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('wrote %s' % fn_out)

# -------------------------------------------------------------- movie ---
if args.mode in ['movie', 'both']:
    import matplotlib.animation as animation
    # interpolate the hourly track onto -sub frames per hour (display only)
    tf = np.linspace(0, hrs[-1], int(round(hrs[-1] * args.sub)) + 1)
    def interp_t(A):
        out = np.full((len(tf), A.shape[1]), np.nan)
        for k in range(A.shape[1]):
            ok = np.isfinite(A[:, k])
            if ok.sum() >= 2:
                out[:, k] = np.interp(tf, hrs[ok], A[ok, k], left=np.nan, right=np.nan)
        return out
    XI, YI = interp_t(plon), interp_t(plat)
    fig, axs, axt = base_panels()
    mark = axt.axvline(0, color='k', lw=1.5) if axt is not None else None
    arts = {}
    for (r, c), (idx, n_all, lab) in SEL.items():
        ax = axs[r, c]
        lc = LineCollection([], linewidths=0.7, alpha=0.8)
        ax.add_collection(lc)
        dots = ax.scatter(XI[0, idx], YI[0, idx], s=8, c='crimson', zorder=6, edgecolors='none')
        arts[(r, c)] = (lc, dots)
    ttl = fig.suptitle('', fontsize=12)

    def update(fi):
        for (r, c), (idx, n_all, lab) in SEL.items():
            lc, dots = arts[(r, c)]
            segs, cols = path_segments(idx, XI[:fi + 1], YI[:fi + 1], tf[:fi + 1])
            lc.set_segments(segs)
            lc.set_color(cols if cols else 'none')
            dots.set_offsets(np.column_stack([XI[fi, idx], YI[fi, idx]]))
        ttl.set_text('%s: released %s UTC at the %s;  t = %.1f h  (%s UTC)%s'
                     % (name, ot[0].strftime('%Y-%m-%d %H:%M'), phase, tf[fi],
                        (ot[0] + pd.Timedelta(hours=tf[fi])).strftime('%H:%M'),
                        '   ssh at pc_lp %+.2f m' % tide_value(tf[fi]) if TIDE is not None else ''))
        if mark is not None:
            mark.set_xdata([tf[fi], tf[fi]])
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(tf), interval=1000 / args.fps, blit=False)
    if animation.writers.is_available('ffmpeg'):
        fn_out = out_dir / ('%s_tidal.mp4' % name)
        anim.save(fn_out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=3000), dpi=120)
    else:
        fn_out = out_dir / ('%s_tidal.gif' % name)
        anim.save(fn_out, writer=animation.PillowWriter(fps=args.fps), dpi=100)
    plt.close(fig)
    print('wrote %s (%d frames)' % (fn_out, len(tf)))
