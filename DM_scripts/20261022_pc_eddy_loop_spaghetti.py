"""
Spaghetti plots and a movie of particles that DO and DON'T loop around the
inner Penn Cove eddy, for one release with the eddy present and one with it
absent. Runs on apogee next to the pcmap tracks (raw hourly positions).

LOOPER = the same definition as 20261021_pc_eddy_winding.py: winding about the
FIXED eddy centre (present-hour mean of 20261021_pc_eddy_center.py, band -band)
W_fix >= 1 counterclockwise, counted hour by hour while the particle is in the
cove, before its first exit, and 200-1200 m from the centre (rmin/rmax from the
winding run); only particles with >= -hmin counted hours are eligible.
NON-LOOPER = eligible and W_fix < 1. The labels come from the per-release
winding pickle; the cumulative winding is recomputed here from the hourly
positions (and checked against the pickle) so the movie can count loops as they
complete.

Default releases (picked from winding_releases.csv, lower band, as typical of
their group: the release whose looper share is closest to the group median):
  F_2025.12.12  eddy present 100% of the first 3 d and of 14 d; 29% loop
                (median of the 61 releases with pres3d = 1: 29%)
  F_2025.10.02  eddy present 0% of the first 3 d and of 14 d; 13% loop
                (median of the 63 releases with pres3d = 0: 13%)
Both are peak-flood (F) releases, so the tidal phase at release matches.

  static  2 x 2: rows = releases, columns = loopers / non-loopers; -n of each,
          paths from release to first exit (or -hours), coloured by hours since
          release; + = fixed centre, dashed rings = rmin / rmax
  movie   the two releases side by side, -hours of hourly frames; loopers blue,
          non-loopers grey, fading -tail h tails; the moving centre (and r_eq
          ring) is drawn only in hours the eddy is present; counter of loops
          completed so far

Output: LO_output/DM_outs/20261022_pc_eddy_loop_spaghetti/

  python 20261022_pc_eddy_loop_spaghetti.py
  python 20261022_pc_eddy_loop_spaghetti.py -mode movie -hours 120
  python 20261022_pc_eddy_loop_spaghetti.py -rels pcmap_3d_sh4_E_2025.01.01,pcmap_3d_sh21_E_2025.01.20
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
from pcmap_regions import regions

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-rels', default='pcmap_3d_sh15_F_2025.12.12,pcmap_3d_sh18_F_2025.10.02',
               help='track dirs: eddy-present release first, eddy-absent second')
p.add_argument('-band', default='lower')
p.add_argument('-region', default='pc_cp')
p.add_argument('-mode', default='both', choices=['static', 'movie', 'both'])
p.add_argument('-n', type=int, default=25, help='particles per group per release')
p.add_argument('-hmin', type=float, default=24, help='min counted hours (eligible), as in the winding run')
p.add_argument('-hours', type=int, default=96, help='movie length, and static cap [h]')
p.add_argument('-tail', type=int, default=12, help='movie tail [h]')
p.add_argument('-fps', type=int, default=8)
p.add_argument('-seed', type=int, default=0)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
trk = Ldir['LOo'] / 'tracks2' / args.gtx
win_dir = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_winding' / args.region / args.band / args.gtx
cen_fn = Ldir['LOo'] / 'DM_outs' / '20261021_pc_eddy_center' / args.region / ('pc_eddy_center_%s.csv' % args.band)
out_dir = Ldir['LOo'] / 'DM_outs' / '20261022_pc_eddy_loop_spaghetti'
Lfun.make_dir(out_dir)
M_PER_DEG = 111.32e3
LAT0 = 48.23
LON_MOUTH = pd.read_pickle(Ldir['LOo'] / 'section_lines' / 'pc_lp.p').x.mean()
TWO_PI = 2 * np.pi
COS0 = np.cos(np.deg2rad(LAT0))
to_xy = lambda lo, la: ((lo - LON_MOUTH) * M_PER_DEG * COS0, (la - LAT0) * M_PER_DEG)
wrap = lambda a: (a + np.pi) % TWO_PI - np.pi

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h = g.lon_rho.values, g.lat_rho.values, g.h.values
mask = g.mask_rho.values.astype(bool)
g.close()
NR, NC = lon.shape
lon_ax, lat_ax = lon[0, :], lat[:, 0]
dlon, dlat = lon_ax[1] - lon_ax[0], lat_ax[1] - lat_ax[0]
REG = regions(Ldir, lon, lat)
cove, QUAD = REG['cove'], REG['QUAD']
sec = {k: pd.read_pickle(Ldir['LOo'] / 'section_lines' / (k + '.p')) for k in ['pc_cp', 'pc_lp']}

CEN = pd.read_csv(cen_fn, parse_dates=['time']).set_index('time')
CEN['present'] = CEN.present.astype(bool)
LONF, LATF = CEN.lon_c[CEN.present].mean(), CEN.lat_c[CEN.present].mean()
XF, YF = to_xy(LONF, LATF)


def ji_of(plon, plat):
    ok = np.isfinite(plon) & np.isfinite(plat)
    i = np.zeros(plon.shape, dtype=int)
    j = np.zeros(plon.shape, dtype=int)
    i[ok] = np.clip(np.round((plon[ok] - lon_ax[0]) / dlon), 0, NC - 1).astype(int)
    j[ok] = np.clip(np.round((plat[ok] - lat_ax[0]) / dlat), 0, NR - 1).astype(int)
    return j, i, ok


def load(rdir):
    """Track (same particle selection and order as the winding run) + labels + cumulative winding."""
    fn = sorted((trk / rdir).glob('release_*.nc'))[0]
    R = pickle.load(open(win_dir / ('%s__%s.p' % (rdir, fn.stem)), 'rb'))
    rmin, rmax = R['meta']['rmin'], R['meta']['rmax']
    nf = int(round(14 * 24)) + 1
    d = xr.open_dataset(fn)
    plon, plat = d.lon.values[:nf], d.lat.values[:nf]
    ot = pd.to_datetime(d.ot.values[:nf])
    d.close()
    j, i, ok = ji_of(plon, plat)
    keep = ok[0] & cove[j[0], i[0]]
    plon, plat, j, i, ok = plon[:, keep], plat[:, keep], j[:, keep], i[:, keep], ok[:, keep]
    P = R['P']
    assert len(P) == plon.shape[1], 'particle count differs from the winding pickle'
    inside = np.where(ok, QUAD[j, i], -1) >= 0
    out_k = np.argmax(~inside[1:], axis=0) + 1
    out_k[inside[1:].all(axis=0)] = nf
    valid = np.arange(nf)[:, None] < out_k[None, :]
    X, Y = to_xy(plon, plat)
    dx, dy = X - XF, Y - YF
    r = np.hypot(dx, dy)
    okr = valid & (r >= rmin) & (r <= rmax)
    use = okr[1:] & okr[:-1]
    W = np.vstack([np.zeros((1, X.shape[1])),
                   np.cumsum(np.where(use, wrap(np.diff(np.arctan2(dy, dx), axis=0)), 0.0), axis=0) / TWO_PI])
    assert np.allclose(W[-1], P.W_fix.values, atol=1e-4), 'recomputed W_fix differs from the winding pickle'
    elig = P.h_fix.values >= args.hmin
    loop = elig & (P.W_fix.values >= 1)
    nonl = elig & (P.W_fix.values < 1)
    rng = np.random.default_rng(args.seed)
    pick = lambda m: np.sort(rng.choice(np.flatnonzero(m), min(args.n, m.sum()), replace=False))
    lon_s, lat_s = plon.copy(), plat.copy()
    lon_s[~valid], lat_s[~valid] = np.nan, np.nan                     # stop at first exit, as the metric does
    c = CEN.reindex(ot, method='nearest', tolerance=pd.Timedelta('31min'))
    t0 = ot[0]
    w3 = (CEN.index >= t0) & (CEN.index < t0 + pd.Timedelta('3D'))
    return dict(name=rdir, t0=t0, ot=ot, lon=lon_s, lat=lat_s, W=W, loop=pick(loop), nonl=pick(nonl),
                share=loop.sum() / max(elig.sum(), 1), n_elig=int(elig.sum()), rmin=rmin, rmax=rmax,
                pres3d=CEN.present[w3].mean(), pres14=R['meta']['pres14'],
                c_lon=np.where(c.present.fillna(False), c.lon_c, np.nan),
                c_lat=np.where(c.present.fillna(False), c.lat_c, np.nan), c_req=c.r_eq.values)


REL = [load(r) for r in args.rels.split(',')]
for D in REL:
    print('%s: eddy present %.0f%% of first 3 d, %.0f%% of 14 d; loopers %.1f%% of %d eligible'
          % (D['name'], 100 * D['pres3d'], 100 * D['pres14'], 100 * D['share'], D['n_elig']))
lab = lambda D: '%s  (eddy present %.0f%% of first 3 d)\n%.0f%% of %d eligible particles loop' \
    % (D['t0'].strftime('%Y-%m-%d %H:%M'), 100 * D['pres3d'], 100 * D['share'], D['n_elig'])
AA = [-122.737, -122.655, 48.212, 48.246]


def circle(ax, lo, la, r, **kw):
    a = np.linspace(0, TWO_PI, 121)
    return ax.plot(lo + r * np.cos(a) / (M_PER_DEG * COS0), la + r * np.sin(a) / M_PER_DEG, **kw)[0]


def base(ax, D):
    ax.pcolormesh(lon, lat, np.where(mask, h, np.nan), cmap='Blues', vmin=0, vmax=60, alpha=0.35,
                  shading='nearest', zorder=0)
    pfun.add_coast(ax, color='gray', linewidth=0.6)
    for k in sec:
        ax.plot(sec[k].x, sec[k].y, color='0.35', lw=0.8)
    ax.plot(LONF, LATF, '+', color='k', ms=12, mew=2, zorder=12)
    for r in [D['rmin'], D['rmax']]:
        circle(ax, LONF, LATF, r, color='k', ls='--', lw=0.7, zorder=11)
    ax.axis(AA)
    pfun.dar(ax)
    ax.tick_params(labelsize=8)


# ------------------------------------------------------------ static ---
if args.mode in ['static', 'both']:
    fig, axs = plt.subplots(len(REL), 2, figsize=(14, 4.6 * len(REL)), layout='constrained', squeeze=False)
    norm = plt.Normalize(0, args.hours)
    for r_, D in enumerate(REL):
        for c_, (key, gl) in enumerate([('loop', 'loopers (W >= 1 CCW)'), ('nonl', 'non-loopers (W < 1)')]):
            ax = axs[r_, c_]
            base(ax, D)
            segs, arr = [], []
            for k in D[key]:
                xy = np.column_stack([D['lon'][:args.hours + 1, k], D['lat'][:args.hours + 1, k]])
                s2 = np.stack([xy[:-1], xy[1:]], axis=1)
                good = np.isfinite(s2).all(axis=(1, 2))
                segs.extend(s2[good]); arr.extend(np.arange(len(s2))[good])
            lc = LineCollection(segs, array=np.array(arr), cmap='viridis', norm=norm, linewidths=0.8, zorder=8)
            ax.add_collection(lc)
            ax.scatter(D['lon'][0, D[key]], D['lat'][0, D[key]], s=10, c='k', zorder=9, ec='none')
            ax.set_title('%s: %d %s, first %d h\n%s' % (D['name'].split('_', 3)[-1], len(D[key]), gl, args.hours,
                                                        lab(D)), fontsize=9)
    fig.colorbar(lc, ax=axs, label='hours since release', shrink=0.6)
    fig.suptitle('Particles that do and do not loop the inner-cove eddy (%s band centre +, dashed = %g / %g m)'
                 % (args.band, REL[0]['rmin'], REL[0]['rmax']))
    fn = out_dir / 'pc_eddy_loop_spaghetti.png'
    fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print('saved ' + str(fn))

# ------------------------------------------------------------- movie ---
if args.mode in ['movie', 'both']:
    fig, axs = plt.subplots(1, len(REL), figsize=(7.5 * len(REL), 5.6), layout='constrained', squeeze=False)
    axs = axs[0]
    ART = []
    for ax, D in zip(axs, REL):
        base(ax, D)
        tails = LineCollection([], linewidths=0.9, zorder=8)
        ax.add_collection(tails)
        idx = np.r_[D['loop'], D['nonl']]
        col = np.array([(0.12, 0.47, 0.71, 1.0)] * len(D['loop']) + [(0.45, 0.45, 0.45, 1.0)] * len(D['nonl']))
        dots = ax.scatter(D['lon'][0, idx], D['lat'][0, idx], s=14, c=col, zorder=10, ec='none')
        cen, = ax.plot([], [], 'o', color='tab:red', ms=7, mec='k', zorder=13)
        ring, = ax.plot([], [], color='tab:red', lw=1.2, zorder=12)
        ttl = ax.set_title('', fontsize=9)
        ART.append(dict(D=D, idx=idx, col=col, tails=tails, dots=dots, cen=cen, ring=ring, ttl=ttl))
    axs[0].plot([], [], 'o', color=(0.12, 0.47, 0.71), label='looper (completes >= 1 CCW loop by 14 d)')
    axs[0].plot([], [], 'o', color='0.45', label='non-looper')
    axs[0].plot([], [], 'o', color='tab:red', mec='k', label='moving eddy centre + r_eq (only when present)')
    axs[0].legend(loc='lower right', fontsize=7)
    ang = np.linspace(0, TWO_PI, 61)

    def update(fi):
        for A in ART:
            D, idx = A['D'], A['idx']
            k0 = max(0, fi - args.tail)
            segs, cc = [], []
            for m, k in enumerate(idx):
                xy = np.column_stack([D['lon'][k0:fi + 1, k], D['lat'][k0:fi + 1, k]])
                s2 = np.stack([xy[:-1], xy[1:]], axis=1) if len(xy) > 1 else np.zeros((0, 2, 2))
                good = np.isfinite(s2).all(axis=(1, 2))
                c = np.repeat(A['col'][m][None], len(s2), axis=0)
                c[:, 3] = np.linspace(0.05, 0.8, len(s2)) if len(s2) else c[:, 3]
                segs.extend(s2[good]); cc.extend(c[good])
            A['tails'].set_segments(segs)
            A['tails'].set_color(cc if cc else 'none')
            A['dots'].set_offsets(np.column_stack([D['lon'][fi, idx], D['lat'][fi, idx]]))
            lo, la, re = D['c_lon'][fi], D['c_lat'][fi], D['c_req'][fi]
            if np.isfinite(lo):
                A['cen'].set_data([lo], [la])
                A['ring'].set_data(lo + re * np.cos(ang) / (M_PER_DEG * COS0), la + re * np.sin(ang) / M_PER_DEG)
            else:
                A['cen'].set_data([], []); A['ring'].set_data([], [])
            nl = int((D['W'][fi, D['loop']] >= 1).sum())
            nin = int(np.isfinite(D['lon'][fi, idx]).sum())
            A['ttl'].set_text('%s   hour %d (%s)   eddy %s\nloopers that have completed a loop so far: %d of %d;'
                              '  %d of %d drawn still in the cove'
                              % (D['name'].split('_', 3)[-1], fi, D['ot'][fi].strftime('%m-%d %H:%M'),
                                 'PRESENT' if np.isfinite(lo) else 'absent', nl, len(D['loop']), nin, len(idx)))
        return []

    anim = animation.FuncAnimation(fig, update, frames=np.arange(args.hours + 1), interval=1000 / args.fps)
    if animation.writers.is_available('ffmpeg'):
        fn = out_dir / 'pc_eddy_loop_movie.mp4'
        anim.save(fn, writer=animation.FFMpegWriter(fps=args.fps, bitrate=2400))
    else:
        fn = out_dir / 'pc_eddy_loop_movie.gif'
        anim.save(fn, writer=animation.PillowWriter(fps=args.fps))
    plt.close(fig)
    print('saved %s (%d frames)' % (fn, args.hours + 1))
