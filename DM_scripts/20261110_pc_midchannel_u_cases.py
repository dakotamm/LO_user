"""
The views of 20261110_pc_midchannel_u_timeseries.py and 20261110_pc_midchannel_u_peaks.py split
by one forcing axis at a time (spring / neap, tropic / equatorial, high / low strat, up- / down-cove
wind). Two views each, rows pc_cp | pc_lj | pc_lp (west to east, top to bottom), blue = into the
cove, red = out (cmocean balance_r):
    profile  y = z, colour = cross-channel averaged u, U_b = Q_b / A_b in 1 m bins below the moving
             surface (bins wetted less than half a face blank), at z = zs - depth
    across   y = latitude of each u face (north at the top), colour = that face's depth-averaged u

1. ONE LUNAR DAY per case, columns baseline | high end | low end of the axis, x = hours since the
   start of each day (the first flood). Days as 20261106_pc_case_compare_animation.py: baseline =
   top of 20261105_pc_standard_cycle/lunar_days.csv; each variant = the best row of variants.csv
   whose tides come in the baseline's order (little flood -> big ebb -> big flood -> little ebb),
   with the same hand picks (spring 2024-09-02 07:00, up-cove wind 2024-01-19 13:00). Hourly
   ocean_his snapshots from the pc_cove box. Top strip: mouth Ubar = Q / A and sea level.
2. EVERY TIDE in each class, columns high end | low end, one column per tide at its peak hour
   (from 20261110_pc_midchannel_u_peaks.py, peaks.npz), x = the tide's number within the class
   (contiguous, no gaps; the date of every n-th tide on the axis); column widths in proportion to
   the number of tides. A tide is in a class when its peak hour is (pc_tidal_cycles.cycle_masks:
   spring / neap and tropic / equatorial = top / bottom third of the D2 / D1 envelope ranked within
   each month; strat and wind from pc_forcing_states). Top strip: each section's Ubar at the peak.

One colour scale per figure (99th percentile of |u| over all its columns). Day figures also carry
direction arrows (every ASTEP h; profile: every ZSTEP-th s level, across: every face): pointing LEFT
= into the cove (west), RIGHT = out, length in proportion to |u| (full length at the colour limit);
none under AMIN of the colour limit.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
       LO_output/DM_outs/20261110_pc_midchannel_u_peaks/peaks.npz (run that script first)
       LO_output/DM_outs/20261110_pc_column_metrics/column_metrics.nc (time axis)
OUTPUT LO_output/DM_outs/20261110_pc_midchannel_u_cases/
    day_[view]_[axis].png             (1)
    peaks_[view]_[kind]_[axis].png    (2), kind = flood | ebb

run 20261110_pc_midchannel_u_cases.py [-axis springneap|tropeq|strat|wind|all]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
import cmocean

from lo_tools import Lfun
from pc_tidal_cycles import cycle_masks

AXES = {'springneap': ('spring', 'neap'), 'tropeq': ('tropic', 'equatorial'),
        'strat': ('high strat', 'low strat'), 'wind': ('up-cove wind', 'down-cove wind')}
CLASS = {'up-cove wind': 'up-cove', 'down-cove wind': 'down-cove'}   # variants.csv name -> cycle_masks name
parser = argparse.ArgumentParser()
parser.add_argument('-axis', default='all', choices=['all'] + list(AXES))
args = parser.parse_args()

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                 # section-positive (east) is out of the cove
FACES = {                                 # u faces in the pc_cove box, north to south; west to east
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
SCOL = {'pc_cp': '#009E73', 'pc_lj': '#CC79A7', 'pc_lp': '#E69F00'}
COL_IN, COL_OUT = '#2166AC', '#B2182B'   # blue = entering the cove, red = leaving
CMAP = cmocean.cm.balance_r
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
NDATE = 6                                 # date labels per peaks column
ASTEP, ZSTEP = 2, 3                       # day figures: an arrow every 2 h, on every 3rd s level (profile) or every face (across)
AMIN = 0.1                                # no arrow where |u| < 10% of the colour limit

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
pk_fn = Ldir['LOo'] / 'DM_outs' / '20261110_pc_midchannel_u_peaks' / 'peaks.npz'
cm_fn = Ldir['LOo'] / 'DM_outs' / '20261110_pc_column_metrics' / 'column_metrics.nc'
sel_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_standard_cycle'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_midchannel_u_cases'
out_dir.mkdir(parents=True, exist_ok=True)


def edges(x):
    """Cell edges from centres: midpoints, ends half a spacing out."""
    return np.concatenate([[x[0] - (x[1] - x[0]) / 2], 0.5 * (x[1:] + x[:-1]), [x[-1] + (x[-1] - x[-2]) / 2]])


def binned(u, zw, dy):
    """U_b = Q_b / A_b in the EDGES bins below the moving surface, returned bottom to top as (u (t, b),
    z edges (t, b + 1)) on z = zs - depth; NaN where a bin is wetted less than half a face."""
    dz = np.diff(zw, axis=1)
    q = u * dz * dy
    dw = zw[:, -1:, :] - zw                                          # depth below the surface of each interface
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    assert d_bot.max() < EDGES[-1]
    nb = len(EDGES) - 1
    Qb, Ab = np.zeros((len(u), nb)), np.zeros((len(u), nb))
    for b in range(nb):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)
        Qb[:, b] = (q * ov / dz).sum(axis=(1, 2))
        Ab[:, b] = (ov * dy).sum(axis=(1, 2))
    assert np.allclose(Qb.sum(axis=1), q.sum(axis=(1, 2)))
    Ub = np.where(Ab >= 0.5 * dy.mean(), Qb / np.maximum(Ab, 1e-12), np.nan)
    zs = (zw[:, -1, :] * dy).sum(axis=-1) / dy.sum()
    return Ub[:, ::-1], zs[:, None] - EDGES[None, ::-1]


def colorbar(fig, axs, norm, label):
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=CMAP, norm=norm), ax=axs, location='right', shrink=0.8, pad=0.01, extend='both')
    cb.set_label(label + ' [cm s$^{-1}$]\n+ into the cove (west)')
    cb.ax.text(0.5, 1.02, 'IN', transform=cb.ax.transAxes, ha='center', va='bottom', fontweight='bold', color=COL_IN)
    cb.ax.text(0.5, -0.02, 'OUT', transform=cb.ax.transAxes, ha='center', va='top', fontweight='bold', color=COL_OUT)


def arrows(ax, x, Y, U, umax, xlen):
    """Direction arrows on a section panel: x (n,) centres, Y (n, m) positions, U (n, m) cm/s, + into the
    cove drawn pointing LEFT (west), length |U| / umax * xlen in x units; none where |U| < AMIN of umax."""
    X = np.broadcast_to(x[:, None], Y.shape)
    m = np.abs(U) >= AMIN * umax
    ax.quiver(X[m], Y[m], -np.clip(U[m], -umax, umax) / umax * xlen, np.zeros(m.sum()), angles='xy', scale_units='xy', scale=1,
              pivot='middle', color='k', width=0.003, headwidth=4, headlength=4, headaxislength=3.5, zorder=3)


def draw(ax, view, xe, r, norm, ylim, x=None):
    """One section panel: x edges xe (n + 1), r with u (n, k), zw (n, w), uface (n, p), late (p + 1);
    with x (cell centres) also direction arrows every ASTEP columns."""
    if view == 'profile':
        Y = np.concatenate([r['zw'][:1], 0.5 * (r['zw'][1:] + r['zw'][:-1]), r['zw'][-1:]], axis=0)   # (n + 1, w)
        X = np.broadcast_to(xe[:, None], Y.shape)
        ax.pcolormesh(X.T, Y.T, np.ma.masked_invalid(100 * r['u']).T, cmap=CMAP, norm=norm, shading='flat', edgecolors='face', linewidth=0, rasterized=True)
    else:
        ax.pcolormesh(xe, r['late'], 100 * r['uface'].T, cmap=CMAP, norm=norm, shading='flat', edgecolors='face', linewidth=0, rasterized=True)
    if x is not None:
        sub, dx = slice(None, None, ASTEP), ASTEP * np.median(np.diff(x))
        if view == 'profile':
            zr = 0.5 * (r['zw'][:, 1:] + r['zw'][:, :-1])
            arrows(ax, x[sub], zr[sub, ::ZSTEP], 100 * r['u'][sub, ::ZSTEP], norm.vmax, 0.9 * dx)
        else:
            arrows(ax, x[sub], np.broadcast_to(0.5 * (r['late'][1:] + r['late'][:-1]), r['uface'][sub].shape),
                   100 * r['uface'][sub], norm.vmax, 0.9 * dx)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)


VIEW = {'profile': dict(key='u', ylab='z [m]', clab='cross-channel averaged u normal to the section'),
        'across': dict(key='uface', ylab='latitude [deg N]\nS ↓   N ↑', clab='depth-averaged u normal to the section')}


def ylims(Rs):
    return {'profile': (min(np.nanmin(np.where(np.isfinite(r['u']), r['zw'][:, :-1], np.nan)) for r in Rs) - 0.5,   # deepest wet bin
                        max(r['zw'][:, -1].max() for r in Rs) + 0.5),
            'across': (min(r['late'].min() for r in Rs), max(r['late'].max() for r in Rs))}


# ------------------------------------------------------------------ 1. the lunar days, baseline | high | low
order = lambda r: (np.sign(r.Qflood2 - r.Qflood1), np.sign(r.Qebb2 - r.Qebb1))
PICK = {'spring': '2024-09-02 07:00', 'up-cove wind': '2024-01-19 13:00'}   # hand picks of 20261106_pc_case_compare_animation
BASE = pd.read_csv(sel_dir / 'lunar_days.csv', parse_dates=['t0', 't1']).iloc[0]
V = pd.read_csv(sel_dir / 'variants.csv', parse_dates=['t0', 't1'])


def pick_day(c):
    g = V[V.variant == c]
    if c in PICK:
        return g[g.t0 == pd.Timestamp(PICK[c])].iloc[0]
    same = [order(r) == order(BASE) for _, r in g.iterrows()]
    return g[same].iloc[0]


def load_day(ds, TA, d):
    it = np.flatnonzero((TA >= d.t0) & (TA <= d.t1))
    sl = dict(ocean_time=slice(it[0], it[-1] + 1))
    R = {}
    for SN, F in FACES.items():
        XI_U, ETA = F['XI_U'], F['ETA']
        u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA, **sl).to_numpy().astype(float)                  # (t, k, p) + into the cove
        zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().mean(axis=-1)     # (t, w, p) at the face
        zeta = ds.zeta.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().mean(axis=(1, 2))
        dy = (1 / ds.pn.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy()).mean(axis=-1)
        dz = np.diff(zw, axis=1)
        ub, ze = binned(u, zw, dy)
        R[SN] = dict(u=ub, zw=ze, zeta=zeta,
                     UBAR=(u * dz * dy).sum(axis=(1, 2)) / (dz * dy).sum(axis=(1, 2)),
                     uface=(u * dz).sum(axis=1) / dz.sum(axis=1),
                     late=edges(ds.lat_u.isel(eta_u=ETA, xi_u=XI_U).to_numpy()))
    return R, (TA[it] - d.t0) / pd.Timedelta('1h')


def day_figures(axis, ds, TA):
    cases = ['baseline'] + list(AXES[axis])
    DAY = {'baseline': BASE}
    DAY.update({c: pick_day(c) for c in cases[1:]})
    RR, HH = {}, {}
    for c in cases:
        RR[c], HH[c] = load_day(ds, TA, DAY[c])
        print('  %-15s %s to %s UTC; ranks D2 %.2f D1 %.2f strat %.2f' % (c, DAY[c].t0, DAY[c].t1, DAY[c].r_D2, DAY[c].r_D1, DAY[c].r_strat))
    YL = ylims([RR[c][SN] for c in cases for SN in FACES])
    hmax = max(HH[c][-1] for c in cases) + 0.5
    for view, vw in VIEW.items():
        umax = np.ceil(100 * np.nanpercentile(np.abs(np.concatenate([RR[c][SN][vw['key']].ravel() for c in cases for SN in FACES])), 99))
        norm = matplotlib.colors.Normalize(-umax, umax)
        fig = plt.figure(figsize=(20, 10), layout='constrained')
        gs = fig.add_gridspec(4, 3, height_ratios=[1, 2, 2, 2])
        axs = []
        for j, c in enumerate(cases):
            R, h = RR[c], HH[c].to_numpy()
            ax_t = fig.add_subplot(gs[0, j])
            ub = 100 * R['pc_lp']['UBAR']
            ax_t.plot(h, ub, color='k', lw=1.5, label='Ubar pc_lp (mouth)')
            ax_t.fill_between(h, 0, ub, where=ub > 0, color=COL_IN, alpha=0.3, lw=0, interpolate=True)
            ax_t.fill_between(h, 0, ub, where=ub < 0, color=COL_OUT, alpha=0.3, lw=0, interpolate=True)
            ax_t.axhline(0, color='k', lw=0.5)
            ax_z = ax_t.twinx()
            ax_z.plot(h, R['pc_lp']['zeta'], color='0.4', lw=1, ls='--', label='sea level pc_lp')
            ax_z.tick_params(axis='y', colors='0.4')
            ax_t.set_xlim(-0.5, hmax)
            ax_t.grid(True, alpha=0.3)
            ax_t.tick_params(labelbottom=False)
            ax_t.set_title('%s: %s to %s UTC\nranks D2 %.2f, D1 %.2f, strat %.2f'
                           % (c.upper(), DAY[c].t0.strftime('%Y-%m-%d %H:%M'), DAY[c].t1.strftime('%m-%d %H:%M'),
                              DAY[c].r_D2, DAY[c].r_D1, DAY[c].r_strat), fontsize=10)
            if j == 0:
                ax_t.set_ylabel('Ubar = Q / A\n[cm s$^{-1}$], + in')
                h1, l1 = ax_t.get_legend_handles_labels()
                h2, l2 = ax_z.get_legend_handles_labels()
                ax_t.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left', ncol=2)
            if j == 2:
                ax_z.set_ylabel('sea level [m]', color='0.4')
            for i, SN in enumerate(FACES):
                ax = fig.add_subplot(gs[i + 1, j], sharex=ax_t)
                draw(ax, view, edges(h), R[SN], norm, YL[view], x=h)
                ax.set_title(NAME[SN], fontsize=9, loc='left')
                if j == 0:
                    ax.set_ylabel(vw['ylab'])
                else:
                    ax.tick_params(labelleft=False)
                if i < 2:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel('hours since the start of the lunar day (first flood)')
                axs.append(ax)
        colorbar(fig, axs, norm, vw['clab'])
        fig.suptitle('%s: one lunar day per case (baseline | %s | %s), hourly ocean_his, %s\n'
                     'blue = into the cove, red = out; arrows: \u2190 IN (west, into the cove)   OUT \u2192, length \u221d |u|'
                     % (axis, *AXES[axis], GTAGEX), fontsize=11)
        fn = out_dir / ('day_%s_%s.png' % (view, axis))
        fig.savefig(fn, dpi=200, transparent=True)
        plt.close(fig)
        print('  saved %s' % fn)


# ------------------------------------------------------------------ 2. every tide in each class, high | low
P = dict(np.load(pk_fn))
cm = xr.open_dataset(cm_fn)
T = pd.DatetimeIndex(cm.time.values)
cm.close()
CM = cycle_masks(T.values)
LATE = {SN: edges(P['lat_' + SN]) for SN in FACES}


def peak_figures(axis):
    classes = [CLASS.get(c, c) for c in AXES[axis]]
    for kind in ['flood', 'ebb']:
        idx = P['idx_' + kind]
        sel = {c: np.flatnonzero(CM[c][idx]) for c in classes}
        RR = {c: {SN: dict(u=P['u_%s_%s' % (kind, SN)][s], zw=P['zw_%s_%s' % (kind, SN)][s],
                           uface=P['uface_%s_%s' % (kind, SN)][s], ubar=P['ubar_%s_%s' % (kind, SN)][s], late=LATE[SN])
                  for SN in FACES} for c, s in sel.items()}
        print('  %s: %s' % (kind, ', '.join('%d %s' % (len(s), c) for c, s in sel.items())))
        YL = ylims([RR[c][SN] for c in classes for SN in FACES])
        for view, vw in VIEW.items():
            umax = np.ceil(100 * np.nanpercentile(np.abs(np.concatenate([RR[c][SN][vw['key']].ravel() for c in classes for SN in FACES])), 99))
            norm = matplotlib.colors.Normalize(-umax, umax)
            fig = plt.figure(figsize=(18, 10), layout='constrained')
            gs = fig.add_gridspec(4, 2, height_ratios=[1, 2, 2, 2], width_ratios=[len(sel[c]) for c in classes])
            axs = []
            ulo = min(100 * RR[c][SN]['ubar'].min() for c in classes for SN in FACES)
            uhi = max(100 * RR[c][SN]['ubar'].max() for c in classes for SN in FACES)
            for j, c in enumerate(classes):
                n = len(sel[c])
                x = np.arange(n)
                xe = np.arange(n + 1) - 0.5
                ax_t = fig.add_subplot(gs[0, j])
                for SN in FACES:
                    ax_t.plot(x, 100 * RR[c][SN]['ubar'], color=SCOL[SN], lw=0.6, marker='.', ms=2, label='Ubar ' + SN)
                ax_t.axhline(0, color='k', lw=0.5)
                ax_t.set_ylim(ulo - 0.5, uhi + 0.5)
                ax_t.set_xlim(xe[0], xe[-1])
                ax_t.grid(True, alpha=0.3)
                ax_t.tick_params(labelbottom=False)
                ax_t.set_title('%s: %d %ss' % (c.upper(), n, kind), fontsize=10)
                if j == 0:
                    ax_t.set_ylabel('Ubar = Q / A at the\npeak hour [cm s$^{-1}$]\n+ in')
                    ax_t.legend(fontsize=7, loc='upper left', ncol=3)
                else:
                    ax_t.tick_params(labelleft=False)
                tick = np.linspace(0, n - 1, NDATE).round().astype(int)
                for i, SN in enumerate(FACES):
                    ax = fig.add_subplot(gs[i + 1, j], sharex=ax_t)
                    draw(ax, view, xe, RR[c][SN], norm, YL[view])
                    ax.set_title(NAME[SN], fontsize=9, loc='left')
                    if j == 0:
                        ax.set_ylabel(vw['ylab'])
                    else:
                        ax.tick_params(labelleft=False)
                    if i < 2:
                        ax.tick_params(labelbottom=False)
                    else:
                        ax.set_xticks(tick)
                        ax.set_xticklabels([T[idx[sel[c][k]]].strftime('%Y-%m-%d') for k in tick], fontsize=8)
                        ax.set_xlabel('%s number within the class (date of the peak)' % kind)
                    axs.append(ax)
            colorbar(fig, axs, norm, vw['clab'])
            fig.suptitle('%s: every %s at its peak hour, %s | %s (%s), %s; blue = into the cove, red = out'
                         % (axis, kind.upper(), *classes, 'tides placed side by side, not on a time axis', GTAGEX), fontsize=11)
            fn = out_dir / ('peaks_%s_%s_%s.png' % (view, kind, axis))
            fig.savefig(fn, dpi=200, transparent=True)
            plt.close(fig)
            print('  saved %s' % fn)


ds = xr.open_dataset(in_fn)
TA = pd.DatetimeIndex(ds.ocean_time.values)
for axis in (list(AXES) if args.axis == 'all' else [args.axis]):
    print(axis)
    day_figures(axis, ds, TA)
    peak_figures(axis)
ds.close()
