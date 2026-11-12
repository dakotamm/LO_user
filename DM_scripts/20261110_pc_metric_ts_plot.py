"""
Stacked time series of the Penn Cove tidal and subtidal metrics from
20261110_pc_metric_ts.py: one figure per band, sections as colours (+ = into the cove).

    tidal_<tag>.png     Q, Qprism, umax, cross-section share against Q, curtain share
                        against Q_mouth, surface u, surface-layer depth
    subtidal_<tag>.png  residual umax, cross-section seaward share, curtain seaward share,
                        residual surface u, residual surface-layer depth, back-cove
                        rotation speed, share of the column rotating counter-clockwise

<tag> = full (2024-2025) and zoom (-z0 .. -z1, default the two weeks around the
standard lunar day of 20261105_pc_standard_cycle.py).

run 20261110_pc_metric_ts_plot.py [-z0 2024.08.24 -z1 2024.09.07]
"""
import argparse

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-z0', default='2024.08.24')
p.add_argument('-z1', default='2024.09.07')
args = p.parse_args()

SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}
BCOL = {'z0-4': '#9ecae1', 'z4-10': '#4292c6', 'z10-bed': '#08306b'}   # shallow light -> deep dark
Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_metric_ts'
d = xr.open_dataset(out_dir / 'pc_metric_ts.nc')
T = TH = d.time.values
SECTS = list(d.sect.values)


W = {}                                     # current window, so y autoscales to what is shown


def inwin(t, y):
    m = (t >= W['x0']) & (t <= W['x1'])
    return t[m], y[m]


def by_sect(ax, k, scale=1, lw=0.6, dots=False):
    for s in SECTS:
        t, y = inwin(T, scale * d[k].sel(sect=s).values)
        if dots:
            ax.plot(t, y, '.', ms=1.2 if W['tag'] == 'full' else 3, color=SCOL[s], label=s)
        else:
            ax.plot(t, y, lw=lw, color=SCOL[s], label=s)


def single(ax, t, y, color='k', lw=0.6, label=None):
    ax.plot(*inwin(t, y), lw=lw, color=color, label=label)


def finish(fig, axs, labels, name, x0, x1):
    for ax, lab in zip(axs, labels):
        ax.set_ylabel(lab)
        ax.grid(True, color='0.85', lw=0.5)
        ax.set_xlim(x0, x1)
    axs[0].legend(loc='upper right', ncol=3, fontsize=8, frameon=False, markerscale=6)
    loc = mdates.AutoDateLocator()
    axs[-1].xaxis.set_major_locator(loc)
    axs[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    fig.savefig(out_dir / name, dpi=200, transparent=True)
    plt.close(fig)
    print('saved', out_dir / name)


WIN = {'full': (pd.Timestamp(T[0]), pd.Timestamp(T[-1])),
       'zoom': (pd.Timestamp(args.z0.replace('.', '-')), pd.Timestamp(args.z1.replace('.', '-')))}
for tag, (x0, x1) in WIN.items():
    W.update(tag=tag, x0=np.datetime64(x0), x1=np.datetime64(x1))
    lw = 0.3 if tag == 'full' else 1.0
    # ------------------------------------------------------------------ tidal
    fig, axs = plt.subplots(7, 1, figsize=(12, 14), sharex=True, layout='constrained')
    by_sect(axs[0], 'Q', lw=lw)
    axs[0].axhline(0, color='0.4', lw=0.5)
    by_sect(axs[1], 'Qprism', lw=1.0)
    by_sect(axs[2], 'umax', 100, lw=lw)
    by_sect(axs[3], 'fopp', 100, lw=lw)
    single(axs[4], TH, 100 * d.fopp_cur.values, lw=lw, label='curtain')
    by_sect(axs[5], 'Usurf', 100, lw=lw)
    axs[5].axhline(0, color='0.4', lw=0.5)
    by_sect(axs[6], 'dint', dots=True)
    axs[6].invert_yaxis()
    fig.suptitle('Penn Cove tidal metrics, hourly (+ into the cove)')
    finish(fig, axs, ['Q [m$^3$/s]', 'Q$_{prism}$ [m$^3$/s]', 'U$_{max}$ [cm/s]', 'cross-section\nagainst Q [%]',
                      'vertical section\nagainst Q$_{mouth}$ [%]', 'surface u [cm/s]', 'surface layer\ndepth [m]'],
           'tidal_%s.png' % tag, x0, x1)

    # ------------------------------------------------------------------ subtidal
    fig, axs = plt.subplots(7, 1, figsize=(12, 14), sharex=True, layout='constrained')
    by_sect(axs[0], 'umax_res', 100, lw=1.0)
    by_sect(axs[1], 'fsea_res', 100, lw=1.0)
    single(axs[2], TH, 100 * d.fsea_cur_res.values, lw=1.0)
    by_sect(axs[3], 'Usurf_res', 100, lw=1.0)
    axs[3].axhline(0, color='0.4', lw=0.5)
    by_sect(axs[4], 'dint_res', dots=True)
    axs[4].invert_yaxis()
    for b in d.band.values:
        single(axs[5], TH, 100 * d.U_rot_band.sel(band=b).values, color=BCOL[b], lw=0.6, label='%s (moving centre)' % b)
    single(axs[5], TH, 100 * d.rot_U.values, lw=1.0, label='fixed ring, depth mean')
    axs[5].axhline(0, color='0.4', lw=0.5)
    axs[5].legend(loc='upper right', ncol=4, fontsize=7, frameon=False)
    single(axs[6], TH, 100 * d.rot_frac.values, lw=1.0)
    for ax in axs[[1, 2, 6]]:
        ax.set_ylim(0, 100)
    fig.suptitle('Penn Cove subtidal metrics, Godin filtered (+ into the cove; rotation + counter-clockwise)')
    finish(fig, axs, ['residual U$_{max}$ [cm/s]', 'cross-section\nseaward [%]', 'vertical section\nseaward [%]',
                      'residual\nsurface u [cm/s]', 'residual surface\nlayer depth [m]', 'back-cove\nrotation [cm/s]',
                      'column rotating\nccw [%]'],
           'subtidal_%s.png' % tag, x0, x1)
d.close()
