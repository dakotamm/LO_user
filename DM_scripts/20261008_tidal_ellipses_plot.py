"""
Plot the harmonic fits from 20261008_tidal_ellipses.py.

  python 20261008_tidal_ellipses_plot.py -job pc_cove -0 2024.01.01 -1 2025.12.31 -dom box
  python 20261008_tidal_ellipses_plot.py -job sp_head_tide -0 2024.01.01 -1 2025.12.31 -dom sp_head
  python 20261008_tidal_ellipses_plot.py -job sp_head_tide -0 2024.01.01 -1 2025.12.31 -dom sp_head -fld bot

-dom  box = the extent of the file itself; otherwise any closed polygon in
      LO_output/section_lines (pc, skagit_delta, wb, ...), or 'full' (= wb).
      Window = polygon bbox + 10 cells; only cells inside wb.p are drawn
      ([[wb1-region-plot-window]]).
-fld  bar (depth-averaged), surf / bot (top / bottom layer: the -surf/-bot
      fits for sp_head_tide), or top / bot from the 3D fit for pc_cove.

Figures (to LO_output/DM_outs/20261008_tidal_ellipses/):
  tidal_ellipses_<job>_<dom>_<fld>_<CON>.png  major axis + glyphs, signed
      eccentricity (Lsmin/Lsmaj, + = counterclockwise), inclination, phase
  tidal_cotidal_<job>_<dom>_<CON>.png         zeta amplitude and phase
  tidal_ellipses_<job>_<dom>_<fld>_allcons.png overview: major axis + glyphs,
      one panel per constituent (own color scale each)
  tidal_cotidal_<job>_<dom>_allcons.png       overview: zeta amplitude, one
      panel per constituent
-cons  'all' (default, every constituent in the fit) or e.g. M2,K1
  tidal_residual_<job>_<dom>.png              Eulerian residual (time-mean
      velocity): depth-averaged, top and bottom layers, one shared scale
  tidal_ellipses_<job>_vertical_<CON>.png     (3D fits only) surface vs bottom
      glyphs and the ellipse along the mouth (last rho column west of pc_lp)
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
import cmocean

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-job', default='pc_cove', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-dom', default='box', type=str)
parser.add_argument('-fld', default='bar', type=str)
parser.add_argument('-cons', default='all', type=str)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
fn = Ldir['LOo'] / 'extract' / args.gtagex / 'tidal_ellipses' / (args.job + '_' + args.ds0 + '_' + args.ds1 + '.nc')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261008_tidal_ellipses'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
PAD_CELLS = 10
NGLYPH = 16 # target glyphs across the window

ds = xr.open_dataset(fn)
CONS = [str(c) for c in ds.con.values] if args.cons == 'all' else args.cons.split(',')
PERIOD = {'M2': 12.42, 'S2': 12.00, 'N2': 12.66, 'K2': 11.97, 'K1': 23.93, 'O1': 25.82,
    'P1': 24.07, 'Q1': 26.87, 'M4': 6.21, 'MS4': 6.10, 'M6': 4.14} # hours
lon = ds.lon_rho.values
lat = ds.lat_rho.values
dx = np.nanmedian(np.diff(lon[0]))
dy = np.nanmedian(np.diff(lat[:, 0]))
plon, plat = pfun.get_plon_plat(lon, lat)

def poly_mask(df):
    pth = MplPath(np.column_stack([df.x.values, df.y.values]))
    return pth.contains_points(np.column_stack([lon.ravel(), lat.ravel()])).reshape(lon.shape)

wb = pd.read_pickle(sect_dir / 'wb.p')
show = (ds.mask_rho.values == 1) & poly_mask(wb)
if args.dom == 'box':
    aa = [plon.min(), plon.max(), plat.min(), plat.max()]
else:
    df = wb if args.dom in ['full', 'wb'] else pd.read_pickle(sect_dir / (args.dom + '.p'))
    aa = [df.x.min() - PAD_CELLS * dx, df.x.max() + PAD_CELLS * dx, df.y.min() - PAD_CELLS * dy, df.y.max() + PAD_CELLS * dy]
    aa = [max(aa[0], plon.min()), min(aa[1], plon.max()), max(aa[2], plat.min()), min(aa[3], plat.max())]
inwin = (lon >= aa[0]) & (lon <= aa[1]) & (lat >= aa[2]) & (lat <= aa[3])
lat0 = 0.5 * (aa[2] + aa[3])
m_per_deg = 111.32e3
cell_m = dy * m_per_deg
step = max(1, int(round((aa[1] - aa[0]) / dx / NGLYPH)))
fig_w = 12
fig_h = fig_w * (aa[3] - aa[2]) / ((aa[1] - aa[0]) * np.cos(np.pi * lat0 / 180)) * 0.95 + 1.0

def fld_slice(vn, con):
    """2D field for -fld: bar/surf from 2D vars, top/bot from the 3D fit."""
    if args.fld in ['top', 'bot'] and (args.fld + '_' + vn) not in ds:
        return ds['3d_' + vn].sel(con=con).isel(s_rho=-1 if args.fld == 'top' else 0).values
    return ds[args.fld + '_' + vn].sel(con=con).values

def masked(f):
    return np.where(show, f, np.nan)

def setup(ax):
    pfun.add_coast(ax, color='gray', linewidth=0.5)
    ax.axis(aa)
    pfun.dar(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='x', labelrotation=45, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(color='lightgray', linestyle='--', alpha=0.5)

def letter(ax, s):
    ax.text(0.025, 0.95, s, transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

def nice(x):
    """Round a speed [m/s] down to 1, 2 or 5 x 10^n."""
    e = 10 ** np.floor(np.log10(x))
    return max(m for m in [1, 2, 5, 10] if m * e <= x) * e

def glyphs(ax, Lsmaj, Lsmin, theta, color='k'):
    """Ellipse outlines on every step-th cell; scale set so the 99th pctl major
    axis fills half the glyph spacing (no overlap). Adds a reference bar."""
    jj, ii = np.meshgrid(np.arange(0, lon.shape[0], step), np.arange(0, lon.shape[1], step), indexing='ij')
    sel = show[jj, ii] & inwin[jj, ii] & np.isfinite(Lsmaj[jj, ii])
    jj, ii = jj[sel], ii[sel]
    big = np.nanpercentile(np.where(show & inwin, Lsmaj, np.nan), 99)
    sc = 0.5 * step * cell_m / big # map metres per (m/s)
    ref = nice(big) # bar length, a round number <= big
    ph = np.linspace(0, 2 * np.pi, 41)
    segs = []
    for j, i in zip(jj, ii):
        a, b, th = Lsmaj[j, i], Lsmin[j, i], np.deg2rad(theta[j, i])
        x, y = a * np.cos(ph), b * np.sin(ph)
        X = (x * np.cos(th) - y * np.sin(th)) * sc / (m_per_deg * np.cos(np.deg2rad(lat[j, i])))
        Y = (x * np.sin(th) + y * np.cos(th)) * sc / m_per_deg
        segs.append(np.column_stack([lon[j, i] + X, lat[j, i] + Y]))
    ax.add_collection(LineCollection(segs, colors=color, linewidths=0.6, zorder=3))
    # reference bar = one semi-major axis of `ref`, lower left
    x0 = aa[0] + 0.06 * (aa[1] - aa[0])
    y0 = aa[2] + 0.05 * (aa[3] - aa[2])
    L = ref * sc / (m_per_deg * np.cos(np.deg2rad(lat0)))
    ax.plot([x0, x0 + L], [y0, y0], '-', color=color, linewidth=2, zorder=4)
    ax.text(x0 + L / 2, y0 + 0.015 * (aa[3] - aa[2]), '%g cm/s' % (100 * ref), ha='center', va='bottom', fontsize=9)

def vrange(f, lo=2, hi=98):
    v = f[show & inwin & np.isfinite(f)]
    return np.percentile(v, lo), np.percentile(v, hi)

# ------------------------------------------------------------ ellipse maps ---
for con in CONS:
    Lsmaj, Lsmin = fld_slice('Lsmaj', con), fld_slice('Lsmin', con)
    theta, g = fld_slice('theta', con), fld_slice('g', con)
    ecc = Lsmin / Lsmaj
    fig, axd = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], layout='constrained', figsize=(fig_w, fig_h))
    panels = [
        ('a', 100 * Lsmaj, cmocean.cm.speed, (0, vrange(100 * Lsmaj)[1]), 'Semi-major axis [cm/s]'),
        ('b', ecc, cmocean.cm.balance, (-1, 1), 'Lsmin / Lsmaj  (+ counterclockwise)'),
        ('c', theta, cmocean.cm.phase, (0, 180), 'Inclination [deg CCW from east]'),
        ('d', g, cmocean.cm.phase, (0, 360), 'Greenwich phase [deg]'),
    ]
    for k, f, cmap, (v0, v1), lab in panels:
        ax = axd[k]
        cs = ax.pcolormesh(plon, plat, masked(f), cmap=cmap, vmin=v0, vmax=v1, shading='flat')
        fig.colorbar(cs, ax=ax, shrink=0.85, label=lab)
        setup(ax)
        letter(ax, k)
    glyphs(axd['a'], Lsmaj, Lsmin, theta)
    fig.suptitle('%s tidal ellipses (%s): %s, %s to %s' % (con, args.fld, args.gtagex, args.ds0, args.ds1))
    out_fn = out_dir / ('tidal_ellipses_%s_%s_%s_%s.png' % (args.job, args.dom, args.fld, con))
    fig.savefig(out_fn, dpi=200, bbox_inches='tight', facecolor='white') # white for now; was transparent=True
    plt.close(fig)
    print('Saved ' + str(out_fn))

# ---------------------------------------------------------------- cotidal ---
for con in CONS:
    fig, axd = plt.subplot_mosaic([[con + 'A', con + 'g']], layout='constrained', figsize=(fig_w, (fig_h - 1.0) / 2 + 1.0))
    A = 100 * ds.zeta_A.sel(con=con).values
    g = ds.zeta_g.sel(con=con).values
    # unwrap around the circular mean so a field straddling 0/360 stays continuous;
    # cyclic colormap only when the phase spread is large enough to need it
    gr = np.deg2rad(g[show & inwin & np.isfinite(g)])
    gm = np.rad2deg(np.arctan2(np.sin(gr).mean(), np.cos(gr).mean())) % 360
    g = (g - gm + 180) % 360 - 180 + gm
    gcmap = cmocean.cm.phase if np.ptp(vrange(g)) > 90 else cmocean.cm.tempo
    for k, f, cmap, (v0, v1), lab in [(con + 'A', A, cmocean.cm.amp, vrange(A), con + ' amplitude [cm]'), (con + 'g', g, gcmap, vrange(g), con + ' Greenwich phase [deg]')]:
        ax = axd[k]
        cs = ax.pcolormesh(plon, plat, masked(f), cmap=cmap, vmin=v0, vmax=v1, shading='flat')
        cb = fig.colorbar(cs, ax=ax, shrink=0.85, label=lab)
        cb.formatter.set_useOffset(False)
        cb.update_ticks()
        setup(ax)
    letter(axd[con + 'A'], 'a')
    letter(axd[con + 'g'], 'b')
    fig.suptitle('%s cotidal (zeta): %s, %s to %s' % (con, args.gtagex, args.ds0, args.ds1))
    out_fn = out_dir / ('tidal_cotidal_%s_%s_%s.png' % (args.job, args.dom, con))
    fig.savefig(out_fn, dpi=200, bbox_inches='tight', facecolor='white') # white for now; was transparent=True
    plt.close(fig)
    print('Saved ' + str(out_fn))

# --------------------------------------------- overviews, one panel per con ---
NCOL = 4
rows = [CONS[i:i + NCOL] for i in range(0, len(CONS), NCOL)]
rows[-1] = rows[-1] + ['.'] * (NCOL - len(rows[-1]))
row_h = (fig_h - 1.0) / 2 * 5 / 6 # same panel aspect as the 2-col figures at 5 in wide
for kind in ['ellipses', 'cotidal']:
    fig, axd = plt.subplot_mosaic(rows, layout='constrained', figsize=(5 * NCOL, row_h * len(rows) + 1.0))
    for n, con in enumerate(CONS):
        ax = axd[con]
        if kind == 'ellipses':
            Lsmaj = fld_slice('Lsmaj', con)
            f, cmap, lab = 100 * Lsmaj, cmocean.cm.speed, 'Semi-major axis [cm/s]'
        else:
            f, cmap, lab = 100 * ds.zeta_A.sel(con=con).values, cmocean.cm.amp, 'Amplitude [cm]'
        v0, v1 = (0, vrange(f)[1]) if kind == 'ellipses' else vrange(f)
        cs = ax.pcolormesh(plon, plat, masked(f), cmap=cmap, vmin=v0, vmax=v1, shading='flat')
        cb = fig.colorbar(cs, ax=ax, shrink=0.85, label=lab)
        cb.formatter.set_useOffset(False)
        cb.update_ticks()
        setup(ax)
        if kind == 'ellipses':
            glyphs(ax, Lsmaj, fld_slice('Lsmin', con), fld_slice('theta', con))
        ax.set_title('%s (%.2f h)' % (con, PERIOD.get(con, np.nan)), fontsize=11)
        letter(ax, 'abcdefghijklmnop'[n])
    if kind == 'ellipses':
        fig.suptitle('Tidal ellipses (%s), all constituents: %s, %s to %s' % (args.fld, args.gtagex, args.ds0, args.ds1))
        out_fn = out_dir / ('tidal_ellipses_%s_%s_%s_allcons.png' % (args.job, args.dom, args.fld))
    else:
        fig.suptitle('Cotidal (zeta) amplitude, all constituents: %s, %s to %s' % (args.gtagex, args.ds0, args.ds1))
        out_fn = out_dir / ('tidal_cotidal_%s_%s_allcons.png' % (args.job, args.dom))
    fig.savefig(out_fn, dpi=200, bbox_inches='tight', facecolor='white') # white for now; was transparent=True
    plt.close(fig)
    print('Saved ' + str(out_fn))

# ----------------------------------------------------- Eulerian residual ---
def mean_uv(P):
    """(u, v) time-mean velocity for bar/surf/bot, or top/bot of the 3D fit."""
    if P + '_umean' in ds:
        return ds[P + '_umean'].values, ds[P + '_vmean'].values
    iz = -1 if P == 'top' else 0
    return ds['3d_umean'].values[iz], ds['3d_vmean'].values[iz]

if 'bar_umean' in ds:
    PANELS = ['bar'] + (['surf'] if 'surf_umean' in ds else ['top'] if '3d_umean' in ds else [])
    PANELS += ['bot'] if ('bot_umean' in ds or '3d_umean' in ds) else []
    LABEL = {'bar': 'depth-averaged', 'surf': 'top layer', 'top': 'top layer', 'bot': 'bottom layer'}
    UV = {P: mean_uv(P) for P in PANELS}
    spd = {P: np.hypot(*UV[P]) for P in PANELS}
    big = np.nanpercentile(np.concatenate([spd[P][show & inwin] for P in PANELS]), 98)
    ref = nice(big)
    fig, axd = plt.subplot_mosaic([PANELS], layout='constrained', figsize=(6 * len(PANELS), (fig_h - 1.0) / 2 + 1.0))
    jj, ii = np.meshgrid(np.arange(0, lon.shape[0], step), np.arange(0, lon.shape[1], step), indexing='ij')
    for n, P in enumerate(PANELS):
        ax = axd[P]
        u, v = UV[P]
        cs = ax.pcolormesh(plon, plat, masked(100 * spd[P]), cmap=cmocean.cm.speed, vmin=0, vmax=100 * big, shading='flat')
        setup(ax)
        sel = show[jj, ii] & inwin[jj, ii] & np.isfinite(u[jj, ii])
        q = ax.quiver(lon[jj, ii][sel], lat[jj, ii][sel], u[jj, ii][sel], v[jj, ii][sel], scale=big / 0.3, scale_units='inches', width=0.004, color='k', zorder=3)
        ax.quiverkey(q, 0.12, 0.06, ref, '%g cm/s' % (100 * ref), labelpos='N', coordinates='axes', fontproperties={'size': 9})
        ax.set_title(LABEL[P], fontsize=11)
        letter(ax, 'abcd'[n])
    fig.colorbar(cs, ax=[axd[P] for P in PANELS], shrink=0.85, label='Eulerian residual speed [cm/s]')
    fig.suptitle('Eulerian residual (time-mean velocity): %s, %s to %s' % (args.gtagex, args.ds0, args.ds1))
    out_fn = out_dir / ('tidal_residual_%s_%s.png' % (args.job, args.dom))
    fig.savefig(out_fn, dpi=200, bbox_inches='tight', facecolor='white') # white for now; was transparent=True
    plt.close(fig)
    print('Saved ' + str(out_fn))

# --------------------------------------------------- vertical (3D fit only) ---
if '3d_Lsmaj' in ds:
    pc_lp = pd.read_pickle(sect_dir / 'pc_lp.p')
    imouth = np.flatnonzero(lon[0] < pc_lp.x.mean())[-1] # last rho column west of the section
    jm = np.flatnonzero(show[:, imouth])
    z0 = ds.z0_rho.values[:, jm, imouth] # (s_rho, nj)
    yy = np.broadcast_to(lat[jm, imouth], z0.shape)
    for con in CONS:
        L3 = {vn: ds['3d_' + vn].sel(con=con).values for vn in ['Lsmaj', 'Lsmin', 'theta']}
        fig, axd = plt.subplot_mosaic([['top', 'bot'], ['smaj', 'ecc']], layout='constrained', figsize=(fig_w, (fig_h - 1.0) + 4.5))
        vmax = vrange(100 * L3['Lsmaj'][-1])[1]
        for k, iz, lab in [('top', -1, 'top layer'), ('bot', 0, 'bottom layer')]:
            ax = axd[k]
            cs = ax.pcolormesh(plon, plat, masked(100 * L3['Lsmaj'][iz]), cmap=cmocean.cm.speed, vmin=0, vmax=vmax, shading='flat')
            setup(ax)
            glyphs(ax, L3['Lsmaj'][iz], L3['Lsmin'][iz], L3['theta'][iz])
            ax.axvline(lon[0, imouth], color='k', linestyle=':', linewidth=1)
            ax.set_title(lab, fontsize=11)
        fig.colorbar(cs, ax=[axd['top'], axd['bot']], shrink=0.85, label='Semi-major axis [cm/s]')
        for k, f, cmap, (v0, v1), lab in [('smaj', 100 * L3['Lsmaj'][:, jm, imouth], cmocean.cm.speed, (0, vmax), 'Semi-major axis [cm/s]'),
                                          ('ecc', (L3['Lsmin'] / L3['Lsmaj'])[:, jm, imouth], cmocean.cm.balance, (-1, 1), 'Lsmin / Lsmaj  (+ counterclockwise)')]:
            ax = axd[k]
            cs = ax.pcolormesh(yy, z0, f, cmap=cmap, vmin=v0, vmax=v1, shading='nearest')
            fig.colorbar(cs, ax=ax, shrink=0.85, label=lab)
            ax.set_xlabel('Latitude along mouth (lon %.4f)' % lon[0, imouth])
            ax.set_ylabel('z at zeta = 0 [m]')
            ax.grid(color='lightgray', linestyle='--', alpha=0.5)
        for k in ['top', 'bot', 'smaj', 'ecc']:
            letter(axd[k], {'top': 'a', 'bot': 'b', 'smaj': 'c', 'ecc': 'd'}[k])
        fig.suptitle('%s tidal ellipses vs depth: %s, %s to %s' % (con, args.gtagex, args.ds0, args.ds1))
        out_fn = out_dir / ('tidal_ellipses_%s_vertical_%s.png' % (args.job, con))
        fig.savefig(out_fn, dpi=200, bbox_inches='tight', facecolor='white') # white for now; was transparent=True
        plt.close(fig)
        print('Saved ' + str(out_fn))
