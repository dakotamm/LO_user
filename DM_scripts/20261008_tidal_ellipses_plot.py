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

Figures (to ~/Desktop/pltz):
  tidal_ellipses_<job>_<dom>_<fld>_<CON>.png  major axis + glyphs, signed
      eccentricity (Lsmin/Lsmaj, + = counterclockwise), inclination, phase
  tidal_cotidal_<job>_<dom>.png               zeta amplitude and phase
  tidal_ellipses_<job>_vertical_<CON>.png     (3D fits only) surface vs bottom
      glyphs and the ellipse along the mouth (last rho column west of pc_lp)
"""

import argparse
from pathlib import Path

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
parser.add_argument('-cons', default='M2,K1', type=str)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
fn = Ldir['LOo'] / 'extract' / args.gtagex / 'tidal_ellipses' / (args.job + '_' + args.ds0 + '_' + args.ds1 + '.nc')
out_dir = Path.home() / 'Desktop' / 'pltz'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
CONS = args.cons.split(',')
PAD_CELLS = 10
NGLYPH = 16 # target glyphs across the window

ds = xr.open_dataset(fn)
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
    """Ellipse outlines on every step-th cell; scale set so the 95th pctl major
    axis fills half the glyph spacing. Adds a reference bar, returns the scale."""
    jj, ii = np.meshgrid(np.arange(0, lon.shape[0], step), np.arange(0, lon.shape[1], step), indexing='ij')
    sel = show[jj, ii] & inwin[jj, ii] & np.isfinite(Lsmaj[jj, ii])
    jj, ii = jj[sel], ii[sel]
    ref = nice(np.nanpercentile(np.where(show & inwin, Lsmaj, np.nan), 95))
    sc = 0.5 * step * cell_m / ref # map metres per (m/s)
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
    fig.savefig(out_fn, dpi=200, bbox_inches='tight', transparent=True)
    plt.close(fig)
    print('Saved ' + str(out_fn))

# ---------------------------------------------------------------- cotidal ---
fig, axd = plt.subplot_mosaic([[c + 'A', c + 'g'] for c in CONS], layout='constrained', figsize=(fig_w, (fig_h - 1.0) * len(CONS) / 2 + 1.0))
for n, con in enumerate(CONS):
    A = 100 * ds.zeta_A.sel(con=con).values
    g = ds.zeta_g.sel(con=con).values
    for k, f, cmap, (v0, v1), lab in [(con + 'A', A, cmocean.cm.amp, vrange(A), con + ' amplitude [cm]'), (con + 'g', g, cmocean.cm.phase, vrange(g), con + ' Greenwich phase [deg]')]:
        ax = axd[k]
        cs = ax.pcolormesh(plon, plat, masked(f), cmap=cmap, vmin=v0, vmax=v1, shading='flat')
        fig.colorbar(cs, ax=ax, shrink=0.85, label=lab)
        setup(ax)
    letter(axd[con + 'A'], 'abcdefgh'[2 * n])
    letter(axd[con + 'g'], 'abcdefgh'[2 * n + 1])
fig.suptitle('Cotidal (zeta): %s, %s to %s' % (args.gtagex, args.ds0, args.ds1))
out_fn = out_dir / ('tidal_cotidal_%s_%s.png' % (args.job, args.dom))
fig.savefig(out_fn, dpi=200, bbox_inches='tight', transparent=True)
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
        fig.savefig(out_fn, dpi=200, bbox_inches='tight', transparent=True)
        plt.close(fig)
        print('Saved ' + str(out_fn))
