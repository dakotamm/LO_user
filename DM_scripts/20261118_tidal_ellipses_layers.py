"""
Top-third, bottom-third and depth-averaged tidal ellipses on ONE map per
constituent, major constituents only. Uses the fits from 20261008_tidal_ellipses.py.

  python 20261118_tidal_ellipses_layers.py -job pc_cove -dom box
  python 20261118_tidal_ellipses_layers.py -job pc_cove -dom box -cons M2,S2,N2,K1,O1,P1

-cons  default M2,S2,K1,O1 (the four largest depth-averaged currents in pc_cove)
-dom   as in 20261008_tidal_ellipses_plot.py

LAYER ELLIPSES
The 20261008 'top'/'bot' were the single surface/bottom s-layers. Here the
3D fit (one ellipse per s-layer) is averaged over the top third (0 to -h/3) and
bottom third (-2h/3 to -h) of the column at zeta = 0. Ellipse parameters are not
linear in u, v, so each layer's ellipse is converted to its rotary components
  W+ = (Lsmaj + Lsmin)/2 exp(i(theta - g)),  W- = (Lsmaj - Lsmin)/2 exp(i(theta + g))
which are linear; those are thickness-weighted averaged (weight = layer overlap with
the third) and converted back. Averaging over the whole column this way should
reproduce the ubar/vbar ('bar') fit -- printed as a check.
Needs a 3D fit (pc_cove), not sp_head_tide.

Figure (to LO_output/DM_outs/20261118_tidal_ellipses_layers/):
  tidal_ellipses_layers_<job>_<dom>.png  one panel per constituent; top third,
      bottom third and depth-averaged ellipses overlaid, one glyph scale per panel
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-job', default='pc_cove', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-dom', default='box', type=str)
parser.add_argument('-cons', default='M2,S2,K1,O1', type=str)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
fn = Ldir['LOo'] / 'extract' / args.gtagex / 'tidal_ellipses' / (args.job + '_' + args.ds0 + '_' + args.ds1 + '.nc')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261118_tidal_ellipses_layers'
Lfun.make_dir(out_dir)
sect_dir = Ldir['LOo'] / 'section_lines'
PAD_CELLS = 10
NGLYPH = 11 # target glyphs across the window (fewer than 20261008: three ellipses per site)
FILL = 0.9 # 99th pctl major axis spans this fraction of the glyph spacing

ds = xr.open_dataset(fn)
if '3d_Lsmaj' not in ds:
    raise SystemExit('No 3D fit in ' + str(fn) + ' -- top/bottom thirds need one (use -job pc_cove).')
CONS = args.cons.split(',')
PERIOD = {'M2': 12.42, 'S2': 12.00, 'N2': 12.66, 'K2': 11.97, 'K1': 23.93, 'O1': 25.82,
    'P1': 24.07, 'Q1': 26.87, 'M4': 6.21, 'MS4': 6.10, 'M6': 4.14} # hours
LAYERS = ['top', 'bot', 'bar']
LABEL = {'top': 'top third', 'bot': 'bottom third', 'bar': 'depth-averaged'}
COLOR = {'top': '#eb6834', 'bot': '#2a78d6', 'bar': 'k'}
LW = {'top': 1.0, 'bot': 1.0, 'bar': 0.8}
lon = ds.lon_rho.values
lat = ds.lat_rho.values
dx = np.nanmedian(np.diff(lon[0]))
dy = np.nanmedian(np.diff(lat[:, 0]))
plon, plat = pfun.get_plon_plat(lon, lat)

# ------------------------------------------------------------ window ---
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

# ------------------------------------------------- layer weights (thirds) ---
# w-levels at zeta = 0: -h at the bed, midpoints between rho levels, 0 at the top
h = ds.h.values
z0 = ds.z0_rho.values # (s_rho, eta, xi), bottom first
zw = np.concatenate([-h[None], 0.5 * (z0[1:] + z0[:-1]), np.zeros_like(h)[None]])

def overlap_weights(zlo, zhi):
    """Thickness of each layer inside [zlo, zhi] (2D bounds), normalized to sum 1."""
    w = np.clip(np.minimum(zw[1:], zhi) - np.maximum(zw[:-1], zlo), 0, None)
    return w / w.sum(axis=0)

WT = {'top': overlap_weights(-h / 3, np.zeros_like(h)),
      'bot': overlap_weights(-h, -2 * h / 3),
      'all': overlap_weights(-h, np.zeros_like(h))}

def to_rotary(Lsmaj, Lsmin, theta, g):
    th, gg = np.deg2rad(theta), np.deg2rad(g)
    return 0.5 * (Lsmaj + Lsmin) * np.exp(1j * (th - gg)), 0.5 * (Lsmaj - Lsmin) * np.exp(1j * (th + gg))

def from_rotary(Wp, Wm):
    ap, am = np.abs(Wp), np.abs(Wm)
    theta = np.rad2deg(0.5 * (np.angle(Wp) + np.angle(Wm))) % 180 # g not needed for glyphs
    return ap + am, ap - am, theta

def layer_ellipse(con, L):
    """(Lsmaj, Lsmin, theta) for L in top/bot/all (3D fit averaged) or bar (ubar fit)."""
    if L == 'bar':
        return tuple(ds['bar_' + vn].sel(con=con).values for vn in ['Lsmaj', 'Lsmin', 'theta'])
    Wp, Wm = to_rotary(*[ds['3d_' + vn].sel(con=con).values for vn in ['Lsmaj', 'Lsmin', 'theta', 'g']])
    return from_rotary((WT[L] * Wp).sum(axis=0), (WT[L] * Wm).sum(axis=0))

# check: whole-column average of the 3D fit vs the ubar fit
cell = show & inwin
for con in CONS:
    a3, _, _ = layer_ellipse(con, 'all')
    ab, _, _ = layer_ellipse(con, 'bar')
    ok = cell & np.isfinite(a3) & np.isfinite(ab)
    print('%s check: column-mean 3D Lsmaj vs bar Lsmaj, median |diff| = %.2f cm/s (median bar %.2f cm/s)'
        % (con, 100 * np.median(np.abs(a3 - ab)[ok]), 100 * np.median(ab[ok])))

# ------------------------------------------------------------- plotting ---
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

jj, ii = np.meshgrid(np.arange(0, lon.shape[0], step), np.arange(0, lon.shape[1], step), indexing='ij')
ph = np.linspace(0, 2 * np.pi, 41)

def glyph_segs(Lsmaj, Lsmin, theta, sc):
    sel = show[jj, ii] & inwin[jj, ii] & np.isfinite(Lsmaj[jj, ii])
    segs = []
    for j, i in zip(jj[sel], ii[sel]):
        a, b, th = Lsmaj[j, i], Lsmin[j, i], np.deg2rad(theta[j, i])
        x, y = a * np.cos(ph), b * np.sin(ph)
        X = (x * np.cos(th) - y * np.sin(th)) * sc / (m_per_deg * np.cos(np.deg2rad(lat[j, i])))
        Y = (x * np.sin(th) + y * np.cos(th)) * sc / m_per_deg
        segs.append(np.column_stack([lon[j, i] + X, lat[j, i] + Y]))
    return segs

NCOL = 2 if len(CONS) <= 4 else 3
rows = [CONS[i:i + NCOL] for i in range(0, len(CONS), NCOL)]
rows[-1] = rows[-1] + ['.'] * (NCOL - len(rows[-1]))
pw = 6.0 # panel width [in]
ph_in = pw * (aa[3] - aa[2]) / ((aa[1] - aa[0]) * np.cos(np.pi * lat0 / 180))
fig, axd = plt.subplot_mosaic(rows, layout='constrained', figsize=(pw * NCOL, (ph_in + 0.6) * len(rows) + 0.8))
for n, con in enumerate(CONS):
    ax = axd[con]
    setup(ax)
    E = {L: layer_ellipse(con, L) for L in LAYERS}
    # one scale per panel: 99th pctl of the largest of the three spans FILL of the glyph spacing
    big = np.nanpercentile(np.concatenate([E[L][0][cell] for L in LAYERS]), 99)
    sc = FILL * step * cell_m / (2 * big) # map metres per (m/s); 2*big = full major axis
    for L in ['bot', 'top', 'bar']: # depth-averaged drawn last, on top
        ax.add_collection(LineCollection(glyph_segs(*E[L], sc), colors=COLOR[L], linewidths=LW[L], zorder=3))
    ref = nice(big)
    x0 = aa[0] + 0.06 * (aa[1] - aa[0])
    y0 = aa[2] + 0.05 * (aa[3] - aa[2])
    Lbar = ref * sc / (m_per_deg * np.cos(np.deg2rad(lat0)))
    ax.plot([x0, x0 + Lbar], [y0, y0], '-', color='k', linewidth=2, zorder=4)
    ax.text(x0 + Lbar / 2, y0 + 0.015 * (aa[3] - aa[2]), '%g cm/s' % (100 * ref), ha='center', va='bottom', fontsize=9)
    ax.set_title('%s (%.2f h)' % (con, PERIOD.get(con, np.nan)), fontsize=11)
    letter(ax, 'abcdefghi'[n])
    for L in LAYERS:
        print('%s %s: median Lsmaj %.2f cm/s' % (con, LABEL[L], 100 * np.nanmedian(E[L][0][cell])))
handles = [Line2D([], [], color=COLOR[L], linewidth=1.5, label=LABEL[L]) for L in LAYERS]
fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False, fontsize=10)
fig.suptitle('Tidal ellipses by layer: %s, %s to %s' % (args.gtagex, args.ds0, args.ds1))
out_fn = out_dir / ('tidal_ellipses_layers_%s_%s.png' % (args.job, args.dom))
fig.savefig(out_fn, dpi=200, bbox_inches='tight', transparent=True)
plt.close(fig)
print('Saved ' + str(out_fn))
