"""
Schematic: two candidate retention mechanisms in Penn Cove -- SLOW FLOW (water
far from the mouth in sluggish residual flow leaves slowly) and the EDDY (the
counterclockwise inner-cove cell recirculating water) -- and how they depend on
each other through the strength of the cove exchange.

Numbers come from the velocity field only (no particle retention results):
  eddy presence at 4-10 m, Nov-Dec / Aug-Oct, its core radius / centre and the
  median rim speed and core turnover in present / absent hours
  (20261021_pc_eddy_center.py, z4-10 csv); mouth d(sigma0) Nov-Dec / Aug-Oct
  6.3 / 2.3 kg m-3 (20261028_pc_eddy_schematic.py). The link from slow flow and
  from the eddy to retention is drawn qualitatively.

Output: LO_output/DM_outs/20261028_pc_retention_schematic/pc_retention_schematic.png

  python 20261028_pc_retention_schematic.py
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261028_pc_retention_schematic'
Lfun.make_dir(out_dir)
LOo = Ldir['LOo']
M_PER_DEG, LAT0 = 111.32e3, 48.23
COS0 = np.cos(np.deg2rad(LAT0))
LON_MOUTH = pd.read_pickle(LOo / 'section_lines' / 'pc_lp.p').x.mean()
INK, MUTED, BLUE, ORANGE, GREEN = '#222222', '#6b6b6b', '#2a6fdb', '#e8590c', '#2b8a3e'

C = pd.read_csv(LOo / 'DM_outs' / '20261021_pc_eddy_center' / 'pc_cp' / 'pc_eddy_center_z4-10.csv')
mon = pd.to_datetime(C.time).dt.month
P = C[C.present.astype(bool)]
ED = dict(lon=LON_MOUTH + P.x_c.mean() / (M_PER_DEG * COS0), lat=LAT0 + P.y_c.mean() / M_PER_DEG, r=P.r_eq.median())
pr = C.present.astype(bool)
NUM = dict(drho_strong=6.3, drho_weak=2.3,
           pres_strong=100 * C.present[mon.isin([11, 12])].mean(), pres_weak=100 * C.present[mon.isin([8, 9, 10])].mean(),
           u_strong=100 * C.U_rot[pr].median(), u_weak=100 * C.U_rot[~pr].median(),
           T_strong=C.T_turn_h[pr].median(), r_weak=C.r_eq[~pr].median())
print(NUM)

g = xr.open_dataset(Ldir['grid'] / 'grid.nc')
lon, lat, h = g.lon_rho.values, g.lat_rho.values, g.h.values
mask = g.mask_rho.values.astype(bool)
g.close()

fig = plt.figure(figsize=(15, 11.5))
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.12)
g2 = gs[1].subgridspec(1, 2, wspace=0.08)


# ----------------------------------------------------- (a) causal links ---
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.axis('off')


def node(x, y, txt, fc, ec, w=0.27, hh=0.13, bold_first=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - hh / 2), w, hh, boxstyle='round,pad=0.012,rounding_size=0.02',
                                fc=fc, ec=ec, lw=1.4, zorder=2))
    lines = txt.split('\n')
    ax.text(x, y + 0.012 * (len(lines) - 1), lines[0], ha='center', va='center', fontsize=10.5,
            fontweight='bold' if bold_first else 'normal', color=INK, zorder=3)
    if len(lines) > 1:
        ax.text(x, y - 0.018, '\n'.join(lines[1:]), ha='center', va='top', fontsize=8.6, color=INK, zorder=3)


def edge(p0, p1, sign, note='', color=INK, lw=2.0, ls='-', rad=0.0, npos=0.5, noff=(0, 0), ha='center'):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=18, color=color, lw=lw, ls=ls,
                        connectionstyle='arc3,rad=%g' % rad, shrinkA=4, shrinkB=4, zorder=1)
    ax.add_patch(a)
    xm, ym = p0[0] + npos * (p1[0] - p0[0]) + noff[0], p0[1] + npos * (p1[1] - p0[1]) + noff[1]
    if sign:
        ax.text(xm, ym, sign, ha='center', va='center', fontsize=12, fontweight='bold', color=color,
                bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec=color, lw=1), zorder=4)
    if note:
        ax.text(xm + (0.03 if ha == 'left' else -0.03 if ha == 'right' else 0), ym - (0.05 if ha == 'center' else 0),
                note, ha=ha, va='center', fontsize=8.4, color=color if color != INK else MUTED, zorder=4)


node(0.5, 0.88, 'COVE EXCHANGE STRENGTH\nlateral exchange at the mouth (in north, out south), set by\nstratification; '
     'strong Nov-Dec and May-Jun, weak Aug-Oct', '#eef3fb', '#5b7fbf', w=0.42, hh=0.15)
node(0.2, 0.55, 'residual flow speed\nin the cove', '#f4f4f4', '#8a8a8a', w=0.2, hh=0.1)
node(0.8, 0.55, 'inflow past the\nsouthern lobe', '#f4f4f4', '#8a8a8a', w=0.2, hh=0.1)
node(0.2, 0.25, 'SLOW-FLOW RETENTION\nwater far from the mouth\nin sluggish flow stays', '#fff1e6', ORANGE, w=0.24)
node(0.8, 0.25, 'EDDY\ncounterclockwise cell over the lobe,\nstrongest at 4-10 m', '#e8f0fd', BLUE, w=0.24)
node(0.5, 0.06, 'RETENTION IN THE COVE\n', '#eaf6ec', GREEN, w=0.3, hh=0.08)

edge((0.42, 0.83), (0.25, 0.61), '+', rad=0.08, npos=0.5)
edge((0.58, 0.83), (0.75, 0.61), '+', rad=-0.08, npos=0.5)
edge((0.2, 0.5), (0.2, 0.32), '-', 'faster flow,\nless of it', color=ORANGE, npos=0.5, ha='left')
edge((0.8, 0.5), (0.8, 0.32), '+', 'spins it up (hypothesis:\nthe eddy follows each\nlayer\'s inflow)', color=BLUE,
     ls='--', npos=0.5, ha='left')
edge((0.27, 0.19), (0.39, 0.08), '+', '', color=GREEN, lw=3.2, npos=0.45)
ax.text(0.15, 0.115, 'water far from the mouth in\nslow flow leaves slowly', ha='center',
        va='center', fontsize=8.6, color=GREEN)
edge((0.73, 0.19), (0.61, 0.08), '0', '', color='#9a9a9a', lw=1.4, ls=':', npos=0.45)
ax.text(0.86, 0.115, '~0: recirculation adds little\nif water leaves within a turn or two',
        ha='center', va='center', fontsize=8.6, color=MUTED)
edge((0.31, 0.5), (0.68, 0.17), '-', '', color=ORANGE, lw=1.6, npos=0.5)
ax.text(0.53, 0.40, 'flushing: the same fast flow carries water\nout of the eddy before it can recirculate', ha='left',
        va='center', fontsize=8.4, color=ORANGE)
ax.text(0.5, 0.64, 'The forcing that spins the eddy up also\nflushes the cove, so the two mechanisms\nare ANTI-PHASED: '
        'retention is longest in\nweak-exchange periods (Aug-Oct, the low-DO\nseason), when the eddy is weak.',
        ha='center', va='center', fontsize=9.2, color=INK, style='italic',
        bbox=dict(boxstyle='round,pad=0.45', fc='white', ec='0.8'))
ax.set_title('(a) how the two retention mechanisms are linked  (+ / - = more / less;  dashed = hypothesis)',
             fontsize=11, loc='left')


# ------------------------------------------------- (b, c) regime maps ---
def regime_map(ax, strong):
    hv = np.where(mask, h, np.nan)
    ax.pcolormesh(lon, lat, hv, cmap='Greys', vmin=-10, vmax=70, shading='nearest', alpha=0.22, zorder=0)
    pfun.add_coast(ax, color='0.45', linewidth=0.8)
    if not strong:                              # slow, retentive water: the inner cove and the head
        inner = mask & (lon < -122.692)
        ax.contourf(lon, lat, inner.astype(float), levels=[0.5, 1.5], colors=[ORANGE], alpha=0.10, zorder=1)
        ax.text(-122.722, 48.2385, 'slow, retentive\ninner cove', fontsize=8.6, color=ORANGE, ha='center', zorder=6)
    w = 9 if strong else 2.2
    for (p0, p1, rad) in [((-122.646, 48.2435), (-122.705, 48.2335), 0.08),   # in along the north side
                          ((-122.700, 48.2245), (-122.646, 48.2235), 0.10)]:  # out along the south side
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='simple', mutation_scale=12 + 1.6 * w, lw=0,
                                     connectionstyle='arc3,rad=%g' % rad, color='#5b5b5b', alpha=0.75, zorder=3))
    th = np.radians(np.linspace(30, 330, 120))
    rr = ED['r'] * (0.9 if strong else 0.55)
    xs = ED['lon'] + rr * np.cos(th) / (M_PER_DEG * COS0)
    ys = ED['lat'] + rr * np.sin(th) / M_PER_DEG
    ax.plot(xs, ys, color=BLUE, lw=6 if strong else 1.6, ls='-' if strong else '--', solid_capstyle='round',
            zorder=4)
    ax.add_patch(FancyArrowPatch((xs[-6], ys[-6]), (xs[-1], ys[-1]), arrowstyle='-|>',
                                 mutation_scale=26 if strong else 12, color=BLUE, lw=0, zorder=5))
    ax.axis([-122.738, -122.640, 48.211, 48.252])
    pfun.dar(ax)
    ax.set_xticks([]); ax.set_yticks([])
    if strong:
        t = ('STRONG EXCHANGE (stratified mouth, %.1f kg m$^{-3}$: Nov-Dec, also May-Jun)\n'
             'fast inflow / outflow; eddy present %.0f%% of hours (4-10 m, Nov-Dec),\n'
             'rim speed %.1f cm/s, core turnover %.0f h'
             % (NUM['drho_strong'], NUM['pres_strong'], NUM['u_strong'], NUM['T_strong']))
        ttl = '(b) strong exchange: strong eddy, fast flushing'
    else:
        t = ('WEAK EXCHANGE (weakly stratified mouth, %.1f kg m$^{-3}$: Aug-Oct)\n'
             'slow residual flow; eddy present %.0f%% of hours (4-10 m, Aug-Oct);\n'
             'otherwise a small weak patch (radius ~%.0f m, rim speed %.1f cm/s)'
             % (NUM['drho_weak'], NUM['pres_weak'], NUM['r_weak'], NUM['u_weak']))
        ttl = '(c) weak exchange: weak eddy, slow flow, long retention'
    ax.text(0.015, 0.03, t, transform=ax.transAxes, fontsize=8.6, color=INK, va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.8', alpha=0.92), zorder=6)
    ax.set_title(ttl, fontsize=11, loc='left')


regime_map(fig.add_subplot(g2[0]), True)
regime_map(fig.add_subplot(g2[1]), False)

fig.suptitle('Retention in Penn Cove: slow flow vs the inner-cove eddy, and how they depend on each other '
             '(wb1_t0_xn11abbur00, 2024-25)', fontsize=12, y=0.95)
fn = out_dir / 'pc_retention_schematic.png'
fig.savefig(fn, dpi=200, transparent=True, bbox_inches='tight')
plt.close(fig)
print('saved ' + str(fn))
