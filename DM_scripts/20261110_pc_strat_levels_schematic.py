"""
Schematic: Penn Cove at four stratification levels, set by the Skagit flow that reached Saratoga
Passage ~3 days earlier (wb1_t0_xn11abbur00, 2024-2025). Along-channel section, head at LEFT,
mouth at the dotted line, upper Saratoga Passage at right.

Every number is a scenario mean from 20261110_pc_skagit_scenarios.py (daily, Godin-filtered):
    scenario     lagged Skagit flow terciles (low / mid / high) and the top 5 % (flood)
    salinities   top and bottom layers at pc_cp (head) and pc_lp (mouth), Saratoga surface (pc_lp_p)
    I_v          layered exchange at pc_lj (+ = deep in / surface out), all days and days without
                 up-cove wind; reversals = days with I_v < 0
    Fr_eff       (ds / (alpha s_ocn))^(3/2) at pc_lj, located on the Fr_f axis of GM14 Fig. 6
                 (dashed boundaries: fjord / strongly stratified ~2e-3, strongly stratified / salt wedge ~7e-2)

DRAWING
    Shading is real salinity on one scale for all panels (cmocean haline, 17-31 g/kg). The along-cove
    differences (tenths of a g/kg) are too small to see in shading, so the surface and bottom values
    at the head, the mouth and in Saratoga Passage are printed. The vertical SHAPE (a pycnocline at
    ~1/3 of the cove depth) is assumed and identical in every panel; only the top and bottom
    salinities change. Skagit and exchange arrow widths scale with their flows.
    M ~ 0.03 in every panel (the tide does not change with the river), so all four stay far on
    the stratified side of the GM14 mixing line. The lateral exchange (~265-300 m3/s, in north /
    out south) is not drawn.

INPUT  LO_output/DM_outs/20261110_pc_skagit_scenarios/{daily,scenarios}_pc_lj.csv
OUTPUT LO_output/DM_outs/20261110_pc_strat_levels_schematic/pc_strat_levels_schematic.png

run 20261110_pc_strat_levels_schematic.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
import cmocean

from lo_tools import Lfun

Ldir = Lfun.Lstart()
in_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_skagit_scenarios'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_strat_levels_schematic'
Lfun.make_dir(out_dir)

X = pd.read_csv(in_dir / 'daily_pc_lj.csv', index_col=0, parse_dates=True)
S = pd.read_csv(in_dir / 'scenarios_pc_lj.csv', index_col=0)
ORDER = ['low', 'mid', 'high', 'flood']
V = X.groupby('scenario')[['stop_pc_cp', 'sbot_pc_cp', 'stop_pc_lp', 'sbot_pc_lp', 's_out']].mean().loc[ORDER]
MONTHS = X.groupby('scenario').apply(lambda g: 'mostly ' + ', '.join(g.index.strftime('%b').value_counts().index[:3]),
                                     include_groups=False).loc[ORDER]

C_IN, C_OUT, C_FW, C_WIND = '#0072B2', '#D55E00', '#56B4E9', '#52514e'
INK, MUTED, LAND = '#0b0b0b', '#52514e', '#d9d4c7'
LBL = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8)
NORM = Normalize(17, 31)
CMAP = cmocean.cm.haline
X0, X1, XM, HC = 0.0, 1.45, 1.0, 0.8        # head, right edge, mouth, cove reference depth
ZETA0, W = 0.33, 0.12                       # assumed pycnocline: fraction of HC, thickness
x = np.linspace(X0, X1, 400)
z = np.linspace(-1.3, 0, 300)
XX, ZZ = np.meshgrid(x, z)


def bottom(xv):
    xv = np.asarray(xv, float)
    zb = -(0.62 * (1 - np.exp(-xv / 0.07)) + 0.18 * xv)
    return np.where(xv > XM, zb - 0.9 * (1 - np.exp(-(xv - XM) / 0.12)), zb)


def field(v):
    """Salinity on (z, x): top/bottom linear head -> mouth -> Saratoga, fixed tanh pycnocline."""
    s_top = np.interp(XX, [X0, XM, X1], [v.stop_pc_cp, v.stop_pc_lp, v.s_out])
    s_bot = np.interp(XX, [X0, XM, X1], [v.sbot_pc_cp, v.sbot_pc_lp, v.sbot_pc_lp])
    zeta = np.clip(-ZZ / HC, 0, 1)
    g = lambda q: np.tanh((q - ZETA0) / W)
    f = (g(zeta) - g(0)) / (g(1) - g(0))
    return s_top + (s_bot - s_top) * f


def arrow(ax, p0, p1, c, lw=3, ms=22, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c,
                                 connectionstyle='arc3,rad=%.2f' % rad, zorder=5))


def gm14_band(fr):
    if fr < 2e-3:
        return 'fjord band'
    if fr < 7e-2:
        return 'strongly stratified band' + (' (near the salt-wedge line)' if fr > 3e-2 else
                                             ' (near the fjord line)' if fr < 4e-3 else '')
    return 'salt-wedge band'


fig, axs = plt.subplots(2, 2, figsize=(15, 9.8), layout='constrained')
TITLES = {'low': 'Low stratification', 'mid': 'Moderate stratification',
          'high': 'High stratification', 'flood': 'Flood (Skagit storm flow)'}
for ax, key, tag in zip(axs.flat, ORDER, 'abcd'):
    v, sc = V.loc[key], S.loc[key]
    s = np.ma.masked_where(ZZ < bottom(XX), field(v))
    cf = ax.contourf(XX, ZZ, s, levels=np.arange(17, 31.01, 0.5), cmap=CMAP, norm=NORM, extend='both')
    ax.contour(XX, ZZ, s, levels=np.arange(18, 31, 2), colors='w', linewidths=0.7, alpha=0.8)
    zb = bottom(x)
    ax.fill_between(x, zb, -1.4, color=LAND, zorder=3)
    ax.plot(x, zb, color=MUTED, lw=1, zorder=4)
    ax.plot([X0, X1], [0, 0], color=MUTED, lw=1)
    ax.axvline(XM, ymin=0.25, ymax=0.8, color=MUTED, lw=0.8, ls=':')
    ax.text(0.03, 0.06, 'head', fontsize=9, color=MUTED)
    ax.text(XM, 0.06, 'mouth', fontsize=9, color=MUTED, ha='center')

    # salinities at the head, the mouth and in Saratoga Passage (surface / bottom)
    for xs, st, sb in ((0.06, v.stop_pc_cp, v.sbot_pc_cp), (XM - 0.04, v.stop_pc_lp, v.sbot_pc_lp)):
        ax.text(xs, -0.04, '%.1f' % st, fontsize=9, color=INK, va='top', bbox=LBL, zorder=6,
                ha='left' if xs < 0.5 else 'right')
        ax.text(xs + (0.12 if xs < 0.5 else 0), bottom(xs + (0.12 if xs < 0.5 else 0)) + 0.04, '%.1f' % sb,
                fontsize=9, color=INK, va='bottom', bbox=LBL, zorder=6, ha='left' if xs < 0.5 else 'right')
    ax.text(X1 - 0.02, -0.04, '%.1f' % v.s_out, fontsize=9, color=INK, va='top', ha='right', bbox=LBL, zorder=6)
    ax.text(0.5, -0.04, 'surface %+.2f head$-$mouth (inverted)' % (v.stop_pc_cp - v.stop_pc_lp),
            fontsize=8, color=INK, ha='center', va='top', bbox=LBL, zorder=6)
    ax.text(0.62, bottom(0.62) + 0.04, 'bottom %+.2f head$-$mouth (classical)' % (v.sbot_pc_cp - v.sbot_pc_lp),
            fontsize=8, color=INK, ha='center', va='bottom', bbox=LBL, zorder=6)

    # Skagit water entering Saratoga Passage, width ~ flow
    lw_r = 1.5 + sc.Q_skagit / 170
    arrow(ax, (1.44, 0.13), (1.12, 0.13), C_FW, lw=lw_r, ms=14 + lw_r)
    ax.text(1.44, 0.2, 'Skagit %.0f m$^3$/s (3 d earlier)' % sc.Q_skagit, fontsize=8.5, color=C_IN, ha='right')

    # layered exchange, width ~ I_v
    lw_e = 1 + sc.Iv / 22
    arrow(ax, (0.25, -0.22), (0.88, -0.22), C_OUT, lw=lw_e, ms=14 + lw_e)
    arrow(ax, (1.08, -0.5), (0.32, -0.44), C_IN, lw=lw_e, ms=14 + lw_e)
    ax.text(0.56, -0.3, 'surface out', fontsize=9, color=C_OUT, ha='center', va='top', bbox=LBL, zorder=6)
    ax.text(0.72, -0.55, 'deep in', fontsize=9, color=C_IN, ha='center', va='top', bbox=LBL, zorder=6)

    ax.text(-0.2, -1.27,
            'top$-$bottom $\\Delta s$: mouth %.1f, head %.1f g/kg\n'
            'layered exchange $I_v$ %.0f m$^3$/s (%.0f without up-cove wind)\n'
            'reversed on %.0f%% of days; up-cove wind on %.0f%%\n'
            'Fr$_{eff}$ = %.3f: GM14 %s; M $\\approx$ 0.03'
            % (sc.ds_pc_lp, sc.ds_pc_cp, sc.Iv, sc.Iv_no_upcove, sc.pct_reversed, sc.pct_upcove,
               sc.Fr_eff, gm14_band(sc.Fr_eff)),
            fontsize=8.5, color=INK, va='bottom', ha='left', zorder=6,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=MUTED, lw=0.6, alpha=0.9))
    ax.set_xlim(-0.22, X1)
    ax.set_ylim(-1.32, 0.3)
    ax.axis('off')
    ax.set_title('%s  %s  (%s; %d days)' % (tag, TITLES[key], MONTHS[key], sc.n_days), loc='left',
                 fontsize=12, weight='bold')

cb = fig.colorbar(cf, ax=axs, shrink=0.5, pad=0.01, label='salinity [g/kg]')
fig.suptitle('Penn Cove at four stratification levels, set by Skagit flow '
             '(wb1_t0, 2024-2025; salinities are scenario means, vertical shape idealized)',
             fontsize=11, x=0.01, ha='left')
fn = out_dir / 'pc_strat_levels_schematic.png'
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
