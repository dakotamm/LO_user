"""
Schematic: a classical estuary, a reverse estuary, and Penn Cove's two states, as along-channel
sections (head at LEFT, mouth at RIGHT, + x toward the mouth). Idealized, not to scale; shading is
relative salinity within each panel (cmocean haline: dark = fresh, light = salty) and the white lines are isohalines.

    a  classical (positive) estuary: buoyancy enters at the HEAD (river); water is denser toward the
       mouth at every depth; deep inflow / surface outflow. GM14 Fig. 6 covers these.
    b  reverse (inverse) estuary: buoyancy is LOST at the head (E >> P), dense water forms there;
       surface inflow / deep outflow. Not in GM14 (Fr_f < 0).
    c  Penn Cove, usual state: no river, P - E ~ 0; buoyancy enters at the MOUTH as Skagit-fed
       Saratoga Passage water. The cove is less stratified than its source (top-bottom 4.1 -> 3.7 g/kg
       mouth -> head), so the surface is saltier toward the head (inverted) but the deep water is
       fresher toward the head (classical); the deep gradient wins, plus the prevailing down-cove
       wind: deep in / surface out on ~81% of days (~150 m3/s). Tides do not mix it (M ~ 0.03).
    d  Penn Cove, up-cove wind event (1-4 d, ~19% of days): the layered exchange reverses.
       Wind and density forcing are the same order at tau ~ 0.02-0.06 Pa (12 tau / (g H^2 drho/dx) ~ 1).

Numbers come from 20260807_pc_alongchannel_wind, 20260818_pc_exchange_reversal,
20261106_pc_strat_counterflow and the 20261109_gm14_* scripts (wb1_t0_xn11abbur00).
The larger LATERAL exchange at the mouth (in north / out south, ~590 m3/s) is not drawn.

Output: LO_output/DM_outs/20261110_estuary_types_schematic/estuary_types_schematic.png

run 20261110_estuary_types_schematic.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import cmocean

from lo_tools import Lfun

Ldir = Lfun.Lstart()
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_estuary_types_schematic'
Lfun.make_dir(out_dir)

C_IN, C_OUT, C_FW, C_WIND = '#0072B2', '#D55E00', '#56B4E9', '#52514e'
INK, MUTED, LAND = '#0b0b0b', '#52514e', '#d9d4c7'
LBL = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75)
X0, X1, XM = 0.0, 1.45, 1.0                 # head, right edge, mouth
x = np.linspace(X0, X1, 400)
z = np.linspace(-1.3, 0, 300)
XX, ZZ = np.meshgrid(x, z)


def bottom(xv, kind):
    """Bed elevation: a beach at the head, then a gently deepening floor; Penn Cove drops into Saratoga Passage."""
    xv = np.asarray(xv, float)
    zb = -(0.62 * (1 - np.exp(-xv / 0.07)) + 0.18 * xv)
    if kind == 'pc':
        zb = np.where(xv > XM, zb - 0.9 * (1 - np.exp(-(xv - XM) / 0.12)), zb)
    return zb


def salinity(kind):
    """Relative salinity 0 (fresh) - 1 (salty) on (z, x)."""
    if kind == 'classical':
        s = 0.75 * XX / X1 + 0.5 * (-ZZ)
    elif kind == 'reverse':
        s = 0.35 + 0.5 * (1 - XX / X1) + 0.12 * (-ZZ)
    else:                                   # Penn Cove: stratification grows toward / beyond the mouth
        D = 0.18 + 0.55 * XX / X1
        s = 0.55 + D * (-ZZ - 0.42)
    return np.clip(s, 0, 1)


def arrow(ax, p0, p1, c, lw=3, ms=22, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=c,
                                 connectionstyle='arc3,rad=%.2f' % rad, zorder=5))


def section(ax, kind, title):
    k = 'pc' if kind.startswith('pc') else 'std'
    zb = bottom(x, k)
    s = np.ma.masked_where(ZZ < bottom(XX, k), salinity('pc' if k == 'pc' else kind))
    ax.contourf(XX, ZZ, s, levels=np.linspace(0, 1, 21), cmap=cmocean.cm.haline, alpha=0.85)
    ax.contour(XX, ZZ, s, levels=np.linspace(0.1, 0.9, 9), colors='w', linewidths=0.8, alpha=0.8)
    ax.fill_between(x, zb, -1.4, color=LAND, zorder=3)
    ax.plot(x, zb, color=MUTED, lw=1, zorder=4)
    ax.plot([X0, X1], [0, 0], color=MUTED, lw=1)
    ax.axvline(XM, ymin=0.25, ymax=0.83, color=MUTED, lw=0.8, ls=':')
    ax.text(0.03, 0.08, 'head', fontsize=9, color=MUTED, ha='left')
    ax.text(XM, 0.08, 'mouth', fontsize=9, color=MUTED, ha='center')
    ax.text(X1 - 0.02, 0.08 if k != 'pc' else 0.22, 'Saratoga Passage' if k == 'pc' else 'ocean', fontsize=9, color=MUTED, ha='right')
    ax.set_xlim(-0.22, X1)
    ax.set_ylim(-1.32, 0.42)
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=12, weight='bold')


def note(ax, text):
    ax.text(-0.2, -1.27, text, fontsize=8.5, color=INK, va='bottom', ha='left', zorder=6,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=MUTED, lw=0.6, alpha=0.9))


def wind(ax, direction, label):
    if direction > 0:                       # down-cove: toward the mouth
        arrow(ax, (0.25, 0.27), (0.75, 0.27), C_WIND, lw=2, ms=16)
    else:
        arrow(ax, (0.75, 0.27), (0.25, 0.27), C_WIND, lw=2, ms=16)
    ax.text(0.5, 0.32, label, fontsize=8.5, color=C_WIND, ha='center', va='bottom')


fig, axs = plt.subplots(2, 2, figsize=(15, 9.5), layout='constrained')

# ---------------------------------------------------- a: classical estuary
ax = axs[0, 0]
section(ax, 'classical', 'a  Classical (positive) estuary')
arrow(ax, (-0.2, -0.04), (0.06, -0.05), C_FW, lw=3)
ax.text(-0.2, 0.02, 'river $Q_R$', fontsize=9, color=C_IN)
arrow(ax, (0.2, -0.1), (0.85, -0.1), C_OUT)
arrow(ax, (0.9, -0.62), (0.3, -0.55), C_IN)
ax.text(0.55, -0.2, 'surface out', fontsize=9, color=C_OUT, ha='center', bbox=LBL)
ax.text(0.6, -0.72, 'deep in', fontsize=9, color=C_IN, ha='center', bbox=LBL)
note(ax, 'buoyancy enters at the HEAD (river)\n'
         'denser toward the mouth at every depth\n'
         'GM14 Fig. 6: Fr$_f$ 1e-4 to 1, M 0.15 to 3')

# ---------------------------------------------------- b: reverse estuary
ax = axs[0, 1]
section(ax, 'reverse', 'b  Reverse (inverse) estuary')
for xe in (0.08, 0.18, 0.28):
    arrow(ax, (xe, 0.0), (xe, 0.2), C_OUT, lw=1.5, ms=12, rad=0.3)
ax.text(0.33, 0.12, 'E $\\gg$ P', fontsize=10, color=C_OUT)
arrow(ax, (0.9, -0.1), (0.25, -0.1), C_IN)
arrow(ax, (0.3, -0.55), (0.9, -0.62), C_OUT)
ax.text(0.58, -0.2, 'surface in', fontsize=9, color=C_IN, ha='center', bbox=LBL)
ax.text(0.6, -0.72, 'deep out', fontsize=9, color=C_OUT, ha='center', bbox=LBL)
note(ax, 'buoyancy LOST at the head (evaporation): dense, salty water forms there\n'
         'denser toward the head\n'
         'not in GM14 (Fr$_f$ < 0)')

# ---------------------------------------------------- c, d: Penn Cove
for ax, title, state in ((axs[1, 0], 'c  Penn Cove: usual state (calm or down-cove wind)', +1),
                         (axs[1, 1], 'd  Penn Cove: up-cove wind event', -1)):
    section(ax, 'pc', title)
    arrow(ax, (1.44, -0.05), (1.12, -0.05), C_FW, lw=3)
    ax.text(1.44, 0.02, 'fresher Skagit-fed\nsurface water', fontsize=8, color=C_IN, ha='right', va='bottom')
    ax.text(0.02, -0.06, 'no river\nP$-$E $\\approx$ 0', fontsize=8, color=MUTED, va='top')
    for xc, zc in ((0.45, -0.32), (0.62, -0.36)):          # interior mixing erodes the imported stratification
        arrow(ax, (xc - 0.04, zc - 0.06), (xc + 0.04, zc + 0.06), 'w', lw=1.2, ms=10, rad=-0.9)
    ax.text(0.54, -0.44, 'interior mixing erodes\nthe imported stratification', fontsize=8, color='w', ha='center', va='top')
    if state > 0:
        wind(ax, +1, 'prevailing down-cove wind adds to it')
        arrow(ax, (0.22, -0.1), (0.9, -0.1), C_OUT)
        arrow(ax, (1.1, -0.66), (0.3, -0.6), C_IN)
        ax.text(0.56, -0.2, 'surface out', fontsize=9, color=C_OUT, ha='center', bbox=LBL)
        ax.text(0.7, -0.75, 'denser deep water slides in', fontsize=9, color=C_IN, ha='center', bbox=LBL)
        note(ax, 'buoyancy enters at the MOUTH; the cove is less stratified than its source (4.1 $\\to$ 3.7 g/kg)\n'
                 'surface saltier toward the head (inverted), deep water fresher toward the head (classical)\n'
                 'deep in / surface out on ~81% of days (~150 m$^3$/s); tides do not mix it (M $\\approx$ 0.03)\n'
                 'larger lateral exchange (in north / out south, ~590 m$^3$/s) not drawn')
    else:
        wind(ax, -1, 'up-cove wind piles surface water into the cove')
        arrow(ax, (0.9, -0.1), (0.22, -0.1), C_IN)
        arrow(ax, (0.3, -0.6), (1.1, -0.66), C_OUT)
        ax.text(0.56, -0.2, 'surface in', fontsize=9, color=C_IN, ha='center', bbox=LBL)
        ax.text(0.7, -0.75, 'deep return flow out', fontsize=9, color=C_OUT, ha='center', bbox=LBL)
        note(ax, 'same density structure; the layered exchange reverses\n'
                 'events last 1-4 d, ~19% of days\n'
                 'wind matches the density forcing at $\\tau \\approx$ 0.02-0.06 Pa (a steady 4-6 m/s along the cove)')

fig.suptitle('Where the buoyancy enters, and which way the layered exchange runs  '
             '(idealized; shading = relative salinity, dark = fresh, light = salty)', fontsize=11, x=0.01, ha='left')
fn = out_dir / 'estuary_types_schematic.png'
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
