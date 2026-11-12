"""
DIAGRAM version of 20261110_pc_conceptual_plan_map.py: how the flood, ebb and residual
circulation of Penn Cove (one flow-defined layer, plan view) changes with the tide,
stratification and wind, drawn as a few arrows instead of cell-by-cell difference maps.

PATHWAYS (fixed, the same in every panel): the cove cut into three reaches at the section
lines, inner (head to pc_cp, box xi_rho 1-15), middle (pc_cp to pc_lj, 16-23) and outer
(pc_lj to the mouth, 24-30), each split into a NORTH and a SOUTH half at the midpoint between
the shores of every rho column (cells on the midpoint are left out). One arrow per pathway:
    direction = the area-weighted mean velocity vector of the composite over that region
    length and width  proportional to its speed (one scale for flood and ebb, one for the
    residual; key arrows in the baseline panels)
In the driver rows each pathway has two arrows side by side: the first condition SOLID
(dark), the second OPEN (outlined), so the change is read from the arrow sizes.
LOBE CELL (residual panels): a 270-degree purple ring around the 500 m lobe disk, line width
proportional to the mean residual vorticity there / f (solid = first condition, dashed = second,
inside it); the arrowhead shows the sense (+ = counterclockwise); both values printed.

COMPOSITES as 20261110_pc_conceptual_plan_map.py (transport-weighted, layers_hourly.nc;
flood / ebb = sign of mouth Q; classes of pc_tidal_cycles.py). The text boxes and their
numbers are the same as there (upper-layer reading for -layer upper only).

OUTPUT LO_output/DM_outs/20261110_pc_conceptual_plan_map/pc_conceptual_plan_diagram_<layer>.png

run 20261110_pc_conceptual_plan_diagram.py [-layer upper|depth_mean|lower]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrow, FancyArrowPatch
from matplotlib.colors import ListedColormap
import numpy as np
import xarray as xr

from lo_tools import Lfun
from pc_tidal_cycles import cycle_masks

parser = argparse.ArgumentParser()
parser.add_argument('-layer', default='upper', choices=['upper', 'depth_mean', 'lower'])
LAYER = parser.parse_args().layer
L = {'depth_mean': 0, 'upper': 1, 'lower': 2}[LAYER]

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}
LOBE = (-122.708, 48.226, 500.0)          # lon, lat, radius [m]
PHASES = ['flood', 'ebb', 'residual']
PCOL = {'flood': '#B2182B', 'ebb': '#2166AC', 'residual': '#333333'}   # solid-arrow color per column
REACH = {'inner': (1, 15), 'middle': (16, 23), 'outer': (24, 30)}      # box xi_rho ranges
LMAX = 1.25                               # km, arrow length of the fastest pathway on the page (per scale)

# rows and text: as 20261110_pc_conceptual_plan_map.py
ROWS = [   # (first, second, box color, driver box, upper-layer reading)
    ('spring', 'neap', '#DCE9F5',
     'SPRING vs NEAP\nsemidiurnal tide (M2 + S2), ~14.8 d\nthirds of the D2 envelope of the\nmouth transport, within each month',
     'Faster flood and ebb along the axis seaward of pc_cp;\nthe residual circulation does not change.'),
    ('tropic', 'equatorial', '#FBE6D4',
     'TROPIC vs EQUATORIAL\ndiurnal tide (K1 + O1), ~13.7 d\nthirds within each month\n(season balanced)',
     'Same pattern, somewhat stronger residual and\nlobe cell on tropic tides.'),
    ('high strat', 'low strat', '#EFE0EF',
     'HIGH vs LOW STRATIFICATION\n$\\Delta\\sigma_0$ bottom - top, record thirds,\nall hours',
     'Turns the inner LOBE CELL on: at low strat the inner\ncove residual is nearly still (same in calm hours);\nthe outer in-north / out-south exchange stays.'),
    ('up-cove', 'down-cove', '#DDEEE4',
     'UP-COVE vs DOWN-COVE STRESS\nalong-cove stress ROMS applied,\n> +0.01 vs < -0.01 Pa (Godin)',
     'Up-cove: surface water driven to the head along the\naxis, stronger cell (if stratified). Down-cove: surface\noutflow seaward of pc_lj. Partly strat (up-cove is winter).'),
    ('northward', 'southward', '#F3F0D9',
     'NORTHWARD vs SOUTHWARD STRESS\ncross-cove stress,\n> +0.01 vs < -0.01 Pa (Godin),\n+ toward the north shore',
     'The most different pair: northward -> surface inflow\nalong the inner-cove axis and a lobe cell; southward ->\nsurface outflow along the axis seaward of pc_cp, no cell.'),
]

Ldir = Lfun.Lstart(gridname='wb1')
in_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_cove_layer_maps'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_conceptual_plan_map'
out_dir.mkdir(parents=True, exist_ok=True)

d = xr.open_dataset(in_dir / 'layers_hourly.nc')
g = xr.open_dataset(Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_' + DS0 + '_' + DS1 + '.nc'))
LON, LAT, MASK = g.lon_rho.values, g.lat_rho.values, g.mask_rho.values == 1
PM, PN, F = g.pm.values, g.pn.values, g.f.values
g.close()
JJ, II = d.eta.values, d.xi.values
dA = 1 / (PM[JJ, II] * PN[JJ, II])
Ux, Vy, H = d.Ux.values.astype(float), d.Vy.values.astype(float), d.H.values.astype(float)
Ux, Vy, H = [np.concatenate([a.sum(axis=1, keepdims=True), a], axis=1)[:, L] for a in (Ux, Vy, H)]
Qm = d.Qmouth.values
CM = cycle_masks(d.time.values)
CM['all'] = np.ones(d.time.size, bool)
PH = {'flood': Qm > 0, 'ebb': Qm < 0, 'residual': np.ones(d.time.size, bool)}

# km coordinates (isotropic), origin at the mouth
km = 111.32
LON0, LAT0 = LON[0, 30], LAT[0, 0]
X = (LON - LON0) * km * np.cos(np.deg2rad(48.23))
Y = (LAT - LAT0) * km
XL = (LOBE[0] - LON0) * km * np.cos(np.deg2rad(48.23))
YL = (LOBE[1] - LAT0) * km
DISK = np.hypot(X[JJ, II] - XL, Y[JJ, II] - YL) * 1000 <= LOBE[2]
f0 = F[JJ, II][DISK].mean()

# pathways: reach x north / south half of each rho column
REG = {}
for rn, (i0, i1) in REACH.items():
    for side in ['north', 'south']:
        sel = np.zeros(len(JJ), bool)
        for i in range(i0, i1 + 1):
            jw = JJ[II == i]
            mid = 0.5 * (jw.min() + jw.max())
            sel |= (II == i) & ((JJ > mid) if side == 'north' else (JJ < mid))
        REG[rn, side] = dict(sel=sel, x=X[JJ, II][sel].mean(), y=Y[JJ, II][sel].mean())


def composite(m):
    Hs = H[m].sum(axis=0)
    with np.errstate(invalid='ignore'):
        return Ux[m].sum(axis=0) / Hs, Vy[m].sum(axis=0) / Hs


def to2d(a):
    out = np.full(LON.shape, np.nan)
    out[JJ, II] = a
    return out


def vorticity(u, v):
    """dv/dx - du/dy at the cove cells, as 20261110_pc_cove_residual_conditions.py"""
    U, V = to2d(u), to2d(v)
    z = np.full(len(JJ), np.nan)
    for n, (j, i) in enumerate(zip(JJ, II)):
        il, ir = (i - 1 if np.isfinite(V[j, i - 1]) else i), (i + 1 if np.isfinite(V[j, i + 1]) else i)
        jl, jr = (j - 1 if np.isfinite(U[j - 1, i]) else j), (j + 1 if np.isfinite(U[j + 1, i]) else j)
        if ir > il and jr > jl:
            z[n] = (V[j, ir] - V[j, il]) * PM[j, i] / (ir - il) - (U[jr, i] - U[jl, i]) * PN[j, i] / (jr - jl)
    return z


def rms(u, v):
    ok = np.isfinite(u)
    return np.sqrt(((u[ok]**2 + v[ok]**2) * dA[ok]).sum() / dA[ok].sum())


def patt(a, b):
    ok = np.isfinite(a[0]) & np.isfinite(b[0])
    w = dA[ok]
    num = ((a[0][ok] * b[0][ok] + a[1][ok] * b[1][ok]) * w).sum()
    return num / np.sqrt(((a[0][ok]**2 + a[1][ok]**2) * w).sum() * ((b[0][ok]**2 + b[1][ok]**2) * w).sum())


def pathways(uv):
    """area-weighted mean (u, v) over each pathway"""
    u, v = uv
    out = {}
    for k, R in REG.items():
        s = R['sel'] & np.isfinite(u)
        w = dA[s]
        out[k] = ((u[s] * w).sum() / w.sum(), (v[s] * w).sum() / w.sum())
    return out


# ------------------------------------------------------------------ composites, pathways, lobe
CONDS = ['all'] + [x for r in ROWS for x in r[:2]]
C = {(c, ph): composite(CM[c] & PH[ph]) for c in CONDS for ph in PHASES}
P = {k: pathways(v) for k, v in C.items()}
LOBEV = {c: np.nanmean(vorticity(*C[c, 'residual'])[DISK]) / f0 for c in CONDS}
spd = lambda uv: np.hypot(*uv)
SMAX = {'tide': max(spd(P[c, ph][k]) for c in CONDS for ph in ['flood', 'ebb'] for k in REG),
        'residual': max(spd(P[c, 'residual'][k]) for c in CONDS for k in REG)}


# ------------------------------------------------------------------ drawing
WATER = ListedColormap(['#E3EEF6'])
LAND = ListedColormap(['#CFCFCF'])


def basemap(ax):
    ax.pcolormesh(X, Y, np.where(MASK, np.nan, 1.0), cmap=LAND, shading='nearest')
    ax.pcolormesh(X, Y, np.where(MASK, 1.0, np.nan), cmap=WATER, shading='nearest')
    for SN, XI_U in SECT_XI_U.items():
        jw = np.flatnonzero(MASK[:, XI_U] & MASK[:, XI_U + 1])
        xs = 0.5 * (X[0, XI_U] + X[0, XI_U + 1])
        ax.plot([xs, xs], [Y[jw.min(), 0] - 0.1, Y[jw.max(), 0] + 0.1], color=SCOL[SN], lw=1, ls='--', alpha=0.8)
    ax.set_xlim(X[0, 0] - 0.1, X[0, -1] + 0.1)
    ax.set_ylim(Y[0, 0] - 0.1, Y[-1, 0] + 0.1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def arrow(ax, x, y, uv, smax, solid, color, off=0.0):
    """one pathway arrow centred at (x, y) (shifted sideways by off km); length and width ~ speed"""
    sp = spd(uv)
    if not np.isfinite(sp) or sp < 1e-6:
        return
    ux, uy = uv[0] / sp, uv[1] / sp
    Lk = LMAX * sp / smax
    w = 0.09 * sp / smax + 0.015
    x0 = x - 0.5 * Lk * ux - off * uy
    y0 = y - 0.5 * Lk * uy + off * ux
    ax.add_patch(FancyArrow(x0, y0, Lk * ux, Lk * uy, width=w, head_width=2.6 * w, head_length=min(0.45 * Lk, 3.0 * w),
                            length_includes_head=True, fc=color if solid else 'white', ec=color, lw=1.2, zorder=5))


def lobe_arc(ax, val, r, solid, color):
    """circular arrow (270 deg) around the lobe disk, line width ~ |vorticity| / f; the head
    points counterclockwise if val > 0, clockwise if val < 0; solid = first condition, dashed = second"""
    if not np.isfinite(val) or abs(val) < 0.02:
        return
    lw = 1 + 6 * abs(val)
    t1, t2 = 100.0, 370.0                                   # degrees, counterclockwise from t1 to t2
    ax.add_patch(Arc((XL, YL), 2 * r, 2 * r, theta1=t1, theta2=t2, color=color, lw=lw,
                     ls='-' if solid else (0, (2, 1.2)), zorder=6, alpha=0.9))
    te, tb = (t2, t2 - 12) if val > 0 else (t1, t1 + 12)   # arrowhead at the leading end
    pe = (XL + r * np.cos(np.deg2rad(te)), YL + r * np.sin(np.deg2rad(te)))
    pb = (XL + r * np.cos(np.deg2rad(tb)), YL + r * np.sin(np.deg2rad(tb)))
    ax.add_patch(FancyArrowPatch(pb, pe, arrowstyle='-|>', mutation_scale=8 + 3 * lw, lw=0, color=color, zorder=7))


def panel(ax, ph, conds, title):
    basemap(ax)
    smax = SMAX['residual' if ph == 'residual' else 'tide']
    styles = [(True, 0.0)] if len(conds) == 1 else [(True, 0.11), (False, -0.11)]
    for c, (solid, off) in zip(conds, styles):
        for k, R in REG.items():
            arrow(ax, R['x'], R['y'], P[c, ph][k], smax, solid, PCOL[ph], off)
        if ph == 'residual':
            lobe_arc(ax, LOBEV[c], 0.55 if solid else 0.36, solid, '#6A3D9A')
    if ph == 'residual':
        lab = ('lobe %+.2f f' % LOBEV[conds[0]]) if len(conds) == 1 else ('lobe: solid %+.2f f, dashed %+.2f f' % (LOBEV[conds[0]], LOBEV[conds[1]]))
        ax.text(0.02, 0.98, lab, transform=ax.transAxes, ha='left', va='top', fontsize=7, color='#6A3D9A')
    ax.set_title(title, fontsize=9)


def box(ax, text, color):
    ax.axis('off')
    ax.text(0.04, 0.5, text, transform=ax.transAxes, fontsize=8, va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.8', fc=color, ec='0.35', lw=1))


# ------------------------------------------------------------------ figure
LNAME = {'depth_mean': 'depth-mean', 'upper': 'upper-layer', 'lower': 'lower-layer'}[LAYER]
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(len(ROWS) + 1, 5, width_ratios=[1.05, 1.6, 1.25, 1.25, 1.25], hspace=0.3, wspace=0.06,
                      left=0.02, right=0.98, top=0.93, bottom=0.02)

ax = fig.add_subplot(gs[0, 0])
box(ax, 'BASELINE, all hours\n%s velocity (layers split at each\ncolumn\'s median two-layer divide)\n'
        'one arrow per pathway: inner / middle /\nouter reach x north / south half;\nlength and width ~ speed;\n'
        'purple ring = lobe cell, width ~ spin\n(head shows the sense; + = counterclockwise)' % LNAME, '#EDEDED')
ua = C['all', 'residual']
ax = fig.add_subplot(gs[0, 1])
box(ax, 'rms speed: flood %.1f, ebb %.1f, residual %.1f cm s$^{-1}$\nlobe cell %+.2f f\n\n'
        'Flood enters along the north side and runs up the axis;\nebb leaves along the south side; the residual is\n'
        'in-north / out-south seaward of pc_lj plus the\ncounterclockwise lobe cell west of pc_cp.'
    % (100 * rms(*C['all', 'flood']), 100 * rms(*C['all', 'ebb']), 100 * rms(*ua), LOBEV['all']), 'white')
for j, ph in enumerate(PHASES):
    ax = fig.add_subplot(gs[0, 2 + j])
    panel(ax, ph, ['all'], '')
    ax.set_title(ph.upper(), fontsize=11, fontweight='bold', color=PCOL[ph])
    # key arrow
    ref = 5.0 if ph != 'residual' else 2.0
    smax = SMAX['residual' if ph == 'residual' else 'tide']
    kx, ky = X[0, 0] + 0.9, Y[0, 0] + 0.35
    arrow(ax, kx, ky, (ref / 100, 0), smax, True, PCOL[ph])
    ax.text(kx, ky - 0.3, '%.0f cm s$^{-1}$' % ref, ha='center', va='top', fontsize=7)

for r, (hi, lo, col, dtext, reading) in enumerate(ROWS, start=1):
    ax = fig.add_subplot(gs[r, 0])
    box(ax, dtext + '\n\nsolid = %s\nopen = %s' % (hi, lo), col)
    rh, rl = C[hi, 'residual'], C[lo, 'residual']
    nums = ('%s / %s  (%d / %d h)\nrms flood %.1f / %.1f, ebb %.1f / %.1f,\nresidual %.1f / %.1f cm s$^{-1}$\n'
            'lobe cell %+.2f / %+.2f f; residual pattern r %.2f'
            % (hi, lo, CM[hi].sum(), CM[lo].sum(), 100 * rms(*C[hi, 'flood']), 100 * rms(*C[lo, 'flood']),
               100 * rms(*C[hi, 'ebb']), 100 * rms(*C[lo, 'ebb']), 100 * rms(*rh), 100 * rms(*rl),
               LOBEV[hi], LOBEV[lo], patt(rh, rl)))
    ax = fig.add_subplot(gs[r, 1])
    box(ax, nums + ('\n\n' + reading if LAYER == 'upper' else ''), 'white')
    ax.annotate('', xy=(0.0, 0.5), xytext=(-0.16, 0.5), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='0.3', lw=1.5, mutation_scale=15))
    for j, ph in enumerate(PHASES):
        ax = fig.add_subplot(gs[r, 2 + j])
        panel(ax, ph, [hi, lo], '%s: solid %s, open %s' % (ph, hi, lo))

fig.suptitle('How the flood / ebb / residual circulation of Penn Cove (%s, plan view) changes with the tide, stratification and wind  (%s, 2024-2025)\n'
             'arrows = mean flow of each pathway (3 reaches x north / south); flood and ebb share one arrow scale, the residual has its own (key arrows, top row)'
             % (LNAME, GTAGEX), fontsize=12)
fn = out_dir / ('pc_conceptual_plan_diagram_%s.png' % LAYER)
fig.savefig(fn, dpi=170, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
for k, R in REG.items():
    print('%-6s %-5s %3d cells, all-hours residual %.1f cm/s' % (*k, R['sel'].sum(), 100 * spd(P['all', 'residual'][k])))
