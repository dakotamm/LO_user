"""
Conceptual PLAN-VIEW map: how the flood, ebb and residual circulation of Penn Cove (one
flow-defined layer) is modified by the tide (spring / neap, tropic / equatorial),
stratification and wind stress, with the model's own composites as the evidence. The plan-
view companion of 20261105_pc_conceptual_map.py (cross-sections); same layout and colors.

LAYOUT
    top row       baseline: flood | ebb | residual maps, all hours (color = along-cove u,
                  + toward the head, red; arrows = vector)
    one row each  driver box -> 'what it changes' -> the DIFFERENCE it makes to the flood,
                  ebb and residual maps (high minus low half, e.g. spring minus neap; color =
                  along-cove difference, + = more toward the head in the first condition;
                  arrows = vector difference). One difference scale on the page.
NUMBERS in the 'what it changes' boxes are computed here, high / low half:
    flood, ebb, residual = area-weighted rms speed over the cove [cm/s]
    lobe = mean residual vorticity in the 500 m lobe disk / f (+ counterclockwise), as
           20261110_pc_cove_residual_conditions.py
    r    = vector pattern correlation of the two residuals (1 = same pattern)
The one-line reading under the numbers is written for the UPPER layer only (from
20261110_pc_cove_layer_maps_plot.py -cycle and 20261110_pc_cove_residual_conditions.py);
other layers get the numbers alone.

COMPOSITES from LO_output/DM_outs/20261110_pc_cove_layer_maps/layers_hourly.nc
(transport-weighted, sum_t Ux / sum_t H); flood / ebb = sign of the hourly mouth Q;
classes of pc_tidal_cycles.py; layers split at each column's median two-layer divide.

OUTPUT LO_output/DM_outs/20261110_pc_conceptual_plan_map/pc_conceptual_plan_map_<layer>.png

run 20261110_pc_conceptual_plan_map.py [-layer upper|depth_mean|lower]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cmocean

from lo_tools import Lfun
from pc_tidal_cycles import cycle_masks

parser = argparse.ArgumentParser()
parser.add_argument('-layer', default='upper', choices=['upper', 'depth_mean', 'lower'])
LAYER = parser.parse_args().layer
L = {'depth_mean': 0, 'upper': 1, 'lower': 2}[LAYER]

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
AXIS = (-0.9789, -0.2043)                 # cove axis, mouth -> head
SECT_XI_U = {'pc_cp': 15, 'pc_lj': 23, 'pc_lp': 30}
SCOL = {'pc_cp': '#CC79A7', 'pc_lj': '#E69F00', 'pc_lp': '#0072B2'}
LOBE = (-122.708, 48.226, 500.0)          # lon, lat, radius [m]
PHASES = ['flood', 'ebb', 'residual']

ROWS = [   # (high half, low half, box color, driver box, upper-layer reading)
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
Ux, Vy, H = [np.concatenate([a.sum(axis=1, keepdims=True), a], axis=1)[:, L] for a in (Ux, Vy, H)]   # (t, cell), this layer
Qm = d.Qmouth.values
CM = cycle_masks(d.time.values)
CM['all'] = np.ones(d.time.size, bool)
PH = {'flood': Qm > 0, 'ebb': Qm < 0, 'residual': np.ones(d.time.size, bool)}
km = 111.32
DISK = np.hypot((LON[JJ, II] - LOBE[0]) * km * np.cos(np.deg2rad(LOBE[1])), (LAT[JJ, II] - LOBE[1]) * km) * 1000 <= LOBE[2]
f0 = F[JJ, II][DISK].mean()


def composite(m):
    Hs = H[m].sum(axis=0)
    with np.errstate(invalid='ignore'):
        return Ux[m].sum(axis=0) / Hs, Vy[m].sum(axis=0) / Hs


def to2d(a):
    out = np.full(LON.shape, np.nan)
    out[JJ, II] = a
    return out


def vorticity(u, v):
    """dv/dx - du/dy at the cove cells, centred differences (one-sided next to land), as 20261110_pc_cove_residual_conditions.py"""
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


al = lambda u, v: u * AXIS[0] + v * AXIS[1]

# ------------------------------------------------------------------ composites
C = {(c, ph): composite(CM[c] & PH[ph]) for c in ['all'] + [x for r in ROWS for x in r[:2]] for ph in PHASES}
vb = 100 * max(np.nanpercentile(np.abs(al(*C['all', ph])), 99) for ph in ['flood', 'ebb'])          # baseline scale (flood, ebb)
vbr = 100 * np.nanpercentile(np.abs(al(*C['all', 'residual'])), 99)
DIFF = {(hi, ph): tuple(C[hi, ph][k] - C[lo, ph][k] for k in range(2)) for hi, lo, *_ in ROWS for ph in PHASES}
vd = 100 * max(np.nanpercentile(np.abs(al(*x)), 99) for x in DIFF.values())                         # one difference scale


def basemap(ax):
    ax.pcolormesh(LON, LAT, np.where(MASK, np.nan, 1.0), cmap='Greys', vmin=0, vmax=2.5, shading='nearest')
    for SN, XI_U in SECT_XI_U.items():
        jw = np.flatnonzero(MASK[:, XI_U] & MASK[:, XI_U + 1])
        lo = 0.5 * (LON[0, XI_U] + LON[0, XI_U + 1])
        ax.plot([lo, lo], [LAT[jw.min(), 0] - 0.0009, LAT[jw.max(), 0] + 0.0009], color=SCOL[SN], lw=1)
    th = np.linspace(0, 2 * np.pi, 60)
    ax.plot(LOBE[0] + LOBE[2] / 1000 * np.cos(th) / (km * np.cos(np.deg2rad(LOBE[1]))),
            LOBE[1] + LOBE[2] / 1000 * np.sin(th) / km, color='0.3', lw=0.6, ls=':')
    ax.set_xlim(LON[0, 0], LON[0, -1])
    ax.set_ylim(LAT[0, 0], LAT[-1, 0])
    ax.set_aspect(1 / np.cos(np.deg2rad(48.23)))
    ax.set_xticks([])
    ax.set_yticks([])


def mapthumb(ax, u, v, vm, cmap):
    basemap(ax)
    cs = ax.pcolormesh(LON, LAT, to2d(100 * al(u, v)), cmap=cmap, vmin=-vm, vmax=vm, shading='nearest')
    ax.quiver(LON[JJ, II], LAT[JJ, II], u, v, scale=vm / 100 * 9, scale_units='width', width=0.005, color='k')
    return cs


def box(ax, text, color):
    ax.axis('off')
    ax.text(0.04, 0.5, text, transform=ax.transAxes, fontsize=8, va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.8', fc=color, ec='0.35', lw=1))


# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(len(ROWS) + 1, 5, width_ratios=[1.05, 1.6, 1.25, 1.25, 1.25], hspace=0.35, wspace=0.08,
                      left=0.02, right=0.93, top=0.93, bottom=0.02)
LNAME = {'depth_mean': 'depth-mean', 'upper': 'upper-layer', 'lower': 'lower-layer'}[LAYER]

# baseline row
ax = fig.add_subplot(gs[0, 0])
box(ax, 'BASELINE, all hours\n%s velocity (layers split at each\ncolumn\'s median two-layer divide)\n'
        'plan view; red = toward the head;\ndotted circle = lobe disk;\npc_cp | pc_lj | pc_lp lines' % LNAME, '#EDEDED')
ua, va = C['all', 'residual']
z = vorticity(ua, va)
ax = fig.add_subplot(gs[0, 1])
box(ax, 'rms speed: flood %.1f, ebb %.1f, residual %.1f cm s$^{-1}$\nlobe cell %+.2f f\n\n'
        'Flood enters along the north side and runs up the axis;\nebb leaves along the south side; the residual is\n'
        'in-north / out-south seaward of pc_lj plus the\ncounterclockwise lobe cell west of pc_cp.'
    % (100 * rms(*C['all', 'flood']), 100 * rms(*C['all', 'ebb']), 100 * rms(ua, va), np.nanmean(z[DISK]) / f0), 'white')
for j, ph in enumerate(PHASES):
    ax = fig.add_subplot(gs[0, 2 + j])
    cs = mapthumb(ax, *C['all', ph], vbr if ph == 'residual' else vb, 'RdBu_r')
    ax.set_title(ph.upper(), fontsize=11, fontweight='bold')
    if ph == 'ebb':
        csb = cs
    if ph == 'residual':
        csr = cs
cax = fig.add_axes([0.945, 0.80, 0.008, 0.12])
cb = fig.colorbar(csb, cax=cax)
cb.set_label('flood / ebb [cm s$^{-1}$]', fontsize=8)
cb.ax.yaxis.set_label_position('left')
cax = fig.add_axes([0.978, 0.80, 0.008, 0.12])
cb = fig.colorbar(csr, cax=cax)
cb.set_label('residual [cm s$^{-1}$]', fontsize=8)
cb.ax.yaxis.set_label_position('left')

# driver rows
for r, (hi, lo, col, dtext, reading) in enumerate(ROWS, start=1):
    ax = fig.add_subplot(gs[r, 0])
    box(ax, dtext, col)
    rh, rl = C[hi, 'residual'], C[lo, 'residual']
    zh, zl = vorticity(*rh), vorticity(*rl)
    nums = ('%s / %s  (%d / %d h)\nrms flood %.1f / %.1f, ebb %.1f / %.1f,\nresidual %.1f / %.1f cm s$^{-1}$\n'
            'lobe cell %+.2f / %+.2f f; residual pattern r %.2f'
            % (hi, lo, CM[hi].sum(), CM[lo].sum(), 100 * rms(*C[hi, 'flood']), 100 * rms(*C[lo, 'flood']),
               100 * rms(*C[hi, 'ebb']), 100 * rms(*C[lo, 'ebb']), 100 * rms(*rh), 100 * rms(*rl),
               np.nanmean(zh[DISK]) / f0, np.nanmean(zl[DISK]) / f0, patt(rh, rl)))
    ax = fig.add_subplot(gs[r, 1])
    box(ax, nums + ('\n\n' + reading if LAYER == 'upper' else ''), 'white')
    ax.annotate('', xy=(0.0, 0.5), xytext=(-0.16, 0.5), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='0.3', lw=1.5, mutation_scale=15))
    for j, ph in enumerate(PHASES):
        ax = fig.add_subplot(gs[r, 2 + j])
        csd = mapthumb(ax, *DIFF[hi, ph], vd, cmocean.cm.curl)
        ax.set_title('%s minus %s' % (hi, lo), fontsize=8)
cax = fig.add_axes([0.945, 0.10, 0.008, 0.45])
fig.colorbar(csd, cax=cax, label='difference [cm s$^{-1}$]\n+ = more toward the head\nin the first condition')
fig.suptitle('How the flood / ebb / residual circulation of Penn Cove (%s, plan view) is modified by the tide, stratification and wind  (%s, 2024-2025, hourly snapshots)'
             % (LNAME, GTAGEX), fontsize=12)
fn = out_dir / ('pc_conceptual_plan_map_%s.png' % LAYER)
fig.savefig(fn, dpi=170, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
