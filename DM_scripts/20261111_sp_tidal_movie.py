"""
Descriptive movie of the tide in upper Saratoga Passage over one lunar day, with Penn Cove as
the side branch it feeds. Hourly frames from the pc_sp_cycle box (ocean_his snapshots).

STRIP   upper Saratoga sea level (area mean over the region enclosed by the S, NE and pc_lp gates), and the transports through three gates (as
        20261111_sp_junction_tidal_cycle.py): S across south Saratoga, NE at the Skagit Bay
        entrance throat, pc_lp the Penn Cove mouth (x10 so it shows). + = toward Penn Cove.
MAPS    surface | depth-averaged | bottom-mixed-layer velocity: colour = speed (one scale for the three layers),
        arrows = direction (every 3rd cell, capped at 1.5x the colour limit). The three section
        lines are drawn white, lettered at their start; the gates in their strip colours.
SECTIONS curtains following the water (hand-picked waypoints, one cell wide) between the centres
        of the S gate, the NE gate and Penn Cove: A S -> cove, B NE -> cove, C S -> NE. Colour =
        velocity along the line (+ = toward the line's end, blue), isohalines every 0.5 g/kg,
        horizontal arrows pointing the way the water goes (every ~0.6 km, every 3rd s level).
STRIP also carries Penn Cove area-mean sea level.
BOTTOM  not the bottom s level (its centre sits only ~0.1-1.6 m above the bed, 2% of the depth) but the
        BOTTOM MIXED LAYER, per column and hour: from the bed up to the first level whose sigma0 is -dsig
        (default 0.05 kg m-3) lighter than the bottom level's; velocity = thickness-weighted mean of u, v
        (at rho points) over those levels. Baseline day, dsig 0.05: median ~6 m in upper Saratoga (14 m where
        h > 50 m), ~2 m in Penn Cove; the whole column is mixed in < 2% of column-hours.

DAY  the baseline lunar day or a one-axis variant (20261105_pc_standard_cycle), shown +-3 h.
INPUT  LO_output/extract/[gtagex]/box/pc_sp_cycle_[ds0]_[ds1].nc
OUTPUT LO_output/DM_outs/20261111_sp_tidal_movie/sp_tide_[variant].mp4 (on white) + transparent
       stills of the phase hours of the mouth (low slack, max flood, high slack, max ebb, x2)

-resid True: the same layout for the RESIDUAL, the mean over 25 hourly snapshots from the lunar day's start
       (T0 .. T0 + 24 h, a 25 h boxcar, the same length for every variant although the lunar days are 24-26 h
       long; a full Godin filter needs ~71 h and the box holds 97 h).
       Maps: surface and bottom = plain means on the top / bottom s level; depth-averaged = FLUX-weighted
       <ubar H> / <H> (H = h + zeta), so the tidal Stokes transport is included. Sections: plain means on the
       s levels, drawn at their mean depths, with the mean salinity. The strip keeps the hourly series, shades
       the averaging window and draws each gate's mean Q. One figure, sp_tide_resid_[variant].png.

run 20261111_sp_tidal_movie.py [-variant baseline|spring|neap|...] [-fps 3] [-resid True]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import animation
import numpy as np
import pandas as pd
import xarray as xr
import cmocean
from scipy import ndimage
import gsw

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-variant', default='baseline')
parser.add_argument('-fps', default=3, type=float)
parser.add_argument('-dsig', default=0.05, type=float)   # bottom mixed layer threshold [kg m-3]
parser.add_argument('-resid', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'])   # lunar-day mean, one figure
args = parser.parse_args()
VARIANT = args.variant.replace('_', ' ')

GTAGEX = 'wb1_t0_xn11abbur00'
COVE_CELL = (65, 20)                      # (j, i) inside Penn Cove: seeds the Whidbey water mask
SP_CELL = (40, 70)                        # (j, i) in Saratoga Passage: seeds the junction
GATES = {                                 # as 20261111_sp_junction_tidal_cycle.py; + = toward Penn Cove
    'S':     dict(kind='v', idx=0,  span=np.arange(0, 109), sign=+1),
    'NE':    dict(kind='u', idx=84, span=np.arange(76, 97), sign=-1),
    'pc_lp': dict(kind='u', idx=32, span=np.arange(60, 72), sign=-1),
}
NAME = {'S': 'S: south Saratoga', 'NE': 'NE: Skagit Bay entrance', 'pc_lp': 'pc_lp: Penn Cove mouth (x10)'}
COL = {'S': '#0072B2', 'NE': '#009E73', 'pc_lp': '#D55E00'}
PAD = pd.Timedelta('3h')

Ldir = Lfun.Lstart(gridname='wb1')
sel_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_standard_cycle'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261111_sp_tidal_movie'
out_dir.mkdir(parents=True, exist_ok=True)

if VARIANT == 'baseline':
    day = pd.read_csv(sel_dir / 'lunar_days.csv', parse_dates=['t0', 't1']).iloc[0]
else:
    V = pd.read_csv(sel_dir / 'variants.csv', parse_dates=['t0', 't1'])
    day = V[V.variant == VARIANT].iloc[0]
T0, T1 = day.t0, day.t1
ds0 = (T0.normalize() - pd.Timedelta('1D')).strftime('%Y.%m.%d')
ds1 = (T1.normalize() + pd.Timedelta('1D')).strftime('%Y.%m.%d')
in_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_sp_cycle_%s_%s.nc' % (ds0, ds1))
print('%s: lunar day %s to %s UTC, box %s' % (VARIANT, T0, T1, in_fn.name))

ds = xr.open_dataset(in_fn)
TA = pd.DatetimeIndex(ds.ocean_time.values)
it = np.flatnonzero((TA >= T0 - PAD) & (TA <= T1 + PAD))
T = TA[it]
sl = dict(ocean_time=slice(it[0], it[-1] + 1))
lon, lat = ds.lon_rho.values, ds.lat_rho.values
pm, pn, h = ds.pm.values, ds.pn.values, ds.h.values
mask = ds.mask_rho.values == 1
D = {vn: ds[vn].isel(**sl).to_numpy().astype(float) for vn in ['u', 'v', 'ubar', 'vbar', 'salt', 'temp', 'z_w', 'z_rho', 'zeta']}
ds.close()

# ------------------------------------------------------------------ Whidbey water, junction, gate transports
lab, _ = ndimage.label(mask)
wet = lab == lab[COVE_CELL]
cut = wet.copy()
cut[0, :] = False
cut[GATES['NE']['span'], GATES['NE']['idx'] + 1] = False
cut[GATES['pc_lp']['span'], GATES['pc_lp']['idx']] = False
lab2, _ = ndimage.label(cut)
junc = lab2 == lab2[SP_CELL]
area = 1 / (pm * pn)
zeta_j = (D['zeta'][:, junc] * area[junc]).sum(axis=1) / area[junc].sum()   # junction area-mean sea level
cc = wet.copy()
cc[GATES['pc_lp']['span'], GATES['pc_lp']['idx'] + 1] = False      # Saratoga side of the mouth faces
lab3, _ = ndimage.label(cc)
cove = lab3 == lab3[COVE_CELL]
zeta_pc = (D['zeta'][:, cove] * area[cove]).sum(axis=1) / area[cove].sum()  # Penn Cove area-mean sea level
print('Penn Cove: %d cells; sea level minus junction %+.3f to %+.3f m' % (cove.sum(), (zeta_pc - zeta_j).min(), (zeta_pc - zeta_j).max()))

for G in GATES.values():
    a, b = G['idx'], G['idx'] + 1
    p = np.array([p for p in G['span'] if (wet[a, p] and wet[b, p] if G['kind'] == 'v' else wet[p, a] and wet[p, b])])
    if G['kind'] == 'v':
        vel, zw, dl = D['v'][:, :, a, p], 0.5 * (D['z_w'][:, :, a, p] + D['z_w'][:, :, b, p]), 0.5 * (1 / pm[a, p] + 1 / pm[b, p])
    else:
        vel, zw, dl = D['u'][:, :, p, a], 0.5 * (D['z_w'][:, :, p, a] + D['z_w'][:, :, p, b]), 0.5 * (1 / pn[p, a] + 1 / pn[p, b])
    G['Q'] = G['sign'] * (vel * np.diff(zw, axis=1) * dl[None, None, :]).sum(axis=(1, 2))
for g, G in GATES.items():
    print('  %-6s Q %+7.0f to %+7.0f m3/s' % (g, G['Q'].min(), G['Q'].max()))

# ------------------------------------------------------------------ velocities at rho points
def to_rho(uu, vv):
    """u (..., eta, xi_u), v (..., eta_v, xi) -> rho points, land faces zeroed first (no normal flow at a
    wall); the outermost row / column of the box, which lacks a face on one side, is NaN"""
    uu, vv = np.nan_to_num(uu), np.nan_to_num(vv)
    ur = np.full(uu.shape[:-1] + (uu.shape[-1] + 1,), np.nan)
    vr = np.full(vv.shape[:-2] + (vv.shape[-2] + 1, vv.shape[-1]), np.nan)
    ur[..., 1:-1] = 0.5 * (uu[..., :-1] + uu[..., 1:])
    vr[..., 1:-1, :] = 0.5 * (vv[..., :-1, :] + vv[..., 1:, :])
    return ur, vr

UV = {'surface': to_rho(D['u'][:, -1], D['v'][:, -1]),
      'depth-averaged': to_rho(D['ubar'], D['vbar']),
      'bottom': to_rho(D['u'][:, 0], D['v'][:, 0])}
SPEED = {k: np.hypot(u_, v_) for k, (u_, v_) in UV.items()}
show = wet & np.isfinite(UV['depth-averaged'][0][0])               # the outer ring of the box has no rho velocity
for k, A in SPEED.items():
    print('  %-15s speed 50/99 pct %.2f / %.2f m/s' % ((k,) + tuple(np.nanpercentile(A[:, show], [50, 99]))))

# ------------------------------------------------------------------ curtain sections between the gate centres and the cove centre
# Centres: the S and NE gates at the depth-weighted centroid of their faces (S i 84.2; NE j 85.8), first rho cell
# inside the junction; Penn Cove at the area centroid of the cove water (i 17.8, j 61.1). A straight line between
# them crosses Whidbey, so each section follows (i, j) waypoints picked on the box bathymetry: up the deep channel
# of Saratoga (the deepest cell per row jitters by up to 5 columns), across the junction to the mouth centre
# (pc_lp rows 60-71 -> j 65.5), then along the cove (midway between the shores).
C_S, C_NE, C_PC = (84, 1), (84, 86), (18, 61)
SAR = [(79, 12), (73, 25), (71, 40)]                                 # Saratoga deep channel, south to north
TO_PC = [(55, 66), (40, 65.5), (33, 65.5), (30, 63.5), (24, 61)]    # junction off the mouth -> mouth -> along the cove
SECTS = {
    'A': dict(way=[C_S] + SAR + [(69, 58)] + TO_PC + [C_PC], name='S gate centre -> Penn Cove centre', plus='toward Penn Cove'),
    'B': dict(way=[C_NE, (76, 78), (68, 70)] + TO_PC + [C_PC], name='NE gate centre -> Penn Cove centre', plus='toward Penn Cove'),
    'C': dict(way=[C_S] + SAR + [(71, 70), (76, 78), C_NE], name='S gate centre -> NE gate centre', plus='toward the NE gate'),
}
DXI, DYJ = np.mean(1 / pm[wet]), np.mean(1 / pn[wet])                # cell size [m] (the box grid is near uniform)
u3, v3 = to_rho(D['u'], D['v'])


def section(way):
    """distinct cells along the waypoint polyline (sampled every quarter cell), with the direction of the
    segment each cell is on, and the fields along it; velocity + = along the line toward its end"""
    W = np.array(way, dtype=float)
    P, DIR = [], []
    for p0, p1 in zip(W[:-1], W[1:]):
        n = int(4 * np.hypot(*(p1 - p0))) + 1
        d = (p1 - p0) * (DXI, DYJ)
        P.append(np.linspace(p0, p1, n))
        DIR.append(np.repeat((d / np.hypot(*d))[None], n, axis=0))
    cells, dirs = [], []
    for (i, j), d_ in zip(np.rint(np.concatenate(P)).astype(int), np.concatenate(DIR)):
        if cells and cells[-1] == (i, j):
            continue
        assert wet[j, i], 'section crosses land at (i %d, j %d)' % (i, j)
        cells.append((i, j)); dirs.append(d_)
    CI, CJ = np.array([c[0] for c in cells]), np.array([c[1] for c in cells])
    dirs = np.array(dirs)
    xs = np.concatenate([[0], np.cumsum(np.hypot(np.diff(CI) * DXI, np.diff(CJ) * DYJ))]) / 1000   # km from the start
    xe = np.concatenate([[xs[0] - 0.5 * (xs[1] - xs[0])], 0.5 * (xs[1:] + xs[:-1]), [xs[-1] + 0.5 * (xs[-1] - xs[-2])]])
    return dict(CI=CI, CJ=CJ, xs=xs, xe=xe,
                U=u3[:, :, CJ, CI] * dirs[:, 0] + v3[:, :, CJ, CI] * dirs[:, 1],   # (t, k, n)
                S=D['salt'][:, :, CJ, CI], ZW=D['z_w'][:, :, CJ, CI], ZR=D['z_rho'][:, :, CJ, CI])


XM = GATES['pc_lp']['idx'] + 0.5                                     # the mouth faces sit between rho columns 32 | 33
for k_, X in SECTS.items():
    X.update(section(X['way']))
    if (X['CI'] <= XM).any():                                       # passes through the mouth
        X['marks'] = [(X['xs'][np.argmax(X['CI'] <= XM)], 'Penn Cove mouth')]
    else:                                                           # passes the mouth: nearest cell to its centre row
        X['marks'] = [(X['xs'][np.argmin(np.hypot(X['CI'] - 71, X['CJ'] - 65.5))], 'off the Penn Cove mouth (to the W)')]
    print('section %s (%s): %d cells, %.1f km, max h %.0f m; %s at %.1f km'
          % (k_, X['name'], len(X['CI']), X['xs'][-1], h[X['CJ'], X['CI']].max(), X['marks'][0][1], X['marks'][0][0]))

# ------------------------------------------------------------------ mouth phases for the stills (as 20261105_pc_tidal_day_schematic.py)
q = GATES['pc_lp']['Q']
cross = [(n, 'up' if q[n + 1] > 0 else 'down') for n in range(len(q) - 1) if np.sign(q[n]) != np.sign(q[n + 1])]
slack = [(n if abs(q[n]) <= abs(q[n + 1]) else n + 1, kind) for n, kind in cross]
n0 = [k for k, (n, kind) in enumerate(slack) if kind == 'up'][0]
slack = slack[n0:n0 + 5]
PH = {}
if len(slack) == 5:
    for c in range(2):
        a, b, e = slack[2 * c][0], slack[2 * c + 1][0], slack[2 * c + 2][0]
        PH.update({a: 'low slack %d' % (c + 1), a + np.argmax(q[a:b]): 'max flood %d' % (c + 1),
                   b: 'high slack %d' % (c + 1), b + np.argmin(q[b:e]): 'max ebb %d' % (c + 1)})
else:
    print('WARNING: mouth phases not found, no stills')

# ------------------------------------------------------------------ bottom mixed layer: replaces the bottom s level on the maps
SA = gsw.SA_from_SP(D['salt'], gsw.p_from_z(D['z_rho'], lat[None, None]), lon[None, None], lat[None, None])
sig = gsw.sigma0(SA, gsw.CT_from_pt(SA, D['temp']))
lighter = (sig[:, :1] - sig) > args.dsig                             # lighter than the bottom level by more than dsig
nk = sig.shape[1]
kf = np.where(lighter.any(axis=1), np.argmax(lighter, axis=1), nk)   # first such level from the bed; levels k < kf are the BML
wgt = np.where(np.arange(nk)[None, :, None, None] < kf[:, None], np.diff(D['z_w'], axis=1), 0)
with np.errstate(invalid='ignore', divide='ignore'):
    UV['bottom'] = tuple(np.nansum(x * wgt, axis=1) / wgt.sum(axis=1) for x in (u3, v3))
SPEED['bottom'] = np.hypot(*UV['bottom'])
BML = np.take_along_axis(D['z_w'], kf[:, None], axis=1)[:, 0] + h[None]   # thickness [m]
for name, R in [('upper Saratoga', junc & show), ('Penn Cove', cove & show)]:
    print('bottom mixed layer (dsig %.2f), %s: median %.1f m (10-90%% %.1f-%.1f), whole column mixed %.1f%% of column-hours'
          % (args.dsig, name, np.median(BML[:, R]), *np.percentile(BML[:, R], [10, 90]), 100 * np.mean(kf[:, R] == nk)))
TITLE = {'surface': 'surface (top s level)', 'depth-averaged': 'depth-averaged',
         'bottom': 'bottom mixed layer (%.2f kg m$^{-3}$ from the bed)' % args.dsig}

# ------------------------------------------------------------------ residual: lunar-day means replace the hourly fields
iday = np.flatnonzero(T >= T0)[:25]                                 # 25 hourly snapshots from the lunar day's start (25 h boxcar;
if args.resid:                                                      # the lunar days are 24-26 h long, set by the mouth-Q crossings)
    assert len(iday) == 25 and (T[iday[-1]] - T[iday[0]]) == pd.Timedelta('24h'), 'need 25 consecutive hourly snapshots from T0'
    Hr = h[None] + D['zeta'][iday]
    ub_, vb_ = UV['depth-averaged']
    UV = {'surface': tuple(np.nanmean(x[iday], axis=0)[None] for x in UV['surface']),
          'depth-averaged': tuple((np.nanmean(x[iday] * Hr, axis=0) / Hr.mean(axis=0))[None] for x in (ub_, vb_)),
          'bottom': tuple(np.nanmean(x[iday], axis=0)[None] for x in UV['bottom'])}   # mean of the hourly BML means
    SPEED = {k: np.hypot(u_, v_) for k, (u_, v_) in UV.items()}
    for X in SECTS.values():
        for f in ['U', 'S', 'ZW', 'ZR']:
            X[f] = X[f][iday].mean(axis=0)[None]
    QBAR = {g: G['Q'][iday].mean() for g, G in GATES.items()}
    for k, A in SPEED.items():
        print('  residual %-15s speed 50/99 pct %.3f / %.3f m/s' % ((k,) + tuple(np.nanpercentile(A[0, show], [50, 99]))))
    dV = ((h[None] + D['zeta'][[iday[0], iday[-1] + 1]]) * area[None])[:, junc].sum(axis=1)
    print('  residual Q: ' + ', '.join('%s %+.0f' % (g, q_) for g, q_ in QBAR.items()) + ' m3/s; '
          'S + NE - pc_lp = %+.0f m3/s vs lunar-day volume change / 25 h = %+.0f m3/s'
          % (QBAR['S'] + QBAR['NE'] - QBAR['pc_lp'], (dV[1] - dV[0]) / (25 * 3600)))

# ------------------------------------------------------------------ figure
LAYERS = ['surface', 'depth-averaged', 'bottom']
VMAX = np.nanpercentile(np.concatenate([SPEED[k][:, show] for k in LAYERS], axis=1), 99)   # one speed scale, all layers
UMAX = np.nanpercentile(np.abs(np.concatenate([X['U'].ravel() for X in SECTS.values()])), 99)   # one scale, all sections
QSTEP, QSCALE = 3, 4 * VMAX                                         # map arrows every 3rd cell; VMAX = a quarter of the panel width
ACAP = 1.5                                                          # map arrows capped at 1.5 x VMAX (the Skagit Bay entrance runs ~1 m/s)
m_ = np.zeros_like(show)
m_[::QSTEP, ::QSTEP] = True
m_ &= show
SDX, SDK, SLEN, SMIN = 0.6, 3, 0.04, 0.1   # section arrows: every ~0.6 km, every 3rd s level; UMAX = 4% of the width; none < 10% of UMAX
ASP = 1 / np.cos(np.deg2rad(np.mean(lat)))
# gate faces as lines on the maps (face centres)
GL = {}
for g, G in GATES.items():
    a, b = G['idx'], G['idx'] + 1
    p = np.array([p for p in G['span'] if (wet[a, p] and wet[b, p] if G['kind'] == 'v' else wet[p, a] and wet[p, b])])
    GL[g] = ((0.5 * (lon[a, p] + lon[b, p]), 0.5 * (lat[a, p] + lat[b, p])) if G['kind'] == 'v'
             else (0.5 * (lon[p, a] + lon[p, b]), 0.5 * (lat[p, a] + lat[p, b])))
# section mesh: column edges halfway between neighbouring columns (ends: the column's own z_w)
def edges(Z):
    return np.concatenate([Z[..., :1], 0.5 * (Z[..., 1:] + Z[..., :-1]), Z[..., -1:]], axis=-1)
for X in SECTS.values():
    X['xs2'] = np.repeat(X['xs'][None], X['ZR'].shape[1], axis=0)
    X['sx'] = np.repeat(X['xe'][None], X['ZW'].shape[1], axis=0)
    X['acol'] = np.unique(np.searchsorted(X['xs'], np.arange(0.3, X['xs'][-1], SDX)))   # arrow columns

STARTS = {}
for k_, X in SECTS.items():
    STARTS.setdefault((X['CJ'][0], X['CI'][0]), []).append(k_)

fig = plt.figure(figsize=(16, 18))
gs = fig.add_gridspec(5, 3, height_ratios=[0.9, 2.8, 1.1, 1.1, 1.1], hspace=0.3, wspace=0.08, left=0.06, right=0.92, top=0.94, bottom=0.04)
ax_s = fig.add_subplot(gs[0, :])
ax_m = [fig.add_subplot(gs[1, c]) for c in range(3)]
ax_x = {k_: fig.add_subplot(gs[2 + r, :]) for r, k_ in enumerate(SECTS)}
cax_m = fig.add_axes([0.93, 0.56, 0.01, 0.22])
cax_x = fig.add_axes([0.93, 0.08, 0.01, 0.3])

# strip: drawn once, only the time marker moves
ax_s.axvspan(T0, T1, color='0.92', zorder=0)
ax_s.axhline(0, color='k', lw=0.6)
for g, G in GATES.items():
    ax_s.plot(T, G['Q'] * (10 if g == 'pc_lp' else 1) / 1e3, color=COL[g], lw=1.8,
              label=NAME[g] + (', lunar-day mean %+.0f m$^3$ s$^{-1}$' % QBAR[g] if args.resid else ''))
ax_s.set_ylabel('Q [10$^3$ m$^3$ s$^{-1}$]\n+ into the junction (S, NE)\n+ into the cove (pc_lp)', fontsize=8)
ax_s.grid(lw=0.3)
ax_z = ax_s.twinx()
ax_z.plot(T, zeta_j, color='0.4', lw=1.2, ls='--', label='upper Saratoga sea level (mean between the gates)')
ax_z.plot(T, zeta_pc, color=COL['pc_lp'], lw=1.2, ls=':', label='Penn Cove sea level')
ax_z.set_ylabel('sea level [m]', fontsize=9, color='0.4')
h1, l1 = ax_s.get_legend_handles_labels()
h2, l2 = ax_z.get_legend_handles_labels()
ax_s.legend(h1 + h2, l1 + l2, fontsize=7 if args.resid else 8, ncol=3 if args.resid else 5, loc='upper left')
ax_s.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
ax_s.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
ax_s.set_xlim(T[0], T[-1])
tline = ax_s.axvline(T[0], color='k', lw=2, visible=not args.resid)
fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(0, VMAX), cmocean.cm.speed), cax=cax_m,
             extend='max').set_label('%sspeed [m s$^{-1}$]; arrows = direction' % ('residual ' if args.resid else ''), fontsize=9)
fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(-UMAX, UMAX), cmocean.cm.balance_r), cax=cax_x,
             extend='both').set_label('velocity along the line [m s$^{-1}$]\nblue = + (see titles), red = -', fontsize=9)


def draw(n):
    tline.set_xdata([T[n], T[n]])
    for ax, k in zip(ax_m, LAYERS):
        ax.clear()
        ax.set_facecolor('0.85')
        ax.pcolormesh(lon, lat, np.where(show, SPEED[k][n], np.nan), cmap=cmocean.cm.speed, vmin=0, vmax=VMAX, shading='nearest')
        u_, v_ = UV[k][0][n][m_], UV[k][1][n][m_]
        sp = np.hypot(u_, v_)
        cap = np.minimum(1, ACAP * VMAX / np.maximum(sp, 1e-9))       # longest arrow = ACAP x the colour limit
        qv = ax.quiver(lon[m_], lat[m_], u_ * cap, v_ * cap, scale=QSCALE, scale_units='width', width=0.003, color='k')
        for X in SECTS.values():
            ax.plot(lon[X['CJ'], X['CI']], lat[X['CJ'], X['CI']], color='w', lw=1.2, ls=':')
        for (j_, i_), L in STARTS.items():                            # lines sharing a start get one label ('A,C')
            ax.text(lon[j_, i_], lat[j_, i_], ','.join(L), color='w', fontsize=8, fontweight='bold',
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='k', ec='none'))
        for g in GATES:
            ax.plot(*GL[g], color=COL[g], lw=2.5)
        ax.set_aspect(ASP)
        ax.set_xlim(lon[show].min() - 0.005, lon[show].max() + 0.005)
        ax.set_ylim(lat[show].min() - 0.003, lat[show].max() + 0.003)
        ax.set_title(TITLE[k], fontsize=10)
        ax.tick_params(labelsize=7)
        if k != 'surface':
            ax.set_yticklabels([])
        if k == 'surface':
            ax.quiverkey(qv, 0.2, 0.93, float('%.1g' % VMAX), '%.1g m s$^{-1}$' % VMAX, labelpos='N', fontproperties={'size': 7})
    for k_, X in SECTS.items():
        ax = ax_x[k_]
        ax.clear()
        ax.set_facecolor('0.85')
        U = X['U'][n]
        ax.pcolormesh(X['sx'], edges(X['ZW'][n]), U, cmap=cmocean.cm.balance_r, vmin=-UMAX, vmax=UMAX, shading='flat')
        cs = ax.contour(X['xs2'], X['ZR'][n], X['S'][n], np.arange(20, 32, 0.5), colors='k', linewidths=0.5)
        ax.clabel(cs, cs.levels[::2], fontsize=6, fmt='%.1f')
        # direction arrows: horizontal, pointing the way the water moves along the line, length in proportion to |u|
        kk = np.arange(1, U.shape[0], SDK)
        Ua = U[np.ix_(kk, X['acol'])]
        Ua = np.where(np.abs(Ua) >= SMIN * UMAX, np.clip(Ua, -UMAX, UMAX), np.nan)
        ax.quiver(X['xs2'][np.ix_(kk, X['acol'])], X['ZR'][n][np.ix_(kk, X['acol'])], Ua, np.zeros_like(Ua),
                  scale=UMAX / SLEN, scale_units='width', width=0.0016, headwidth=4, color='k', pivot='mid')
        for x_, t_ in X['marks']:
            ax.axvline(x_, color='k', lw=0.8, ls='--')
            ax.text(x_, 1.01, t_, fontsize=8, va='bottom', ha='center', transform=ax.get_xaxis_transform())
        ax.set_xlim(X['xe'][0], X['xe'][-1])
        ax.set_ylim(-h[X['CJ'], X['CI']].max() - 2, 3)
        ax.set_title('%s: %s (+ = %s)' % (k_, X['name'], X['plus']), fontsize=9, loc='left')
        ax.set_ylabel('z [m]', fontsize=9)
        ax.tick_params(labelsize=8)
    ax_x['C'].set_xlabel('km along the line from its start (lettered label on the maps)', fontsize=9)
    if args.resid:
        fig.suptitle('Upper Saratoga Passage, %s: RESIDUAL = mean of the 25 hourly snapshots %s to %s UTC (lunar day grey in the strip)\n'
                     'depth-averaged map flux-weighted <ubar H>/<H>; mean isohalines every 0.5 g kg$^{-1}$; map arrows capped at %.3f m s$^{-1}$'
                     % (VARIANT, T[iday[0]].strftime('%Y-%m-%d %H:%M'), T[iday[-1]].strftime('%m-%d %H:%M'), ACAP * VMAX), fontsize=11)
        return
    lbl = ' | %s' % PH[n] if n in PH else ''
    fig.suptitle('Upper Saratoga Passage, %s lunar day (grey in the strip): %s UTC%s\nisohalines every 0.5 g kg$^{-1}$; map arrows capped at %.2f m s$^{-1}$'
                 % (VARIANT, T[n].strftime('%Y-%m-%d %H:%M'), lbl, ACAP * VMAX), fontsize=11)


tag = VARIANT.replace(' ', '_')
if args.resid:
    draw(0)
    fig.savefig(out_dir / ('sp_tide_resid_%s.png' % tag), dpi=120, transparent=True)
    plt.close(fig)
    print('saved', out_dir / ('sp_tide_resid_%s.png' % tag))
    raise SystemExit
for n, l in PH.items():
    draw(n)
    fig.savefig(out_dir / ('sp_tide_%s_%s.png' % (tag, l.replace(' ', '_'))), dpi=120, transparent=True)
EVEN = ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']
anim = animation.FuncAnimation(fig, draw, frames=len(T), interval=1000 / args.fps, blit=False)
out = out_dir / ('sp_tide_%s.mp4' % tag)
anim.save(out, writer=animation.FFMpegWriter(fps=args.fps, extra_args=EVEN + ['-crf', '18', '-pix_fmt', 'yuv420p']),
          savefig_kwargs=dict(facecolor='white'), dpi=120)
plt.close(fig)
print('saved %s (%d frames at %g fps) + %d stills' % (out, len(T), args.fps, len(PH)))
