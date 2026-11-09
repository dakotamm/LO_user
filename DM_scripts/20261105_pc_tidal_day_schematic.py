"""
The evolution of one lunar day (flood, ebb, flood, ebb) in Penn Cove and upper Saratoga
Passage, as phase snapshots: the baseline day (calm wind, median stratification, median
spring/neap and tropic/equatorial) or one of the one-axis variants from
20261105_pc_standard_cycle.py.

PHASES from the MODEL transport through the mouth, Q(t) = int u dA at pc_lp (hourly
    snapshots, + into the cove), within the lunar day: low slack (Q crosses 0 upward),
    max flood (max Q), high slack (Q crosses 0 downward), max ebb (min Q), twice. A slack
    is the hour with the smaller |Q| of the two either side of the crossing.
ROWS (one column per phase)
    surface   salinity of the top cell, with the top-cell velocity (u, v averaged to rho)
    bottom    salinity of the bottom cell, with the bottom-cell velocity
    section   velocity ALONG the line (+ = toward the cove head; the line's local direction,
              so in the cove ~westward, on the NE leg ~southwestward) along a line from
              the cove head through the mouth, then NE along upper Saratoga Passage to the
              Skagit Bay entrance, with isohalines. In the cove the line is the cell(s)
              midway between the shores per column (as 20261103_pc_alongchannel_salt.py);
              from the mouth it follows hand-picked waypoints (NE_WAY), one cell wide.
    plus a strip on top: Q(t) and the mouth sea level over the day, phases marked.
Only Whidbey Basin water is drawn: the wet cells connected to Penn Cove inside the box
(the SW corner of the box holds Admiralty Inlet water, a different basin).

INPUT  LO_output/extract/[gtagex]/box/pc_sp_cycle_[ds0]_[ds1].nc (hourly ocean_his),
       the box from the day before the lunar day to the day after;
       LO_output/DM_outs/20261105_pc_standard_cycle/lunar_days.csv, variants.csv (the days)
-zoom True: maps of the cove and ~3.5 km of Saratoga only (ZOOM_EXT), arrows every 2nd cell, the
section cut 3 km past the mouth, isohalines every 0.2 g/kg; colour limits and arrow scale from
what is in view.

OUTPUT LO_output/DM_outs/20261105_pc_tidal_day_schematic/tidal_day_[variant][_zoom].png

run 20261105_pc_tidal_day_schematic.py [-variant baseline|spring|neap|tropic|equatorial|
    high_strat|low_strat|up-cove_wind|down-cove_wind] [-zoom True]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cmocean
from scipy import ndimage

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-variant', default='baseline')
parser.add_argument('-zoom', default=False, type=lambda x: x.lower() in ['true', '1', 'yes'])   # Penn Cove and a bit of Saratoga only
args = parser.parse_args()
VARIANT = args.variant.replace('_', ' ')

GTAGEX = 'wb1_t0_xn11abbur00'
SIGN = -1                                 # section-positive (east) is out of the cove
MOUTH = dict(XI_U=32, ETA=np.arange(71, 59, -1))   # pc_lp u faces, north to south (pc_cove box + (2, 52))
COVE = dict(I=np.arange(2, 33), J=np.arange(52, 73))  # pc_cove box footprint, rho columns west of the mouth face
COVE_CELL = (65, 20)                      # (j, i), a cell inside the cove, seeds the Whidbey water mask

Ldir = Lfun.Lstart(gridname='wb1')
sel_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_standard_cycle'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_tidal_day_schematic'
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
T = pd.DatetimeIndex(ds.ocean_time.values)
lon, lat = ds.lon_rho.values, ds.lat_rho.values
pm, pn, h = ds.pm.values, ds.pn.values, ds.h.values

# ------------------------------------------------------------------ Whidbey water: wet cells connected to the cove
lab, _ = ndimage.label(ds.mask_rho.values == 1)
assert lab[COVE_CELL] > 0, 'COVE_CELL is land'
wet = lab == lab[COVE_CELL]
print('Whidbey water: %d of %d wet cells in the box' % (wet.sum(), (lab > 0).sum()))

# ------------------------------------------------------------------ section line: cove centre, then NE from the mouth to the Skagit
XI, JC = [], []
for i in COVE['I']:                       # cove: the cell(s) midway between the shores, per column
    jw = COVE['J'][wet[COVE['J'], i]]
    if len(jw) == 0:
        continue
    mid = 0.5 * (jw.min() + jw.max())
    XI.append(i)
    JC.append(np.unique([int(np.floor(mid)), int(np.ceil(mid))]))
# Saratoga leg: a polyline through these (i, j) waypoints, from the mouth centre NE along the channel
# to the Skagit Bay entrance at the NE corner of the box (picked by eye on the box mask), sampled
# every quarter cell and kept as the sequence of distinct cells it passes through
NE_WAY = np.array([(32, 65.5), (45, 71), (60, 77), (72, 81), (82, 85), (90, 93), (97, 101), (104, 108), (108, 111)])
DXI, DYJ = np.mean(1 / pm[wet]), np.mean(1 / pn[wet])                # cell size [m] (the box grid is plaid, near uniform)
seg = [np.linspace(p0, p1, int(4 * np.hypot(*(p1 - p0))) + 1) for p0, p1 in zip(NE_WAY[:-1], NE_WAY[1:])]
P = np.concatenate(seg)
arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(P[:, 0]) * DXI, np.diff(P[:, 1]) * DYJ))])   # [m] from the mouth centre
dP = np.concatenate([[(p1 - p0) * (DXI, DYJ)] * (int(4 * np.hypot(*(p1 - p0))) + 1) for p0, p1 in zip(NE_WAY[:-1], NE_WAY[1:])])
dP /= np.hypot(dP[:, 0], dP[:, 1])[:, None]                          # direction of the segment each sample is on
n_cove = len(XI)
S_NE, D_NE = [], []
for (i, j), a_, d_ in zip(np.rint(P).astype(int), arc, dP):
    if i <= COVE['I'][-1] or (XI[-1] == i and JC[-1][0] == j):
        continue
    assert wet[j, i], 'NE leg crosses land at (i %d, j %d)' % (i, j)
    XI.append(i)
    JC.append(np.array([j]))
    S_NE.append(a_)                       # arclength where the line enters the cell
    D_NE.append(d_)
XI = np.array(XI)
# distance from the head: 1/pm per column in the cove (as 20261103_pc_alongchannel_salt.py), arclength of the
# polyline on the NE leg (counting cell-to-cell steps would overcount its staircase); edges halfway between centres
dxc = 1 / pm[JC[0][0], XI[:n_cove]]
xc_cove = np.cumsum(dxc) - 0.5 * dxc                                  # cove column centres [m]
xc = np.concatenate([xc_cove, xc_cove[-1] + np.array(S_NE) + 0.5 * DXI]) / 1000   # NE cells: centre half a cell past entry
xe = np.concatenate([[0], 0.5 * (xc[1:] + xc[:-1]), [xc[-1] + 0.5 * (xc[-1] - xc[-2])]])   # km from the head
xe[n_cove] = np.cumsum(dxc)[-1] / 1000    # the mouth face sits exactly at the end of the cove columns
lon_line = np.array([lon[j, i].mean() for i, j in zip(XI, JC)])
lat_line = np.array([lat[j, i].mean() for i, j in zip(XI, JC)])
# direction of the line, away from the head: the cove axis in the cove (mouth -> head is (-0.9789, -0.2043),
# reduce_wind_cove.py), the polyline segment on the NE leg (cell-to-cell steps would jitter on the staircase)
TX = np.concatenate([np.full(n_cove, 0.9789), [d[0] for d in D_NE]])
TY = np.concatenate([np.full(n_cove, 0.2043), [d[1] for d in D_NE]])
x_mouth = xe[np.flatnonzero(XI == MOUTH['XI_U'])[0] + 1]            # the pc_lp face is the east edge of its rho column
print('section: %d columns, %.1f km, mouth at %.1f km, max depth %.0f m'
      % (len(XI), xe[-1], x_mouth, max(h[j, i].mean() for i, j in zip(XI, JC))))

# ------------------------------------------------------------------ transport through the mouth and the phases
ETA, XU = MOUTH['ETA'], MOUTH['XI_U']
u = SIGN * ds.u.isel(xi_u=XU, eta_u=ETA).to_numpy().astype(float)                       # (t, k, p) + into the cove
zwm = ds.z_w.isel(eta_rho=ETA, xi_rho=[XU, XU + 1]).to_numpy().mean(axis=-1)
dym = (1 / pn[ETA][:, [XU, XU + 1]]).mean(axis=-1)
Q = (u * np.diff(zwm, axis=1) * dym[None, None, :]).sum(axis=(1, 2))
zeta_m = ds.zeta.isel(eta_rho=ETA, xi_rho=XU).to_numpy().mean(axis=1)                   # mouth sea level

it = np.flatnonzero((T >= T0 - pd.Timedelta('3h')) & (T <= T1 + pd.Timedelta('3h')))
q = Q[it]
cross = [(n, 'up' if q[n + 1] > 0 else 'down') for n in range(len(q) - 1) if np.sign(q[n]) != np.sign(q[n + 1])]
slack = [(n if abs(q[n]) <= abs(q[n + 1]) else n + 1, kind) for n, kind in cross]
n0 = [k for k, (n, kind) in enumerate(slack) if kind == 'up'][0]
slack = slack[n0:n0 + 5]                  # up, down, up, down, up: bounds the two floods and two ebbs
assert len(slack) == 5 and [s[1] for s in slack] == ['up', 'down', 'up', 'down', 'up'], slack
PH, LAB = [], []
for c in range(2):
    a, b, e = slack[2 * c][0], slack[2 * c + 1][0], slack[2 * c + 2][0]
    PH += [a, a + np.argmax(q[a:b]), b, b + np.argmin(q[b:e])]
    LAB += ['low slack %d' % (c + 1), 'max flood %d' % (c + 1), 'high slack %d' % (c + 1), 'max ebb %d' % (c + 1)]
PH = it[np.array(PH)]
for n, l in zip(PH, LAB):
    print('  %-13s %s  Q %+6.0f m3/s' % (l, T[n].strftime('%m-%d %H:%M'), Q[n]))

# ------------------------------------------------------------------ fields at the phase hours
def to_rho(uu, vv):
    """u, v (..., eta_u, xi_u), (..., eta_v, xi_v) -> rho points; land faces count as 0 (no flow through a wall)."""
    uu, vv = np.nan_to_num(uu), np.nan_to_num(vv)
    ur = np.full(uu.shape[:-1] + (uu.shape[-1] + 1,), np.nan)
    vr = np.full(vv.shape[:-2] + (vv.shape[-2] + 1, vv.shape[-1]), np.nan)
    ur[..., 1:-1] = 0.5 * (uu[..., :-1] + uu[..., 1:])
    vr[..., 1:-1, :] = 0.5 * (vv[..., :-1, :] + vv[..., 1:, :])
    return ur, vr


S = ds.salt.isel(ocean_time=PH).to_numpy().astype(float)            # (phase, k, eta, xi)
U, V = to_rho(ds.u.isel(ocean_time=PH).to_numpy().astype(float), ds.v.isel(ocean_time=PH).to_numpy().astype(float))
ZW = ds.z_w.isel(ocean_time=PH).to_numpy()
ds.close()
for A in (S, U, V):
    A[:, :, ~wet] = np.nan
F = {'surface': (S[:, -1], U[:, -1], V[:, -1]), 'bottom': (S[:, 0], U[:, 0], V[:, 0])}
SEC_U = np.stack([-(U[:, :, j, i].mean(axis=-1) * tx + V[:, :, j, i].mean(axis=-1) * ty)
                  for i, j, tx, ty in zip(XI, JC, TX, TY)], axis=-1)                       # (phase, k, n) + toward the head
SEC_S = np.stack([S[:, :, j, i].mean(axis=-1) for i, j in zip(XI, JC)], axis=-1)
SEC_Z = np.stack([ZW[:, :, j, i].mean(axis=-1) for i, j in zip(XI, JC)], axis=-1)              # (phase, k+1, n)

# ------------------------------------------------------------------ plot
NQ = 2 if args.zoom else 3              # arrow every NQ cells
# colour limits and arrow scale from the cove and Saratoga only: the Skagit Bay entrance in the NE
# corner of the box is far fresher and faster and would set both
REF = wet & ~((lat > 48.28) & (lon > -122.56))
ZOOM_EXT = [-122.74, -122.60, 48.20, 48.27]    # the cove and ~3.5 km of Saratoga east of the mouth
X_OUT = 3.0                                    # -zoom: section drawn to this many km past the mouth
if args.zoom:                             # limits and arrow scale from what is in view
    REF = wet & (lon >= ZOOM_EXT[0]) & (lon <= ZOOM_EXT[1]) & (lat >= ZOOM_EXT[2]) & (lat <= ZOOM_EXT[3])
LAND = np.where(wet, np.nan, 1.0)
jj, ii = np.where(wet)
EXT = ZOOM_EXT if args.zoom else [lon[jj, ii].min() - 0.005, lon[jj, ii].max() + 0.005, lat[jj, ii].min() - 0.003, lat[jj, ii].max() + 0.003]
XMAX = x_mouth + X_OUT if args.zoom else xe[-1]   # section x-limit [km from the head]
DS_ISO = 0.2 if args.zoom else 0.25              # isohaline interval [g/kg]
lon_lp = 0.5 * (lon[ETA, XU] + lon[ETA, XU + 1])
lat_lp = lat[ETA, XU]


def lims(A, pct=1):
    A = A[:, REF]
    return np.nanpercentile(A, pct), np.nanpercentile(A, 100 - pct)


fig = plt.figure(figsize=(26, 11 if args.zoom else 12.5), layout='constrained')
gs = fig.add_gridspec(4, 8, height_ratios=[0.7, 1.6, 1.6, 1.5] if args.zoom else [0.7, 2.2, 2.2, 1.5])

# Q and sea level over the day, phases marked
ax = fig.add_subplot(gs[0, :])
tw = T[it]
ax.plot(tw, Q[it], color='k', lw=1.5)
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('mouth Q [m$^3$ s$^{-1}$]\n+ into the cove')
ax2 = ax.twinx()
ax2.plot(tw, zeta_m[it], color='#0072B2', lw=1)
ax2.set_ylabel('mouth sea level [m]', color='#0072B2')
ax2.tick_params(axis='y', colors='#0072B2')
for n, l in zip(PH, LAB):
    ax.axvline(T[n], color='0.5', lw=0.8, ls='--')
    ax.plot(T[n], Q[n], 'o', color='k', ms=5)
    ax.text(T[n], ax.get_ylim()[1], ' ' + l, rotation=0, ha='center', va='bottom', fontsize=8)
ax.axvspan(T0, T1, color=(1, 0.93, 0.8), lw=0, zorder=0)    # opaque pale orange (alpha turns solid on a transparent png)
ax.set_xlim(tw[0], tw[-1])
ax.grid(True, alpha=0.3)

# plan views
for r, (row, (smin_pct, ttl)) in enumerate(zip(['surface', 'bottom'], [(1, 'surface (top cell)'), (1, 'bottom (bottom cell)')])):
    s, uu, vv = F[row]
    vmin, vmax = lims(s, smin_pct)
    spd = np.hypot(uu, vv)
    qref = max(0.05, np.round(np.nanpercentile(spd[:, REF], 95) * 20) / 20)    # arrow key speed [m/s], nearest 0.05
    axs = [fig.add_subplot(gs[1 + r, c]) for c in range(8)]
    for c, ax in enumerate(axs):
        pc = ax.pcolormesh(lon, lat, s[c], cmap=cmocean.cm.haline, vmin=vmin, vmax=vmax, shading='nearest')
        ax.pcolormesh(lon, lat, LAND, cmap=matplotlib.colors.ListedColormap(['0.8']), shading='nearest')
        sl = (slice(None, None, NQ), slice(None, None, NQ))
        qv = ax.quiver(lon[sl], lat[sl], uu[c][sl], vv[c][sl], scale=qref * 12, scale_units='width', width=0.004, color='k')
        ax.plot(lon_line, lat_line, color='w', lw=1.2, ls=(0, (2, 1.5)))
        ax.plot(lon_lp, lat_lp, color='#D55E00', lw=2)
        ax.set_xlim(EXT[:2])
        ax.set_ylim(EXT[2:])
        ax.set_aspect(1 / np.cos(np.deg2rad(48.22)))
        ax.tick_params(labelsize=6)
        if args.zoom:
            ax.set_xticks([-122.72, -122.68, -122.64, -122.60])
        if c:
            ax.set_yticklabels([])
        if r == 0:
            ax.set_title('%s\n%s UTC (+%.0f h)' % (LAB[c], T[PH[c]].strftime('%m-%d %H:%M'), (T[PH[c]] - T[PH[0]]).total_seconds() / 3600), fontsize=9)
        if c == 0:
            ax.set_ylabel(ttl, fontsize=10)
    ax.quiverkey(qv, 0.75, 0.04, qref, '%.2f m s$^{-1}$' % qref, labelpos='N', fontproperties={'size': 7})
    pc.set_clim(vmin, vmax)
    fig.colorbar(pc, ax=axs, location='right', shrink=0.8, pad=0.01, extend='both').set_label('%s salinity [g kg$^{-1}$]' % row, fontsize=9)

# sections
ref_line = np.array([REF[j, i].all() for i, j in zip(XI, JC)]) & (xc <= XMAX)   # same exclusion as the maps, and in view
umax = np.nanpercentile(np.abs(SEC_U[:, :, ref_line]), 99)
axs = [fig.add_subplot(gs[3, c]) for c in range(8)]
for c, ax in enumerate(axs):
    zw = SEC_Z[c]
    for n in range(len(XI)):
        pc = ax.pcolormesh(xe[n:n + 2], zw[:, n], SEC_U[c][:, n:n + 1], cmap=cmocean.cm.balance, vmin=-umax, vmax=umax, shading='flat')
    zr = 0.5 * (zw[1:] + zw[:-1])
    cs = ax.contour(np.broadcast_to(xc, zr.shape), zr, SEC_S[c], levels=np.arange(20, 32, DS_ISO), colors='k', linewidths=0.5)
    ax.clabel(cs, cs.levels[::2], fontsize=5, fmt='%.1f')
    ax.axvline(x_mouth, color='0.3', lw=1, ls='--')
    ax.set_xlim(xe[0], XMAX)
    ax.set_ylim(np.nanmin(SEC_Z[:, :, xc <= XMAX]) - 1, 1.5)
    ax.tick_params(labelsize=7)
    ax.set_xlabel('km from the cove head', fontsize=8)
    if c:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('z [m]\n(head | mouth -- | Saratoga to Skagit)', fontsize=9)
fig.colorbar(pc, ax=axs, location='right', shrink=0.9, pad=0.01, extend='both').set_label('velocity along the line [m s$^{-1}$]\n+ toward the cove head', fontsize=9)

fig.suptitle('Penn Cove and upper Saratoga Passage through one lunar day: %s, %s to %s UTC\n'
             'ranks D2 %.2f, D1 %.2f, strat %.2f; arrows = cell velocity; orange = pc_lp (mouth); white dotted = section line; '
             'contours = isohalines (%.2g g/kg)' % (VARIANT, T0.strftime('%Y-%m-%d %H:%M'), T1.strftime('%m-%d %H:%M'), day.r_D2, day.r_D1, day.r_strat, DS_ISO),
             fontsize=12)
out_fn = out_dir / ('tidal_day_%s%s.png' % (VARIANT.replace(' ', '_'), '_zoom' if args.zoom else ''))
fig.savefig(out_fn, dpi=150, transparent=True)
plt.close(fig)
print('saved ' + str(out_fn))
