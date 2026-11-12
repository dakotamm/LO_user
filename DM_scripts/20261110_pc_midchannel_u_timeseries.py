"""
Time series of the cross-channel averaged velocity profile at the three Penn Cove sections over one lunar day:
one row per section, pc_cp | pc_lj | pc_lp (west to east, inner cove to mouth, top to bottom),
x = time, y = z, colour = u normal to the section, + = into the cove (west) drawn BLUE, out RED
(cmocean balance_r). A strip on top gives sea level and the section-mean velocity Ubar = Q / A at the
mouth (pc_lp) for the tidal phase.

DAY  the baseline lunar day (calm wind, median strat, median spring/neap and tropic/equatorial),
    2024-08-31 05:00 -> 09-01 06:00 UTC, or a one-axis variant, read from 20261105_pc_standard_cycle
    (lunar_days.csv, variants.csv). Hourly ocean_his snapshots from the pc_cove box.
PROFILE  cross-channel average over the whole section, U_b = Q_b / A_b in 1 m bins of depth below
    the moving surface (20261103_pc_section_profiles.py): each cell's transport split among the
    bins it overlaps, so sum_b Q_b = Q exactly; bins wetted less than half a face (dy x 0.5 m)
    are blank. Bin b sits at z = zs - depth, zs = width-mean sea surface of the section at that hour.
GRID  each hour is a column between its own bin edges (the surface moves); the column edges sit
    half an hour either side of the snapshot, with z at an edge = mean of the two hours it
    separates. Thin black line = the u = 0 contour. One colour scale for all three rows (99th
    percentile of |u|).

ACROSS-CHANNEL version (second figure): y = latitude of each u face (rows share one latitude axis,
    so the sections line up geographically, north at the top), colour = that face's depth-averaged
    u = sum_k u dz / sum_k dz over the whole water column at that hour. Own colour scale (99th
    percentile of |depth-averaged u|), zero contour as above.
ARROWS  on both figures, every ASTEP h (profile: every ZSTEP-th s level; across: every face):
    pointing LEFT = into the cove (west), RIGHT = out, length in proportion to |u| (full length at
    the colour limit); none under AMIN of the colour limit.

INPUT  LO_output/extract/[gtagex]/box/pc_cove_[ds0]_[ds1].nc
OUTPUT LO_output/DM_outs/20261110_pc_midchannel_u_timeseries/profile_u_[variant].png,
       across_ubar_[variant].png

run 20261110_pc_midchannel_u_timeseries.py [-variant baseline|spring|neap|...]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
import cmocean

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-variant', default='baseline')
args = parser.parse_args()
VARIANT = args.variant.replace('_', ' ')

GTAGEX = 'wb1_t0_xn11abbur00'
DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                 # section-positive (east) is out of the cove
FACES = {                                 # u faces in the pc_cove box, north to south; west to east
    'pc_cp': dict(XI_U=15, ETA=np.arange(13, 5, -1)),   # global i 52, j 222..215
    'pc_lj': dict(XI_U=23, ETA=np.arange(14, 6, -1)),   # global i 60, j 223..216
    'pc_lp': dict(XI_U=30, ETA=np.arange(19, 7, -1)),   # global i 67, j 228..217
}
NAME = {'pc_lp': 'pc_lp (mouth)', 'pc_lj': 'pc_lj (middle)', 'pc_cp': 'pc_cp (Coupeville)'}
COL_IN, COL_OUT = '#2166AC', '#B2182B'   # blue = entering the cove, red = leaving
EDGES = np.arange(0, 31, 1.0)             # profile bins, depth below the surface [m]
ASTEP, ZSTEP = 2, 3                       # direction arrows every 2 h, on every 3rd s level (profile) or every face (across)
AMIN = 0.1                                # no arrow where |u| < 10% of the colour limit

Ldir = Lfun.Lstart(gridname='wb1')
in_fn = Ldir['LOo'] / 'extract' / GTAGEX / 'box' / ('pc_cove_%s_%s.nc' % (DS0, DS1))
sel_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_standard_cycle'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261110_pc_midchannel_u_timeseries'
out_dir.mkdir(parents=True, exist_ok=True)

if VARIANT == 'baseline':
    day = pd.read_csv(sel_dir / 'lunar_days.csv', parse_dates=['t0', 't1']).iloc[0]
else:
    V = pd.read_csv(sel_dir / 'variants.csv', parse_dates=['t0', 't1'])
    day = V[V.variant == VARIANT].iloc[0]
T0, T1 = day.t0, day.t1
print('%s: lunar day %s to %s UTC' % (VARIANT, T0, T1))

# ------------------------------------------------------------------ cross-channel averaged profile of each section
def binned(u, zw, dy):
    """U_b = Q_b / A_b in the EDGES bins below the moving surface, returned bottom to top as (u (t, b),
    z edges (t, b + 1)) on z = zs - depth; NaN where a bin is wetted less than half a face."""
    dz = np.diff(zw, axis=1)
    q = u * dz * dy
    dw = zw[:, -1:, :] - zw                                          # depth below the surface of each interface
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    assert d_bot.max() < EDGES[-1]
    nb = len(EDGES) - 1
    Qb, Ab = np.zeros((len(u), nb)), np.zeros((len(u), nb))
    for b in range(nb):
        ov = np.clip(np.minimum(d_bot, EDGES[b + 1]) - np.maximum(d_top, EDGES[b]), 0, None)
        Qb[:, b] = (q * ov / dz).sum(axis=(1, 2))
        Ab[:, b] = (ov * dy).sum(axis=(1, 2))
    assert np.allclose(Qb.sum(axis=1), q.sum(axis=(1, 2)))
    Ub = np.where(Ab >= 0.5 * dy.mean(), Qb / np.maximum(Ab, 1e-12), np.nan)
    zs = (zw[:, -1, :] * dy).sum(axis=-1) / dy.sum()
    return Ub[:, ::-1], zs[:, None] - EDGES[None, ::-1]


ds = xr.open_dataset(in_fn)
TA = pd.DatetimeIndex(ds.ocean_time.values)
it = np.flatnonzero((TA >= T0) & (TA <= T1))
T = TA[it]
sl = dict(ocean_time=slice(it[0], it[-1] + 1))

R = {}
for SN, F in FACES.items():
    XI_U, ETA = F['XI_U'], F['ETA']
    u = SIGN * ds.u.isel(xi_u=XI_U, eta_u=ETA, **sl).to_numpy().astype(float)                  # (t, k, p) + into the cove
    zw = ds.z_w.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().mean(axis=-1)     # (t, w, p) at the face
    zeta = ds.zeta.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1], **sl).to_numpy().mean(axis=(1, 2))
    dy = (1 / ds.pn.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy()).mean(axis=-1)
    dz = np.diff(zw, axis=1)
    UBAR = (u * dz * dy).sum(axis=(1, 2)) / (dz * dy).sum(axis=(1, 2))
    h = ds.h.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy().mean(axis=-1)
    lat = ds.lat_u.isel(eta_u=ETA, xi_u=XI_U).to_numpy()            # north to south
    ub, ze = binned(u, zw, dy)
    R[SN] = dict(u=ub, zw=ze, zeta=zeta, UBAR=UBAR,
                 uface=(u * dz).sum(axis=1) / dz.sum(axis=1),        # (t, p) depth average of each face
                 late=np.concatenate([[lat[0] - (lat[1] - lat[0]) / 2], 0.5 * (lat[1:] + lat[:-1]), [lat[-1] + (lat[-1] - lat[-2]) / 2]]))
    print('  %-6s %d faces (h max %.1f m): U_b %+.1f to %+.1f cm/s'
          % (SN, len(ETA), h.max(), 100 * np.nanmin(ub), 100 * np.nanmax(ub)))
ds.close()

def arrows(ax, x, Y, U, umax, xlen):
    """Direction arrows: x (n,) centres, Y (n, m) positions, U (n, m) cm/s, + into the cove drawn pointing
    LEFT (west), length |U| / umax * xlen in x units; none where |U| < AMIN of umax."""
    X = np.broadcast_to(x[:, None], Y.shape)
    m = np.abs(U) >= AMIN * umax
    ax.quiver(X[m], Y[m], -np.clip(U[m], -umax, umax) / umax * xlen, np.zeros(m.sum()), angles='xy', scale_units='xy', scale=1,
              pivot='middle', color='k', width=0.003, headwidth=4, headlength=4, headaxislength=3.5, zorder=3)


# time edges half an hour either side of each snapshot; z_w at an edge = mean of the hours either side
tnum = mdates.date2num(T)
te = np.concatenate([[tnum[0] - 0.5 / 24], 0.5 * (tnum[1:] + tnum[:-1]), [tnum[-1] + 0.5 / 24]])

# ------------------------------------------------------------------ figure
umax = np.ceil(100 * np.nanpercentile(np.abs(np.concatenate([R[a]['u'].ravel() for a in FACES])), 99))
norm = matplotlib.colors.Normalize(-umax, umax)
cmap = cmocean.cm.balance_r
zbot = min(np.nanmin(np.where(np.isfinite(R[a]['u']), R[a]['zw'][:, :-1], np.nan)) for a in FACES) - 0.5   # deepest wet bin
ztop = max(R[a]['zw'][:, -1].max() for a in FACES) + 0.5

fig = plt.figure(figsize=(11, 10), layout='constrained')
gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 2, 2])
ax_t = fig.add_subplot(gs[0])
axs = [fig.add_subplot(gs[c + 1], sharex=ax_t) for c in range(3)]

ax_t.plot(T, 100 * R['pc_lp']['UBAR'], color='k', lw=1.5, label='Ubar pc_lp (mouth)')
ax_t.fill_between(T, 0, 100 * R['pc_lp']['UBAR'], where=R['pc_lp']['UBAR'] > 0, color=COL_IN, alpha=0.3, lw=0, interpolate=True)
ax_t.fill_between(T, 0, 100 * R['pc_lp']['UBAR'], where=R['pc_lp']['UBAR'] < 0, color=COL_OUT, alpha=0.3, lw=0, interpolate=True)
ax_t.axhline(0, color='k', lw=0.5)
ax_t.set_ylabel('Ubar = Q / A\n[cm s$^{-1}$], + in')
ax_z = ax_t.twinx()
ax_z.plot(T, R['pc_lp']['zeta'], color='0.4', lw=1, ls='--', label='sea level pc_lp')
ax_z.set_ylabel('sea level [m]', color='0.4')
ax_z.tick_params(axis='y', colors='0.4')
h1, l1 = ax_t.get_legend_handles_labels()
h2, l2 = ax_z.get_legend_handles_labels()
ax_t.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left', ncol=2)
ax_t.grid(True, alpha=0.3)
ax_t.tick_params(labelbottom=False)

for ax, SN in zip(axs, FACES):
    r = R[SN]
    zw = r['zw']                                                     # (t, w)
    zwe = np.concatenate([zw[:1], 0.5 * (zw[1:] + zw[:-1]), zw[-1:]], axis=0)   # (t + 1, w) at the time edges
    X = np.broadcast_to(te[:, None], zwe.shape)
    ax.pcolormesh(X.T, zwe.T, np.ma.masked_invalid(100 * r['u']).T, cmap=cmap, norm=norm, shading='flat')
    zr = 0.5 * (zw[:, 1:] + zw[:, :-1])                              # (t, k) cell centres
    ax.contour(np.broadcast_to(tnum[:, None], zr.shape).T, zr.T, np.ma.masked_invalid(r['u']).T, levels=[0], colors='k', linewidths=0.6)
    ax.plot(te, zwe[:, -1], color='0.3', lw=0.6)                     # the moving surface
    arrows(ax, tnum[::ASTEP], zr[::ASTEP, ::ZSTEP], 100 * r['u'][::ASTEP, ::ZSTEP], umax, 0.9 * ASTEP / 24)
    ax.set_ylim(zbot, ztop)
    ax.set_ylabel('z [m]')
    ax.set_title('%s: cross-channel average U$_b$ = Q$_b$ / A$_b$, 1 m bins below the surface' % NAME[SN], fontsize=10, loc='left')
    ax.grid(True, alpha=0.3)
    if SN != 'pc_lp':
        ax.tick_params(labelbottom=False)
axs[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
axs[-1].set_xlabel('UTC')
ax_t.set_xlim(te[0], te[-1])

sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=axs, location='right', shrink=0.8, pad=0.01, extend='both')
cb.set_label('cross-channel averaged u normal to the section [cm s$^{-1}$]\n+ into the cove (west)')
cb.ax.text(0.5, 1.02, 'IN', transform=cb.ax.transAxes, ha='center', va='bottom', fontweight='bold', color=COL_IN)
cb.ax.text(0.5, -0.02, 'OUT', transform=cb.ax.transAxes, ha='center', va='top', fontweight='bold', color=COL_OUT)
fig.suptitle('%s lunar day %s to %s UTC: cross-channel averaged velocity profile, hourly ocean_his, %s\n'
             'blue = into the cove, red = out; black line = u = 0; arrows: \u2190 IN (west, into the cove)   OUT \u2192, length \u221d |u|'
             % (VARIANT, T0.strftime('%Y-%m-%d %H:%M'), T1.strftime('%m-%d %H:%M'), GTAGEX), fontsize=11)

fn = out_dir / ('profile_u_%s.png' % VARIANT.replace(' ', '_'))
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % fn)

# ------------------------------------------------------------------ across-channel figure: depth-averaged u per face
umax = np.ceil(100 * np.percentile(np.abs(np.concatenate([R[a]['uface'].ravel() for a in FACES])), 99))
norm = matplotlib.colors.Normalize(-umax, umax)
ylo = min(R[a]['late'].min() for a in FACES)
yhi = max(R[a]['late'].max() for a in FACES)

fig = plt.figure(figsize=(11, 10), layout='constrained')
gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 2, 2])
ax_t = fig.add_subplot(gs[0])
axs = [fig.add_subplot(gs[c + 1], sharex=ax_t) for c in range(3)]
ax_t.plot(T, 100 * R['pc_lp']['UBAR'], color='k', lw=1.5, label='Ubar pc_lp (mouth)')
ax_t.fill_between(T, 0, 100 * R['pc_lp']['UBAR'], where=R['pc_lp']['UBAR'] > 0, color=COL_IN, alpha=0.3, lw=0, interpolate=True)
ax_t.fill_between(T, 0, 100 * R['pc_lp']['UBAR'], where=R['pc_lp']['UBAR'] < 0, color=COL_OUT, alpha=0.3, lw=0, interpolate=True)
ax_t.axhline(0, color='k', lw=0.5)
ax_t.set_ylabel('Ubar = Q / A\n[cm s$^{-1}$], + in')
ax_z = ax_t.twinx()
ax_z.plot(T, R['pc_lp']['zeta'], color='0.4', lw=1, ls='--', label='sea level pc_lp')
ax_z.set_ylabel('sea level [m]', color='0.4')
ax_z.tick_params(axis='y', colors='0.4')
h1, l1 = ax_t.get_legend_handles_labels()
h2, l2 = ax_z.get_legend_handles_labels()
ax_t.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left', ncol=2)
ax_t.grid(True, alpha=0.3)
ax_t.tick_params(labelbottom=False)

for ax, SN in zip(axs, FACES):
    r = R[SN]
    ax.pcolormesh(te, r['late'], 100 * r['uface'].T, cmap=cmap, norm=norm, shading='flat')
    latc = 0.5 * (r['late'][1:] + r['late'][:-1])
    ax.contour(tnum, latc, r['uface'].T, levels=[0], colors='k', linewidths=0.6)
    arrows(ax, tnum[::ASTEP], np.broadcast_to(latc, r['uface'][::ASTEP].shape), 100 * r['uface'][::ASTEP], umax, 0.9 * ASTEP / 24)
    ax.set_ylim(ylo, yhi)                                            # north at the top
    ax.set_ylabel('latitude [deg N]\nS \u2193   N \u2191')
    ax.set_title('%s: depth-averaged u of each face (%d faces)' % (NAME[SN], len(latc)), fontsize=10, loc='left')
    ax.grid(True, alpha=0.3)
    if SN != 'pc_lp':
        ax.tick_params(labelbottom=False)
axs[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%m-%d'))
axs[-1].set_xlabel('UTC')
ax_t.set_xlim(te[0], te[-1])

sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=axs, location='right', shrink=0.8, pad=0.01, extend='both')
cb.set_label('depth-averaged u normal to the section [cm s$^{-1}$]\n+ into the cove (west)')
cb.ax.text(0.5, 1.02, 'IN', transform=cb.ax.transAxes, ha='center', va='bottom', fontweight='bold', color=COL_IN)
cb.ax.text(0.5, -0.02, 'OUT', transform=cb.ax.transAxes, ha='center', va='top', fontweight='bold', color=COL_OUT)
fig.suptitle('%s lunar day %s to %s UTC: depth-averaged velocity across each section\n'
             'hourly ocean_his, %s; blue = into the cove, red = out; black line = 0\n'
             'arrows: \u2190 IN (west, into the cove)   OUT \u2192, length \u221d |u|'
             % (VARIANT, T0.strftime('%Y-%m-%d %H:%M'), T1.strftime('%m-%d %H:%M'), GTAGEX), fontsize=11)
fn = out_dir / ('across_ubar_%s.png' % VARIANT.replace(' ', '_'))
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved %s' % fn)
