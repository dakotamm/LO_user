"""
Upper Saratoga Passage as a junction over one lunar day: how the tide is shared between
south Saratoga, the Skagit Bay entrance and Penn Cove, and what water each gate carries.

CONTROL VOLUME  the Whidbey water in the pc_sp_cycle box (wet cells connected to Penn Cove)
    between three gates, cut on the native grid and checked by flood fill (the volume must be
    sealed for the budget to close; hand-drawn lines overshoot into land):
      S      v faces eta_v = 0 (lat 48.119), across Saratoga, + north = INTO the junction
      NE     u faces xi_u = 84 (rho columns 84 | 85), rows 76-96, the Skagit Bay entrance throat,
             + west = INTO the junction
      pc_lp  u faces xi_u = 32, rows 60-71, the Penn Cove mouth, + west = into the cove = OUT of
             the junction
    Budget: dV/dt = Q_S + Q_NE - Q_pc (+ tiny rivers/WWTPs, ignored). dV/dt from hourly zeta by
    centred difference; transports are hourly snapshots, so the residual includes snapshot
    aliasing, not only rivers.
GATE PROFILES  cross-gate averages U_b = Q_b / A_b and S_b = (salt dA)_b / A_b in 1 m bins below the
    moving surface (as 20261110_pc_midchannel_u_timeseries.py), x = time, y = z.

DAY  the baseline lunar day or a one-axis variant (20261105_pc_standard_cycle lunar_days.csv /
    variants.csv), shown +-3 h.
INPUT  LO_output/extract/[gtagex]/box/pc_sp_cycle_[ds0]_[ds1].nc (hourly ocean_his)
OUTPUT LO_output/DM_outs/20261111_sp_junction_tidal_cycle/budget_[variant].png,
       gates_[variant].png

run 20261111_sp_junction_tidal_cycle.py [-variant baseline|spring|neap|...]
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
from scipy import ndimage

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-variant', default='baseline')
args = parser.parse_args()
VARIANT = args.variant.replace('_', ' ')

GTAGEX = 'wb1_t0_xn11abbur00'
COVE_CELL = (65, 20)                      # (j, i) inside Penn Cove: seeds the Whidbey water mask
SP_CELL = (40, 70)                        # (j, i) in Saratoga Passage: seeds the junction
SKB_CELL = (100, 105)                     # (j, i) in Skagit Bay, must end up OUTSIDE the junction
# gates: kind, face index, rows/cols it spans, sign that makes + = into the junction
GATES = {
    'S':     dict(kind='v', idx=0,  span=np.arange(0, 109),   sign=+1),
    'NE':    dict(kind='u', idx=84, span=np.arange(76, 97),   sign=-1),
    'pc_lp': dict(kind='u', idx=32, span=np.arange(60, 72),   sign=-1),   # + = into the COVE
}
NAME = {'S': 'S: south Saratoga', 'NE': 'NE: Skagit Bay entrance', 'pc_lp': 'pc_lp: Penn Cove mouth'}
COL = {'S': '#0072B2', 'NE': '#009E73', 'pc_lp': '#D55E00'}
PAD = pd.Timedelta('3h')

Ldir = Lfun.Lstart(gridname='wb1')
sel_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_standard_cycle'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261111_sp_junction_tidal_cycle'
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
pm, pn, h = ds.pm.values, ds.pn.values, ds.h.values

# ------------------------------------------------------------------ control volume: Whidbey water between the three gates
lab, _ = ndimage.label(ds.mask_rho.values == 1)
wet = lab == lab[COVE_CELL]
cut = wet.copy()
cut[0, :] = False                                                   # south of the S faces (eta_v 0 sits between rho rows 0 | 1)
cut[GATES['NE']['span'], GATES['NE']['idx'] + 1] = False             # Skagit side of the NE faces
cut[GATES['pc_lp']['span'], GATES['pc_lp']['idx']] = False           # cove side of the mouth faces
lab2, _ = ndimage.label(cut)
junc = lab2 == lab2[SP_CELL]
assert not junc[COVE_CELL] and not junc[SKB_CELL], 'a gate leaks'
assert not (junc[-1, :].any() or junc[:, 0].any() or junc[:, -1].any()), 'junction reaches a box edge'
area = 1 / (pm * pn)
print('junction: %d cells, %.1f km2, %.2f km3 at h' % (junc.sum(), area[junc].sum() / 1e6, (h * area)[junc].sum() / 1e9))

# faces of each gate: both neighbouring rho cells in Whidbey water
for G in GATES.values():
    a, b = G['idx'], G['idx'] + 1
    G['p'] = np.array([p for p in G['span'] if (wet[a, p] and wet[b, p] if G['kind'] == 'v' else wet[p, a] and wet[p, b])])
for g, G in GATES.items():
    print('  %-6s %2d %s faces' % (g, len(G['p']), G['kind']))

# ------------------------------------------------------------------ transports through the gates, junction volume
D = {vn: ds[vn].isel(**sl).to_numpy().astype(float) for vn in ['u', 'v', 'salt', 'z_w', 'zeta']}
for g, G in GATES.items():
    a, b, p = G['idx'], G['idx'] + 1, G['p']
    if G['kind'] == 'v':                  # face between rho rows a | b, columns p
        vel = D['v'][:, :, a, p]
        zw = 0.5 * (D['z_w'][:, :, a, p] + D['z_w'][:, :, b, p])
        dl = 0.5 * (1 / pm[a, p] + 1 / pm[b, p])
        s = 0.5 * (D['salt'][:, :, a, p] + D['salt'][:, :, b, p])
        zeta = 0.5 * (D['zeta'][:, a, p] + D['zeta'][:, b, p])
    else:                                 # face between rho columns a | b, rows p
        vel = D['u'][:, :, p, a]
        zw = 0.5 * (D['z_w'][:, :, p, a] + D['z_w'][:, :, p, b])
        dl = 0.5 * (1 / pn[p, a] + 1 / pn[p, b])
        s = 0.5 * (D['salt'][:, :, p, a] + D['salt'][:, :, p, b])
        zeta = 0.5 * (D['zeta'][:, p, a] + D['zeta'][:, p, b])
    G.update(u=G['sign'] * vel, zw=zw, dl=dl, s=s, zeta=(zeta * dl).sum(axis=1) / dl.sum())
    dA = np.diff(zw, axis=1) * dl[None, None, :]
    G['Q'] = (G['u'] * dA).sum(axis=(1, 2))
    G['A'] = dA.sum(axis=(1, 2))
    G['S'] = (s * dA).sum(axis=(1, 2)) / G['A']                        # gate-mean salinity

V = ((h[None] + D['zeta']) * area[None])[:, junc].sum(axis=1)
dVdt = np.gradient(V, 3600.0)                                       # centred; one-sided at the two ends
Qnet = GATES['S']['Q'] + GATES['NE']['Q'] - GATES['pc_lp']['Q']      # into the junction
res = dVdt - Qnet
ok = slice(1, -1)                                                    # skip the one-sided ends in the stats
print('budget: rms dV/dt %.0f, rms residual %.0f m3/s, r = %.4f'
      % (np.sqrt(np.mean(dVdt[ok] ** 2)), np.sqrt(np.mean(res[ok] ** 2)), np.corrcoef(dVdt[ok], Qnet[ok])[0, 1]))
for g, G in GATES.items():
    print('  %-6s Q %+6.0f to %+6.0f m3/s, mean %+5.0f; A %.0f m2; S %.2f-%.2f'
          % (g, G['Q'].min(), G['Q'].max(), G['Q'].mean(), G['A'].mean(), G['S'].min(), G['S'].max()))

# ------------------------------------------------------------------ cross-gate averaged profiles, 1 m bins below the moving surface
def binned(G):
    """U_b = Q_b / A_b and S_b = (s dA)_b / A_b, bottom to top as (t, b), with z edges (t, b + 1) on
    z = zs - depth; NaN where a bin is wetted less than half a face (as 20261110_pc_midchannel_u_timeseries)."""
    zw, dl = G['zw'], G['dl'][None, None, :]
    dz = np.diff(zw, axis=1)
    dw = zw[:, -1:, :] - zw
    d_top, d_bot = dw[:, 1:, :], dw[:, :-1, :]
    E = np.arange(0, np.ceil(d_bot.max()) + 1)
    nb = len(E) - 1
    Qb, Sb, Ab = np.zeros((len(zw), nb)), np.zeros((len(zw), nb)), np.zeros((len(zw), nb))
    for b in range(nb):
        ov = np.clip(np.minimum(d_bot, E[b + 1]) - np.maximum(d_top, E[b]), 0, None) * dl
        Qb[:, b] = (G['u'] * ov).sum(axis=(1, 2))
        Sb[:, b] = (G['s'] * ov).sum(axis=(1, 2))
        Ab[:, b] = ov.sum(axis=(1, 2))
    assert np.allclose(Qb.sum(axis=1), G['Q'])
    full = Ab >= 0.5 * dl.mean()
    Ub = np.where(full, Qb / np.maximum(Ab, 1e-12), np.nan)
    Sb = np.where(full, Sb / np.maximum(Ab, 1e-12), np.nan)
    zs = (zw[:, -1, :] * dl[0]).sum(axis=-1) / dl.sum()
    return Ub[:, ::-1], Sb[:, ::-1], zs[:, None] - E[None, ::-1]


for g, G in GATES.items():
    G['Ub'], G['Sb'], G['ze'] = binned(G)

# phase: lag of each transport and sea level behind the mouth's, by cross-correlation on a 6 min spline
from scipy.interpolate import CubicSpline
th = (T - T[0]).total_seconds().to_numpy() / 3600
tf = np.arange(th[0], th[-1], 0.1)
def lag(x, y, maxlag=6.5):
    """hours by which y lags x (positive = y later), at the peak of |correlation| within +-maxlag h
    (half an M2 period, so in-phase and anti-phase are both reachable); r keeps its sign"""
    xf, yf = CubicSpline(th, x)(tf), CubicSpline(th, y)(tf)
    xf, yf = xf - xf.mean(), yf - yf.mean()
    L = np.arange(-int(maxlag / 0.1), int(maxlag / 0.1) + 1)
    c = [np.sum(xf[max(0, -l):len(xf) - max(0, l)] * yf[max(0, l):len(yf) - max(0, -l)]) for l in L]
    k = np.argmax(np.abs(c))
    return 0.1 * L[k], c[k] / np.sqrt(np.sum(xf ** 2) * np.sum(yf ** 2))
print('lag behind pc_lp (h, r):')
for g, G in GATES.items():
    print('  %-6s Q %+.1f h (r %+.2f)   zeta %+.1f h (r %+.2f)'
          % ((g,) + lag(GATES['pc_lp']['Q'], G['Q']) + lag(GATES['pc_lp']['zeta'], G['zeta'])))
print('  dV/dt  %+.1f h (r %+.2f)' % lag(GATES['pc_lp']['Q'], dVdt))
ds.close()

# ------------------------------------------------------------------ figure 1: the junction budget
def shade(ax, zero=True):
    ax.axvspan(T0, T1, color='0.92', zorder=0)
    if zero:
        ax.axhline(0, color='k', lw=0.6)
    ax.grid(lw=0.3)

fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True, gridspec_kw=dict(height_ratios=[1, 2, 1.3, 1.3]))
ax = axs[0]
for g, G in GATES.items():
    ax.plot(T, G['zeta'], color=COL[g], lw=1.2, label=g)
shade(ax)
ax.set_ylabel('gate sea level [m]')
ax.legend(fontsize=8, ncol=3, loc='upper left')
ax = axs[1]
for g in ['S', 'NE']:
    ax.plot(T, GATES[g]['Q'] / 1e3, color=COL[g], lw=1.8, label=NAME[g])
ax.plot(T, GATES['pc_lp']['Q'] / 1e3, color=COL['pc_lp'], lw=1.8, label=NAME['pc_lp'])
ax.plot(T, dVdt / 1e3, color='k', lw=1, ls='--', label='junction dV/dt')
ax.plot(T, res / 1e3, color='0.5', lw=0.8, label='residual dV/dt - (Q$_S$ + Q$_{NE}$ - Q$_{pc}$)')
shade(ax)
ax.set_ylabel('Q [10$^3$ m$^3$ s$^{-1}$]\n+ toward Penn Cove')
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax = axs[2]
ax.plot(T, GATES['pc_lp']['Q'], color=COL['pc_lp'], lw=1.8)
shade(ax)
ax.set_ylabel('Q$_{pc}$ [m$^3$ s$^{-1}$]\n+ into the cove')
ax = axs[3]
for g, G in GATES.items():
    ax.plot(T, G['S'], color=COL[g], lw=1.5, label=g)
shade(ax, zero=False)
ax.set_ylabel('gate-mean salinity\n[g kg$^{-1}$]')
ax.legend(fontsize=8, ncol=3, loc='lower left')
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
axs[0].set_title('Upper Saratoga junction, %s lunar day (grey), %s to %s UTC\nvolume %.2f km$^3$, %.0f km$^2$; '
                 'budget r = %.3f, rms residual %.0f m$^3$ s$^{-1}$ (%.0f%% of rms dV/dt)'
                 % (VARIANT, T0.strftime('%Y-%m-%d %H:%M'), T1.strftime('%m-%d %H:%M'), (h * area)[junc].sum() / 1e9,
                    area[junc].sum() / 1e6, np.corrcoef(dVdt[ok], Qnet[ok])[0, 1], np.sqrt(np.mean(res[ok] ** 2)),
                    100 * np.sqrt(np.mean(res[ok] ** 2) / np.mean(dVdt[ok] ** 2))), fontsize=10, loc='left')
fig.tight_layout()
fig.savefig(out_dir / ('budget_%s.png' % VARIANT.replace(' ', '_')), dpi=150, transparent=True)
plt.close(fig)

# ------------------------------------------------------------------ figure 2: gate profiles
def grid(ze):
    """pcolormesh corners: hour columns with edges half an hour either side, z at an edge = mean of the two hours"""
    t = mdates.date2num(T)
    te = np.concatenate([[t[0] - (t[1] - t[0]) / 2], 0.5 * (t[1:] + t[:-1]), [t[-1] + (t[-1] - t[-2]) / 2]])
    zz = np.concatenate([ze[:1], 0.5 * (ze[1:] + ze[:-1]), ze[-1:]])                   # (t + 1, b + 1)
    return np.repeat(te[:, None], zz.shape[1], axis=1), zz                            # x as matplotlib date numbers

fig, axs = plt.subplots(3, 2, figsize=(14, 11), sharex=True, gridspec_kw=dict(height_ratios=[2, 1.2, 1]))
for r, (g, G) in enumerate(GATES.items()):
    X, Z = grid(G['ze'])
    zc = 0.5 * (G['ze'][:, 1:] + G['ze'][:, :-1])
    Tc = np.repeat(mdates.date2num(T)[:, None], zc.shape[1], axis=1)
    ul = np.nanpercentile(np.abs(G['Ub']), 99)
    s0, s1 = np.nanpercentile(G['Sb'], [1, 99])
    for c, (F, cmap, vlim, lab_) in enumerate([(G['Ub'], cmocean.cm.balance_r, (-ul, ul), 'U$_b$ [m s$^{-1}$], blue = toward Penn Cove'),
                                               (G['Sb'], cmocean.cm.haline, (s0, s1), 'S$_b$ [g kg$^{-1}$]')]):
        ax = axs[r, c]
        pc = ax.pcolormesh(X.T, Z.T, F.T, cmap=cmap, vmin=vlim[0], vmax=vlim[1], shading='flat')
        ax.contour(Tc, zc, np.nan_to_num(G['Ub']), [0], colors='k', linewidths=0.6)
        ax.axvline(T0, color='k', lw=0.8, ls=':')
        ax.axvline(T1, color='k', lw=0.8, ls=':')
        fig.colorbar(pc, ax=ax, pad=0.01).set_label(lab_, fontsize=8)
        ax.set_title('%s: %s' % (NAME[g], 'velocity' if c == 0 else 'salinity'), fontsize=9, loc='left')
        if c == 0:
            ax.set_ylabel('z [m]')
for ax in axs[-1]:
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
fig.suptitle('Gate-averaged profiles (1 m bins below the surface), %s lunar day between dotted lines; black = U$_b$ = 0'
             % VARIANT, fontsize=10)
fig.tight_layout()
fig.savefig(out_dir / ('gates_%s.png' % VARIANT.replace(' ', '_')), dpi=150, transparent=True)
plt.close(fig)
print('saved to', out_dir)
