"""
How Penn Cove's density structure and layered exchange change with Skagit River flow,
wb1_t0_xn11abbur00, 2024-2025, daily (Godin-filtered hourly series, daily means).

SKAGIT FLOW  model forcing (gauged USGS, trapsN00), DM_outs/20260806_river_hydrographs/daily_flow.csv,
    LAGGED by the lag (0-20 d) that maximizes its correlation with top-bottom salinity at the
    section, so a day is classed by the river water that has actually arrived.
SCENARIOS  terciles of lagged flow (low / mid / high) plus 'flood' = top 5 % of days.

PER DAY
    s_out        surface salinity of upper Saratoga Passage (segment pc_lp_p salt_surf): the source water
    ds_[sect]    top - bottom salinity (bottom minus top layer, width-weighted) at pc_lp (mouth),
                 the analysis section, and pc_cp (head)
    dsurf, dbot  head minus mouth (pc_cp - pc_lp) in the top and bottom layers; dsurf > 0 is the
                 inverted surface gradient, dbot < 0 the classical deep one
    Iv           layered exchange at the section = 0.5 (Q_lower - Q_upper), sigma split at mid-depth,
                 Q + into the cove: + = deep in / surface out (classical), - = reversed
    Qlat         lateral exchange magnitude = 0.5 sum_p |Q_p - Q A_p/A| (column transports about the
                 section mean)
    tau          along-cove wind stress the model applied (+ = up-cove), from pc_forcing_states
    Fr_eff       (ds / (alpha s_ocn))^(3/2) at the section (as 20261109_gm14_Frf_axis.py)

INPUT  LO_output/extract/[gtagex]/tef2/extractions_avg_[ds0]_[ds1]/{pc_cp,[sect],pc_lp}.nc
       LO_output/extract/[gtagex]/tef2/segments_[ds0]_[ds1]_wb1_pc1_trapsN00.nc
       LO_output/DM_outs/20260806_river_hydrographs/daily_flow.csv
       LO_output/DM_outs/20261104_pc_forcing_states/tau_cove.npz
OUTPUT LO_output/DM_outs/20261110_pc_skagit_scenarios/
    daily_[sect].csv, scenarios_[sect].csv, skagit_scenarios_[sect].png

run 20261110_pc_skagit_scenarios.py -sect pc_lj
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', type=str, default='wb1_t0_xn11abbur00')
parser.add_argument('-sect', type=str, default='pc_lj')
args = parser.parse_args()

DS0, DS1 = '2024.01.01', '2025.12.31'
SIGN = -1                                  # wb1_pc1 + is out of the cove; flip so + = into the cove
ALPHA = 3.4
FLOOD_PCT = 95
SECTS = list(dict.fromkeys(['pc_lp', args.sect, 'pc_cp']))     # mouth, analysis section, head

Ldir = Lfun.Lstart(gridname='wb1')
tef = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
dm = Ldir['LOo'] / 'DM_outs'
out_dir = dm / '20261110_pc_skagit_scenarios'
out_dir.mkdir(parents=True, exist_ok=True)
godin_daily = lambda T, v: pd.Series(zfun.lowpass(np.asarray(v, float), f='godin'), index=T).resample('D').mean()

# ------------------------------------------------------------------ daily series
D = {}
for sn in SECTS:
    ds = xr.open_dataset(tef / ('extractions_avg_%s_%s' % (DS0, DS1)) / (sn + '.nc'))
    T = pd.DatetimeIndex(ds.time.values)
    dd = ds.dd.values
    salt = ds.salt.values
    s_top = (salt[:, -1, :] * dd).sum(1) / dd.sum()
    s_bot = (salt[:, 0, :] * dd).sum(1) / dd.sum()
    D['stop_' + sn], D['sbot_' + sn] = godin_daily(T, s_top), godin_daily(T, s_bot)
    D['ds_' + sn] = D['sbot_' + sn] - D['stop_' + sn]
    if sn == args.sect:
        q = SIGN * ds.q.values                                  # (t, z, p), + into the cove
        A_p = (ds.DZ.values * dd).sum(1).mean(0)                # (p,) mean column area
        nz = q.shape[1]
        Qlow, Qup = q[:, :nz // 2, :].sum((1, 2)), q[:, nz // 2:, :].sum((1, 2))
        D['Iv'] = 0.5 * (godin_daily(T, Qlow) - godin_daily(T, Qup))
        Qp = np.column_stack([zfun.lowpass(q[:, :, p].sum(1), f='godin') for p in range(len(dd))])
        Qlat = 0.5 * np.abs(Qp - Qp.sum(1, keepdims=True) * A_p / A_p.sum()).sum(1)
        D['Qlat'] = pd.Series(Qlat, index=T).resample('D').mean()
    ds.close()
D['dsurf'] = D['stop_pc_cp'] - D['stop_pc_lp']                  # head - mouth, top layer
D['dbot'] = D['sbot_pc_cp'] - D['sbot_pc_lp']                   # head - mouth, bottom layer

sg = xr.open_dataset(tef / ('segments_%s_%s_wb1_pc1_trapsN00.nc' % (DS0, DS1)))
D['s_out'] = godin_daily(pd.DatetimeIndex(sg.time.values), sg.salt_surf.sel(seg='pc_lp_p').values)
sg.close()

z = np.load(dm / '20261104_pc_forcing_states' / 'tau_cove.npz')
D['tau'] = godin_daily(pd.DatetimeIndex(z['t']), z['tau'])

X = pd.DataFrame(D).loc[DS0.replace('.', '-'):DS1.replace('.', '-')]
s_ocn = pd.read_csv(dm / '20261109_gm14_M_axis' / ('M_summary_%s.csv' % args.sect), index_col=0).loc['median', 's_ocn']
X['Fr_eff'] = (X['ds_' + args.sect].clip(lower=0) / (ALPHA * s_ocn)) ** 1.5
X = X.dropna()
print('%d complete days; columns: %s' % (len(X), ', '.join(X.columns)))
print(X.describe().loc[['mean', 'min', 'max']].round(3).to_string())

# ------------------------------------------------------------------ Skagit flow, lag, scenarios
Qs = pd.read_csv(dm / '20260806_river_hydrographs' / 'daily_flow.csv', index_col=0, parse_dates=True)['skagit']
lags = np.arange(0, 21)
r = [np.corrcoef(Qs.shift(L).reindex(X.index).dropna(), X['ds_' + args.sect].reindex(Qs.shift(L).dropna().index).dropna())[0, 1]
     if L else np.corrcoef(Qs.reindex(X.index), X['ds_' + args.sect])[0, 1] for L in lags]
LAG = int(lags[np.nanargmax(r)])
print('\ncorr(Skagit lagged, ds_%s): %s' % (args.sect, ' '.join('%d:%.2f' % (L, v) for L, v in zip(lags, r))))
print('best lag %d d (r = %.2f)' % (LAG, max(r)))
X['Q_skagit'] = Qs.shift(LAG).reindex(X.index)
X = X.dropna()

q1, q2, qf = X.Q_skagit.quantile([1 / 3, 2 / 3, FLOOD_PCT / 100])
X['scenario'] = np.select([X.Q_skagit >= qf, X.Q_skagit >= q2, X.Q_skagit >= q1], ['flood', 'high', 'mid'], 'low')
ORDER = ['low', 'mid', 'high', 'flood']
print('class edges (lagged Skagit, m3/s): low < %.0f <= mid < %.0f <= high < %.0f <= flood' % (q1, q2, qf))

cols = ['Q_skagit', 's_out', 'ds_pc_lp', 'ds_' + args.sect, 'ds_pc_cp', 'dsurf', 'dbot', 'Iv', 'Qlat', 'tau', 'Fr_eff']
S = X.groupby('scenario')[cols].mean().loc[ORDER]
S.insert(0, 'n_days', X.groupby('scenario').size().loc[ORDER])
X['reversed'], X['upcove'] = X.Iv < 0, X.tau > 0.01             # up-cove = Godin stress > 0.01 Pa (pc_forcing_states)
S['pct_reversed'] = 100 * X.groupby('scenario').reversed.mean().loc[ORDER]
S['pct_upcove'] = 100 * X.groupby('scenario').upcove.mean().loc[ORDER]
S['Iv_no_upcove'] = X[~X.upcove].groupby('scenario').Iv.mean().loc[ORDER]          # wind confound check
S['pct_rev_no_upcove'] = 100 * X[~X.upcove].groupby('scenario').reversed.mean().loc[ORDER]
months = X.groupby('scenario').apply(lambda g: ' '.join(g.index.strftime('%b').value_counts().index[:3]),
                                     include_groups=False).loc[ORDER]
pd.set_option('display.width', 250)
print('\n' + S.T.round(3).to_string())
print('\ncommonest months: ' + '; '.join('%s %s' % (k, v) for k, v in months.items()))
X.to_csv(out_dir / ('daily_%s.csv' % args.sect))
S.to_csv(out_dir / ('scenarios_%s.csv' % args.sect))

# ------------------------------------------------------------------ figure
C1, C2, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
bins = pd.qcut(X.Q_skagit, 10)
B = X.groupby(bins, observed=True)
Bq = B.Q_skagit.median()
fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')


def frame(ax, title, ylab):
    for e in (q1, q2, qf):
        ax.axvline(e, color=MUTED, lw=0.7, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('Skagit flow, lagged %d d [m$^3$/s]' % LAG)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc='left', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e5e4e0', lw=0.5)


def scen_labels(ax, y):
    edges = [X.Q_skagit.min(), q1, q2, qf, X.Q_skagit.max()]
    for k, lab in enumerate(ORDER):
        ax.text(np.sqrt(edges[k] * edges[k + 1]), y, lab, ha='center', fontsize=8, color=MUTED,
                transform=ax.get_xaxis_transform())


ax = axs[0, 0]
for col, c, lab in (('s_out', C1, 'Saratoga surface salinity (source)'), ('ds_pc_lp', C2, 'top-bottom $\\Delta s$, mouth')):
    ax.plot(X.Q_skagit, X[col], '.', ms=2, color=c, alpha=0.25)
    ax.plot(Bq, B[col].mean(), '-o', color=c, lw=2, ms=5, label=lab)
ax.plot(Bq, B['ds_pc_cp'].mean(), '--o', color=C2, lw=1.5, ms=4, mfc='none', label='top-bottom $\\Delta s$, head')
frame(ax, '(a) more Skagit water: fresher source, stronger stratification', 'g/kg')
scen_labels(ax, 1.01)
ax.legend(fontsize=8, frameon=False, loc='center left')

ax = axs[0, 1]
for col, c, lab in (('dsurf', C1, 'surface, head $-$ mouth (+ = inverted)'), ('dbot', C2, 'bottom, head $-$ mouth ($-$ = classical)')):
    ax.plot(X.Q_skagit, X[col], '.', ms=2, color=c, alpha=0.25)
    ax.plot(Bq, B[col].mean(), '-o', color=c, lw=2, ms=5, label=lab)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_ylim(-1.0, 1.0)
frame(ax, '(b) both along-cove gradients steepen', 'salinity difference [g/kg]')
scen_labels(ax, 1.01)
ax.legend(fontsize=8, frameon=False, loc='lower left')

ax = axs[1, 0]
ax.plot(X.Q_skagit, X.Iv, '.', ms=2, color=MUTED, alpha=0.3)
ax.plot(Bq, B.Iv.mean(), '-o', color=C1, lw=2, ms=5, label='all days')
Bn = X[~X.upcove].groupby(bins[~X.upcove], observed=True)
ax.plot(Bn.Q_skagit.median(), Bn.Iv.mean(), '--o', color=C2, lw=2, ms=5, mfc='none', label='days without up-cove wind')
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_ylim(-400, 600)
frame(ax, '(c) layered exchange at %s (+ = deep in / surface out)' % args.sect, '$I_v$ [m$^3$/s]')
scen_labels(ax, 1.01)
ax.legend(fontsize=8, frameon=False, loc='upper left')

ax = axs[1, 1]
xs = np.arange(len(ORDER))
ax.bar(xs - 0.2, S.pct_reversed, 0.38, color=C1, label='days with layered exchange reversed')
ax.bar(xs + 0.2, S.pct_upcove, 0.38, color=C2, label='days with up-cove wind (> 0.01 Pa)')
ax.set_xticks(xs)
ax.set_xticklabels(['%s\n%.0f m$^3$/s\n(%d d)' % (k, S.Q_skagit[k], S.n_days[k]) for k in ORDER], fontsize=8)
ax.set_ylabel('% of days')
ax.set_title('(d) reversals vs up-cove wind, by scenario', loc='left', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc='upper left')

fig.suptitle('Penn Cove under different Skagit flows (%s, wb1_t0, 2024-2025, Godin daily)' % args.gtx, x=0.01, ha='left', fontsize=11)
fn = out_dir / ('skagit_scenarios_%s.png' % args.sect)
fig.savefig(fn, dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(fn))
