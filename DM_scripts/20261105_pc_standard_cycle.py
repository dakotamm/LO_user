"""
Pick a 'standard' tidal cycle (one flood + one ebb) of Penn Cove as a baseline for a
schematic of the cove and upper Saratoga Passage.

CYCLE  flood start to the next flood start, from the zero up-crossings of the TIDAL
    transport through the mouth (pc_lp), i.e. the utide reconstruction of the
    20261103_pc_tidal_cycles fit without S1 (so the daily wind and the residual do
    not move the slack times). + = into the cove.

CRITERIA (every hour of the cycle, or its mean)
    calm wind     every hour wind_calm: Godin along- and cross-cove stress both
                  within +-0.01 Pa (pc_forcing_states.py)
    medium strat  mean record-wide rank of Godin drho in the middle third
    spring/neap   mean within-month rank of the D2 envelope in the middle third
    trop/equat    mean within-month rank of the D1 envelope in the middle third
                  (both as pc_tidal_cycles.cycle_masks)
RANK of the cycles that pass: distance of the three ranks from 0.5, plus the distance
    of the cycle's peak flood and peak ebb transport from the record medians (so the
    flood and the ebb are both ordinary sizes, not one big and one small half of the
    diurnal inequality).

LUNAR DAYS (2026-11-05)  two consecutive cycles (flood, ebb, flood, ebb; ~25 h), so the
    diurnal inequality is in the picture. BASELINE = every hour calm, and the D2, D1 and
    strat ranks all in the middle third, sorted by their summed distance from 0.5
    (the chosen baseline: 2024-08-31 05:00 -> 09-01 06:00 UTC). Also listed: drho range
    over the day +-24 h (steadiness), calm share over the same +-24 h, and the two
    floods' and two ebbs' peaks with their min/max ratios.

VARIANTS (2026-11-05)  one axis moved at a time, the rest held at baseline:
    spring / neap          D2 rank > 2/3 / <= 1/3        (D1, strat middle third; calm)
    tropic / equatorial    D1 rank > 2/3 / <= 1/3        (D2, strat middle third; calm)
    high / low strat       strat rank > 2/3 / <= 1/3     (D2, D1 middle third; calm)
    up- / down-cove wind   >= 75% of hours Godin along-cove stress > +0.01 / < -0.01 Pa
                           (D2, D1, strat middle third). Cross-cove stress NOT restricted:
                           only 3 of 97 all-day up-cove lunar days have weak cross stress
                           and none of them has the other axes at median; tau_cr is listed.
    ranked by the distance of the held axes from 0.5 plus the distance of the moved axis
    from the MIDDLE of its extreme third (rank 5/6 or 1/6): a typical spring, not the
    most extreme one (the far end of 'high strat' is the Dec 2025 freshet, drho 11);
    for wind, plus (1 - share of hours in the wind class).

OUTPUT LO_output/DM_outs/20261105_pc_standard_cycle/
    lunar_days.csv              every baseline lunar day, most median first
    variants.csv                every lunar day passing each variant, best first
    candidates.csv              every cycle passing the criteria, best first
    candidates.png              the top candidates in context (+-2 d): mouth Q, ranks, drho, stress

run 20261105_pc_standard_cycle.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import utide

from lo_tools import Lfun
from pc_tidal_cycles import box_fn, mouth_fit, envelopes, CONS
from pc_forcing_states import series, classify, nearest, TTH

NTOP = 4
Ldir = Lfun.Lstart(gridname='wb1')
out_dir = Ldir['LOo'] / 'DM_outs' / '20261105_pc_standard_cycle'
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ hourly series on the box time axis
ds = xr.open_dataset(box_fn)
T = pd.DatetimeIndex(ds.ocean_time.values)
coef = mouth_fit()
Qt = utide.reconstruct(T.values, coef, constit=[c for c in CONS if c != 'S1'], verbose=False).h
E = envelopes(T.values)
ym = T.to_period('M')
r2 = pd.Series(E['D2']).groupby(ym).rank(pct=True).to_numpy()
r1 = pd.Series(E['D1']).groupby(ym).rank(pct=True).to_numpy()
X = series()
i_s, i_t = nearest(X['t_s'], T), nearest(X['t_t'], T)
dr = X['drho_lp'][i_s]
rs = pd.Series(dr).rank(pct=True).to_numpy()
ta, tc = X['tau_lp'][i_t], X['tau_cr_lp'][i_t]

# ------------------------------------------------------------------ cycles and criteria
up = np.where((Qt[:-1] <= 0) & (Qt[1:] > 0))[0] + 1
rows = []
for a, b in zip(up[:-1], up[1:]):
    s = slice(a, b)
    if not (np.all(np.isfinite(dr[s])) and np.all(np.isfinite(ta[s]))):
        continue                          # Godin ends
    rows.append(dict(t0=T[a], t1=T[b], hours=b - a, r_D2=r2[s].mean(), r_D1=r1[s].mean(), r_strat=rs[s].mean(),
                     drho=dr[s].mean(), calm=np.mean((np.abs(ta[s]) <= TTH) & (np.abs(tc[s]) <= TTH)),
                     Qflood=Qt[s].max(), Qebb=-Qt[s].min()))
D = pd.DataFrame(rows)
mid = lambda x: x.between(1 / 3, 2 / 3)
D['score'] = (np.abs(D.r_D2 - .5) + np.abs(D.r_D1 - .5) + np.abs(D.r_strat - .5)
              + np.abs(D.Qflood / D.Qflood.median() - 1) + np.abs(D.Qebb / D.Qebb.median() - 1))
ok = D[(D.calm == 1) & mid(D.r_D2) & mid(D.r_D1) & mid(D.r_strat)].sort_values('score')
ok.to_csv(out_dir / 'candidates.csv', index=False, float_format='%.3f')
print('%d cycles, %d pass; record median peak flood %.0f, ebb %.0f m3/s' % (len(D), len(ok), D.Qflood.median(), D.Qebb.median()))
print(ok.round(3).to_string(index=False))

# ------------------------------------------------------------------ lunar days (two cycles), most equatorial first
calm = (np.abs(ta) <= TTH) & (np.abs(tc) <= TTH)
rows = []
for a, m, b in zip(up[:-2], up[1:-1], up[2:]):
    s, w = slice(a, b), slice(max(a - 24, 0), b + 24)
    if not (np.all(np.isfinite(dr[w])) and np.all(np.isfinite(ta[s]))):
        continue
    q1, q2 = Qt[a:m], Qt[m:b]
    rows.append(dict(t0=T[a], t1=T[b], hours=b - a, r_D2=r2[s].mean(), r_D1=r1[s].mean(), r_strat=rs[s].mean(),
                     drho=dr[s].mean(), drho_range_pm1d=np.ptp(dr[w]), calm=calm[s].mean(), calm_pm1d=calm[w].mean(),
                     tau_al=ta[s].mean(), up=np.mean(ta[s] > TTH), down=np.mean(ta[s] < -TTH),
                     cross_weak=np.mean(np.abs(tc[s]) <= TTH), tau_cr=tc[s].mean(),
                     Qflood1=q1.max(), Qebb1=-q1.min(), Qflood2=q2.max(), Qebb2=-q2.min()))
LA = L = pd.DataFrame(rows)
L['flood_ratio'] = L[['Qflood1', 'Qflood2']].min(axis=1) / L[['Qflood1', 'Qflood2']].max(axis=1)
L['ebb_ratio'] = L[['Qebb1', 'Qebb2']].min(axis=1) / L[['Qebb1', 'Qebb2']].max(axis=1)
LA = L.copy()
dist = lambda df, keys: sum(np.abs(df[k] - .5) for k in keys)
L = L[(L.calm == 1) & mid(L.r_D2) & mid(L.r_D1) & mid(L.r_strat)].copy()
L['score'] = dist(L, ['r_D2', 'r_D1', 'r_strat'])
L = L.sort_values('score')
L.to_csv(out_dir / 'lunar_days.csv', index=False, float_format='%.3f')
print('%d baseline lunar days, most median first:' % len(L))
print(L.head(5).round(2).to_string(index=False))

# ------------------------------------------------------------------ variants: one axis moved, the rest at baseline
AX = {'r_D2': 'D2', 'r_D1': 'D1', 'r_strat': 'strat'}
VAR = {'spring': ('r_D2', 1), 'neap': ('r_D2', 0), 'tropic': ('r_D1', 1), 'equatorial': ('r_D1', 0),
       'high strat': ('r_strat', 1), 'low strat': ('r_strat', 0), 'up-cove wind': ('wind', 1), 'down-cove wind': ('wind', 0)}
V = []
for name, (k, end) in VAR.items():
    held = [a for a in AX if a != k]
    m = np.all([mid(LA[a]) for a in held], axis=0)
    if k == 'wind':
        share = LA.up if end else LA.down
        m &= share >= 0.75
        far = 1 - share
    else:
        m &= (LA.calm == 1) & ((LA[k] > 2 / 3) if end else (LA[k] <= 1 / 3))
        far = np.abs(LA[k] - (5 / 6 if end else 1 / 6))
    c = LA[m].assign(variant=name, score=dist(LA[m], held) + far[m]).sort_values('score')
    V.append(c)
    print('%-15s %3d pass' % (name, len(c)))
V = pd.concat(V)
cols = ['variant', 't0', 't1', 'r_D2', 'r_D1', 'r_strat', 'drho', 'drho_range_pm1d', 'tau_al', 'tau_cr', 'calm_pm1d',
        'Qflood1', 'Qebb1', 'Qflood2', 'Qebb2', 'score']
V[cols].to_csv(out_dir / 'variants.csv', index=False, float_format='%.3f')
pd.set_option('display.width', 250)
print(V.groupby('variant', sort=False).head(3)[cols].round(2).to_string(index=False))

# ------------------------------------------------------------------ check figure: top candidates in context
XI_U, ETA = 30, np.arange(19, 7, -1)     # pc_lp u faces (pc_tidal_cycles.mouth_fit)
dy = (1 / ds.pn.isel(eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy()).mean(axis=-1)
fig, axs = plt.subplots(4, NTOP, figsize=(4.5 * NTOP, 10), layout='constrained', sharey='row')
for n, (_, c) in enumerate(ok.head(NTOP).iterrows()):
    w = (T >= c.t0 - pd.Timedelta('2D')) & (T <= c.t1 + pd.Timedelta('2D'))
    iw = np.where(w)[0]
    u = -ds.u.isel(ocean_time=slice(iw[0], iw[-1] + 1), xi_u=XI_U, eta_u=ETA).to_numpy().astype(float)
    zw = ds.z_w.isel(ocean_time=slice(iw[0], iw[-1] + 1), eta_rho=ETA, xi_rho=[XI_U, XI_U + 1]).to_numpy().mean(axis=-1)
    Q = (u * np.diff(zw, axis=1) * dy[None, None, :]).sum(axis=(1, 2))
    tw = T[w]
    ax = axs[0, n]
    ax.plot(tw, Q, color='k', lw=1, label='model Q (hourly snapshot)')
    ax.plot(tw, Qt[w], color='#0072B2', lw=1, ls='--', label='tidal Q (utide, no S1)')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title('%s to %s UTC\nflood %.0f, ebb %.0f m$^3$ s$^{-1}$' % (c.t0.strftime('%Y-%m-%d %H:%M'), c.t1.strftime('%H:%M'), c.Qflood, c.Qebb), fontsize=9)
    ax = axs[1, n]
    ax.plot(tw, r2[w], color='#D55E00', label='D2 rank (spring/neap)')
    ax.plot(tw, r1[w], color='#009E73', label='D1 rank (tropic/equatorial)')
    ax.plot(tw, rs[w], color='#CC79A7', label='strat rank')
    for y in [1 / 3, 2 / 3]:
        ax.axhline(y, color='k', lw=0.5, ls=':')
    ax.set_ylim(0, 1)
    ax = axs[2, n]
    ax.plot(tw, dr[w], color='k')
    ax = axs[3, n]
    ax.plot(tw, ta[w], color='k', label='along (+ up-cove)')
    ax.plot(tw, tc[w], color='0.5', label='cross (+ northward)')
    ax.plot(X['t_t'][i_t][w], X['tau'][i_t][w], color='k', lw=0.4, alpha=0.5)
    for y in [-TTH, TTH]:
        ax.axhline(y, color='k', lw=0.5, ls=':')
    for a_ in axs[:, n]:
        a_.axvspan(c.t0, c.t1, color='#E69F00', alpha=0.25, lw=0)
        a_.grid(True, alpha=0.3)
        a_.tick_params(axis='x', labelsize=7, rotation=30)
ds.close()
axs[0, 0].set_ylabel('mouth Q [m$^3$ s$^{-1}$], + into cove')
axs[1, 0].set_ylabel('rank (0-1); middle third dotted')
axs[2, 0].set_ylabel(r'Godin $\Delta\sigma_0$ [kg m$^{-3}$]')
axs[3, 0].set_ylabel('Godin wind stress [Pa]\n(thin: hourly along)')
for r in range(4):
    axs[r, 0].legend(fontsize=7, loc='upper left') if r != 2 else None
fig.suptitle('Penn Cove standard-cycle candidates (best first; shaded = the cycle)', fontsize=11)
fig.savefig(out_dir / 'candidates.png', dpi=200, transparent=True)
plt.close(fig)
print('saved ' + str(out_dir / 'candidates.png'))
