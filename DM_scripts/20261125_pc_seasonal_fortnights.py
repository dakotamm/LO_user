"""
Penn Cove seasonal fortnights: one representative spring-neap cycle per season
(winter vs late summer, the low-DO season), to carry into the velocity
cross-sections (pc_cp | pc_lj | pc_lp) and the cove velocity maps (zoomed out to
the head of Saratoga Passage). Built step by step.

Input (wb1_t0_xn11abbur00, 2024-2025):
    LO_output/DM_outs/20261112_pc_ind_var_ts/ind_var_ts.p             hourly forcing
    LO_output/DM_outs/20261117_pc_saratoga_strat_DO/
        wb1_t0_xn11abbur00_pc_west-M5_2024.01.02_2025.12.30.p        daily (Godin) bottom DO
Output: LO_output/DM_outs/20261125_pc_seasonal_fortnights/

STEP 1: read the forcing, one series per axis.
    hourly (from ind_var_ts.p):
    D2      [m3/s]  spring/neap: envelope of the semidiurnal tidal transport at pc_lp
                    (sets the fortnight; semidiurnal, not the ssh envelope, which
                    blends in the 13.66 d tropic cycle)
    D1      [m3/s]  tropic/equatorial: envelope of the diurnal tidal transport
    drho    [kg/m3] mean of drho_pc_cp, drho_pc_lj, drho_pc_lp, Godin-filtered
    tau_al  [Pa]    along-cove wind stress, + = up-cove, Godin-filtered
    tau_mag [Pa]    |tau| = sqrt(tau_ew^2 + tau_ns^2), Godin-filtered: wind strength
                    whatever the direction (opposing events cancel in tau_al)
    daily (12:00 UTC, lowpassed.nc-equivalent, mg/L):
    DO_pcw  bottom DO at pc_west (inner cove, h 17.2 m)
    DO_m5   bottom DO at M5 (Saratoga Passage, h 37.4 m)

run 20261125_pc_seasonal_fortnights.py
"""
import pickle

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']
DO_FN = '20261117_pc_saratoga_strat_DO/wb1_t0_xn11abbur00_pc_west-M5_2024.01.02_2025.12.30.p'

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
in_dir = Ldir['LOo'] / 'DM_outs'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261125_pc_seasonal_fortnights'
Lfun.make_dir(out_dir)

di = pd.read_pickle(in_dir / '20261112_pc_ind_var_ts' / 'ind_var_ts.p')

dh = pd.DataFrame(index=di.index)
dh['D2'] = di['D2']
dh['D1'] = di['D1']
dh['drho'] = zfun.lowpass(di[['drho_' + sn for sn in SECTS]].mean(axis=1).values, f='godin')
dh['tau_al'] = zfun.lowpass(di['tau_al'].values, f='godin')
dh['tau_mag'] = zfun.lowpass(np.sqrt(di['tau_ew']**2 + di['tau_ns']**2).values, f='godin')

D = pickle.load(open(in_dir / DO_FN, 'rb'))
dd = pd.DataFrame(index=pd.to_datetime(D['time']))
dd['DO_pcw'] = D['st']['pc_west']['oxygen'][:, 0]
dd['DO_m5'] = D['st']['M5']['oxygen'][:, 0]

print('STEP 1: hourly %d, %s to %s; daily DO %d, %s to %s' % (len(dh), dh.index[0], dh.index[-1], len(dd), dd.index[0].date(), dd.index[-1].date()))
print(pd.concat([dh.describe().T, dd.describe().T])[['count', 'mean', 'std', 'min', '50%', 'max']].to_string(float_format='%.3g'))

"""
STEP 2: cut the record into fortnights, neap to neap.
    Neap = local minimum of D2 (find_peaks on -D2, at least NSPACE days apart; the
    semidiurnal envelope has one minimum per 14.77 d). Fortnight = neap -> next neap,
    so the spring is in the middle and every fortnight holds one whole spring-neap
    cycle (a sliding 14-day window would mix tidal phase into the comparison).
    Dropped: fortnights touching the Godin NaN ends, or missing any daily DO.
    Season = SEASONS month list containing the fortnight's middle day; only middles
    in YEARS are kept (2024 for now, as the candidate days). Others are labelled
    'other' (kept for the figure, never picked).
    Per fortnight (t0 <= time < t1):
        days                                length
        D2, D1, drho, tau_al, tau_mag       means of the hourly series
        D2_sp, D2_np                        spring peak (max) and the mean of the two
                                            neap ends: the size of this cycle
        drho_range                          max - min of the Godin drho
        DO_pcw, DO_m5                       mean daily bottom DO [mg/L]
        DO_pcw_min, hyp_days                min, and days with DO_pcw < HYP
    Saved: fortnights.p
"""
NSPACE = 10                                  # days between neaps, minimum
SEASONS = {'winter': [12, 1, 2], 'late summer': [8, 9]}
YEARS = [2024]
HYP = 2.0                                    # mg/L

neap, _ = find_peaks(-dh.D2.values, distance=NSPACE * 24)
T = dh.index

rows = []
for a, b in zip(neap[:-1], neap[1:]):
    d = dh.iloc[a:b]
    o = dd[(dd.index >= T[a]) & (dd.index < T[b])]
    if d.isna().any().any() or o.isna().any().any() or len(o) < (b - a) // 24:
        continue
    tmid = T[a] + (T[b] - T[a]) / 2
    season = next((s for s, mm in SEASONS.items() if tmid.month in mm), 'other')
    if tmid.year not in YEARS:
        season = 'other'
    rows.append(dict(t0=T[a], t1=T[b], tmid=tmid, season=season, days=(b - a) / 24,
                     D2=d.D2.mean(), D1=d.D1.mean(), drho=d.drho.mean(), tau_al=d.tau_al.mean(), tau_mag=d.tau_mag.mean(),
                     D2_sp=d.D2.max(), D2_np=(dh.D2.iloc[a] + dh.D2.iloc[b]) / 2,
                     drho_range=d.drho.max() - d.drho.min(),
                     DO_pcw=o.DO_pcw.mean(), DO_m5=o.DO_m5.mean(),
                     DO_pcw_min=o.DO_pcw.min(), hyp_days=int((o.DO_pcw < HYP).sum())))
fn = pd.DataFrame(rows)
fn.to_pickle(out_dir / 'fortnights.p')

print('\nSTEP 2: %d neaps, %d whole fortnights; length %.1f-%.1f d' % (len(neap), len(fn), fn.days.min(), fn.days.max()))
print('per season (middle day in %s): %s' % (YEARS, fn.season.value_counts().to_dict()))
print(fn[fn.season != 'other'].drop(columns=['t1', 'tmid']).to_string(index=False, float_format=lambda x: '%.3g' % x))
print('saved %s' % (out_dir / 'fortnights.p'))

"""
STEP 3: score each season's fortnights by distance from the season's middle.
    z_k = (x_k - median of x_k over the season's fortnights) / std of x_k over ALL
    fortnights (record-wide std: 4-6 fortnights per season is too few for a stable
    within-season std, and one scale keeps the scores comparable across seasons).
    score = sum of |z_k| over AXES (all weight 1); best = smallest = the season's
    medoid-like fortnight, typical on every axis at once rather than on one.
    AXES:
        D2_sp     spring size (consecutive springs alternate big/small with the 27.6 d
                  perigee cycle; the fortnight mean D2 hides this)
        D1        diurnal strength
        drho      stratification
        tau_al    mean along-cove wind
        tau_mag   wind strength
        DO_pcw    inner-cove bottom DO
    PICK overrides the best per season with a t0 string (empty = take rank 1).
    Saved: ranked.csv (season, rank, score, z_<axis>, the fortnights.p columns),
           picks.csv  (one row per season)
"""
AXES = ['D2_sp', 'D1', 'drho', 'tau_al', 'tau_mag', 'DO_pcw']
PICK = {'winter': '2024-01-21 01:00',        # rank 2: 14.2 d (rank 1 Dec is 16.8 d, DO still recovering); steady up-cove wind, DO 7.9
        'late summer': '2024-08-28 17:00'}   # rank 4: 99% calm, DO minimum (0.59 mean, -> ~0); holds the 2024-08-31 standard-cycle day

sd = fn[AXES].std()
out = []
for season in SEASONS:
    s = fn[fn.season == season].copy()
    z = (s[AXES] - s[AXES].median()) / sd
    s['score'] = z.abs().sum(axis=1)
    for k in AXES:
        s['z_' + k] = z[k]
    s = s.sort_values('score')
    s.insert(0, 'rank', np.arange(1, len(s) + 1))
    out.append(s)
rk = pd.concat(out, ignore_index=True)
rk.to_csv(out_dir / 'ranked.csv', index=False, float_format='%.4g')

picks = []
for season in SEASONS:
    s = rk[rk.season == season]
    picks.append(s[s.t0 == pd.Timestamp(PICK[season])] if PICK.get(season) else s.iloc[:1])
picks = pd.concat(picks, ignore_index=True)
picks.to_csv(out_dir / 'picks.csv', index=False, float_format='%.4g')

print('\nSTEP 3: axis std over all %d fortnights: %s' % (len(fn), sd.round(3).to_dict()))
for season in SEASONS:
    s = rk[rk.season == season]
    print('\n%s (%d):' % (season, len(s)))
    print(s[['rank', 't0', 't1', 'score'] + ['z_' + k for k in AXES]].to_string(index=False, float_format=lambda x: '%+.2f' % x))
print('\npicks:')
print(picks[['season', 'rank', 't0', 't1', 'days'] + AXES + ['hyp_days']].to_string(index=False, float_format=lambda x: '%.3g' % x))
print('saved %s, %s' % (out_dir / 'ranked.csv', out_dir / 'picks.csv'))

"""
STEP 4: context figures.
    year.png    2024, shared time axis, rows:
                    D2 and D1 (neaps marked), drho, tau_al and tau_cr (Godin; tau_cr + = toward
                    the north shore), bottom DO at pc_west and M5 (HYP dashed)
                every season fortnight shaded light in its season colour, the picks dark
    picks.png   the two picks side by side (columns), x = days from the starting neap,
                same y limits across a row so the seasons compare directly; rows:
                    Q_tide hourly (+ = into the cove; ind_var_ts has + = out) with D2, D1
                    drho, tau_al / tau_cr, bottom DO pc_west / M5
"""
SCOL = {'winter': 'tab:blue', 'late summer': 'tab:orange'}
dh['tau_cr'] = zfun.lowpass(di['tau_cr'].values, f='godin')
dh['Qf'] = -di['Q_tide']
yr = (dh.index.year == 2024)
yd = (dd.index.year == 2024)

plt.close('all')
fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
ax = axs[0]
ax.plot(dh.index[yr], dh.D2[yr], c='k', lw=1, label='D2 (semidiurnal)')
ax.plot(dh.index[yr], dh.D1[yr], c='0.55', lw=1, label='D1 (diurnal)')
nn = [n for n in neap if T[n].year == 2024]
ax.plot(T[nn], dh.D2.values[nn], 'v', c='k', ms=4, label='neap')
ax.set_ylabel('tidal Q envelope [m$^3$ s$^{-1}$]')
ax = axs[1]
ax.plot(dh.index[yr], dh.drho[yr], c='k', lw=1)
ax.set_ylabel(r'$\Delta\rho$ [kg m$^{-3}$]')
ax = axs[2]
ax.axhline(0, c='0.7', lw=0.5)
ax.plot(dh.index[yr], dh.tau_al[yr], c='k', lw=1, label=r'$\tau_{al}$ (+ up-cove)')
ax.plot(dh.index[yr], dh.tau_cr[yr], c='0.55', lw=1, label=r'$\tau_{cr}$ (+ to north shore)')
ax.set_ylabel('wind stress [Pa]')
ax = axs[3]
ax.axhline(HYP, c='k', ls='--', lw=0.8)
ax.plot(dd.index[yd], dd.DO_pcw[yd], c='k', lw=1, label='pc_west')
ax.plot(dd.index[yd], dd.DO_m5[yd], c='0.55', lw=1, label='M5')
ax.set_ylabel('bottom DO [mg L$^{-1}$]')
for ax in axs:
    for _, r in fn[fn.season.isin(SEASONS)].iterrows():
        ax.axvspan(r.t0, r.t1, color=SCOL[r.season], alpha=0.12, lw=0)
    for _, r in picks.iterrows():
        ax.axvspan(r.t0, r.t1, color=SCOL[r.season], alpha=0.35, lw=0)
    ax.grid(c='0.9', lw=0.5)
    ax.legend(loc='upper right', fontsize=8, frameon=False) if ax.get_legend_handles_labels()[0] else None
for _, r in picks.iterrows():
    axs[0].text(r.tmid, axs[0].get_ylim()[1], r.season, ha='center', va='bottom', color=SCOL[r.season], fontsize=9)
axs[-1].set_xlim(pd.Timestamp('2024-01-01'), pd.Timestamp('2025-01-01'))
fig.suptitle('%s: seasonal fortnights (light = candidates, dark = picks)' % GTAGEX, y=0.995)
fig.tight_layout()
fig.savefig(out_dir / 'year.png', dpi=200, transparent=True)
plt.close(fig)

fig, axs = plt.subplots(4, len(picks), figsize=(12, 10), sharex=True, sharey='row', squeeze=False)
for c, (_, r) in enumerate(picks.iterrows()):
    h = dh[(dh.index >= r.t0) & (dh.index < r.t1)]
    o = dd[(dd.index >= r.t0) & (dd.index < r.t1)]
    xh = (h.index - r.t0) / pd.Timedelta('1D')
    xd = (o.index - r.t0) / pd.Timedelta('1D')
    ax = axs[0, c]
    ax.axhline(0, c='0.7', lw=0.5)
    ax.plot(xh, h.Qf, c=SCOL[r.season], lw=0.6, label='Q_tide (+ flood)')
    ax.plot(xh, h.D2, c='k', lw=1, label='D2')
    ax.plot(xh, h.D1, c='0.55', lw=1, label='D1')
    ax.set_title('%s: %s to %s' % (r.season, r.t0.strftime('%Y-%m-%d %H:%M'), r.t1.strftime('%m-%d %H:%M')), color=SCOL[r.season])
    ax = axs[1, c]
    ax.plot(xh, h.drho, c='k', lw=1)
    ax = axs[2, c]
    ax.axhline(0, c='0.7', lw=0.5)
    ax.plot(xh, h.tau_al, c='k', lw=1, label=r'$\tau_{al}$ (+ up-cove)')
    ax.plot(xh, h.tau_cr, c='0.55', lw=1, label=r'$\tau_{cr}$ (+ to north shore)')
    ax = axs[3, c]
    ax.axhline(HYP, c='k', ls='--', lw=0.8)
    ax.plot(xd, o.DO_pcw, 'o-', c='k', ms=3, lw=1, label='pc_west')
    ax.plot(xd, o.DO_m5, 'o-', c='0.55', ms=3, lw=1, label='M5')
    ax.set_xlabel('days from neap')
    for ax in axs[:, c]:
        ax.grid(c='0.9', lw=0.5)
for ax in axs[:, 0]:
    ax.legend(loc='upper right', fontsize=8, frameon=False) if ax.get_legend_handles_labels()[0] else None
axs[0, 0].set_ylabel('tidal Q [m$^3$ s$^{-1}$]')
axs[1, 0].set_ylabel(r'$\Delta\rho$ [kg m$^{-3}$]')
axs[2, 0].set_ylabel('wind stress [Pa]')
axs[3, 0].set_ylabel('bottom DO [mg L$^{-1}$]')
fig.tight_layout()
fig.savefig(out_dir / 'picks.png', dpi=200, transparent=True)
plt.close(fig)
print('\nSTEP 4: saved %s, %s' % (out_dir / 'year.png', out_dir / 'picks.png'))
