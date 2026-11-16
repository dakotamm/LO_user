"""
Penn Cove candidate days: lunar days of the model run that vary one forcing at a time
(spring/neap, tropic/equatorial, stratification, wind), with the others held at
their middle. Built step by step.

Input (wb1_t0_xn11abbur00, 2024-2025, hh:00 his snapshots):
    LO_output/DM_outs/20261112_pc_ind_var_ts/ind_var_ts.p

STEP 1: read the independent variables and make one series per axis, hourly.
    D2      [m3/s]  spring/neap: envelope of the semidiurnal tidal transport at pc_lp
    D1      [m3/s]  tropic/equatorial: envelope of the diurnal tidal transport
                    (both harmonic predictions already, so no filtering)
    drho    [kg/m3] stratification: mean of drho_pc_cp, drho_pc_lj, drho_pc_lp
                    (sigma0 bottom - top), Godin-filtered so tidal straining does not
                    move a day's value with the time of day it starts
    tau_al  [Pa]    along-cove wind stress, + = up-cove (toward the head), Godin-filtered
    tau_cr  [Pa]    cross-cove wind stress, + = toward the north shore, Godin-filtered
                    (not an axis; kept to hold the wind days calm across the cove too)
    Godin leaves 35 h at each end NaN.

run 20261113_pc_candidate_days.py
"""
import numpy as np
import pandas as pd

from lo_tools import Lfun, zfun

GTAGEX = 'wb1_t0_xn11abbur00'
SECTS = ['pc_cp', 'pc_lj', 'pc_lp']

gridname, tag, ex_name = GTAGEX.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
in_dir = Ldir['LOo'] / 'DM_outs'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261113_pc_candidate_days'
Lfun.make_dir(out_dir)

di = pd.read_pickle(in_dir / '20261112_pc_ind_var_ts' / 'ind_var_ts.p')

dh = pd.DataFrame(index=di.index)
dh['D2'] = di['D2']
dh['D1'] = di['D1']
dh['drho'] = zfun.lowpass(di[['drho_' + sn for sn in SECTS]].mean(axis=1).values, f='godin')
dh['tau_al'] = zfun.lowpass(di['tau_al'].values, f='godin')
dh['tau_cr'] = zfun.lowpass(di['tau_cr'].values, f='godin')

print('%d hours, %s to %s' % (len(dh), dh.index[0], dh.index[-1]))
print(dh.describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].to_string(float_format='%.3g'))

"""
STEP 2: cut the record into lunar days and give each one row of forcing values.
    Lunar day = two tidal cycles, flood start -> flood start -> flood start, from the
    zero up-crossings of the flood transport -Q_tide (+ = into the cove; ind_var_ts.p
    has + = out). Two cycles so both floods and both ebbs of the mixed tide are in
    every day (one cycle would pick out the greater or the lesser half). A day starts
    at every flood start, so consecutive days overlap by one cycle; the choice of
    which to keep is left to the selection.
    Days touching the Godin NaN ends are dropped, and so are days not 24-26 h long:
    those have a lesser flood or ebb that never reversed Q_tide (no zero crossing), so
    they are not two floods + two ebbs.
    Only days whose middle hour is in YEAR are kept (2026-11-13: 2024 for now), so every
    rank and tercile below is within that year.
    Per day (all over the day's hours, t0 <= time < t1):
        D2, D1, drho, tau_al, tau_cr    means
        drho_range                      max - min of the Godin drho (steadiness)
        calm                            share of hours with |tau_al| and |tau_cr| <= TTH
        up, down                        share of hours with tau_al > +TTH / < -TTH
        Qf1, Qe1, Qf2, Qe2              peak flood / ebb |Q_tide| of the first and second
                                        cycle [m3/s], to match the tide order between days
    TTH = 0.01 Pa (~3 m/s of steady wind), the threshold of pc_forcing_states.py.
    Saved: days.p
"""
TTH = 0.01
YEAR = 2024

qf = -di['Q_tide'].values                               # + = flood
up = np.nonzero((qf[:-1] <= 0) & (qf[1:] > 0))[0] + 1    # first hour of each flood
T = dh.index

rows = []
for a, m, b in zip(up[:-2], up[1:-1], up[2:]):
    d = dh.iloc[a:b]
    if d.isna().any().any():
        continue
    rows.append(dict(t0=T[a], t1=T[b], hours=b - a,
                     D2=d.D2.mean(), D1=d.D1.mean(), drho=d.drho.mean(), tau_al=d.tau_al.mean(), tau_cr=d.tau_cr.mean(),
                     drho_range=d.drho.max() - d.drho.min(),
                     calm=np.mean((d.tau_al.abs() <= TTH) & (d.tau_cr.abs() <= TTH)),
                     up=np.mean(d.tau_al > TTH), down=np.mean(d.tau_al < -TTH),
                     Qf1=qf[a:m].max(), Qe1=-qf[a:m].min(), Qf2=qf[m:b].max(), Qe2=-qf[m:b].min()))
days = pd.DataFrame(rows)
nall = len(days)
print('STEP 2: day length [h] before the 24-26 h cut: %s' % days.hours.value_counts().sort_index().to_dict())
days = days[days.hours.between(24, 26)].reset_index(drop=True)
days = days[(days.t0 + (days.t1 - days.t0) / 2).dt.year == YEAR].reset_index(drop=True)
days.to_pickle(out_dir / 'days.p')

print('%d flood starts, %d lunar days, %d kept (24-26 h, %d)' % (len(up), nall, len(days), YEAR))
print('all-calm days %d, all-up-cove %d, all-down-cove %d' % ((days.calm == 1).sum(), (days.up == 1).sum(), (days.down == 1).sum()))
print(days.drop(columns=['t0', 't1']).describe().T[['mean', 'std', 'min', '50%', 'max']].to_string(float_format='%.3g'))
print('saved %s' % (out_dir / 'days.p'))

"""
STEP 3: rank every day on each axis.
    r_D2, r_D1   percentile (0-1) of the day's mean D2 / D1 among the days of the same
                 calendar month of the same year (month of the day's middle hour). The
                 envelopes have a semiannual modulation (strongest tropic tides at the
                 solstices, strongest springs near the equinoxes), so a record-wide rank
                 would compare seasons; within the month, 'spring' is the stronger third
                 of the tides at that time of year, as in pc_tidal_cycles.py.
    r_strat      percentile (0-1) of the day's mean drho over the whole record: the
                 seasonal swing of stratification is the contrast wanted.
    wind         class from the hour shares of STEP 2, not a percentile:
                     'up'    up   >= 0.75   (>= 75% of hours tau_al > +TTH)
                     'down'  down >= 0.75
                     'calm'  calm == 1      (every hour weak along AND across the cove)
                     'mixed' anything else
    Thirds: low <= 1/3 < middle <= 2/3 < high.
    Saved: days.p (adds the columns)
"""
WSHARE = 0.75

tmid = days.t0 + (days.t1 - days.t0) / 2
ym = tmid.dt.to_period('M')
days['r_D2'] = days.D2.groupby(ym).rank(pct=True)
days['r_D1'] = days.D1.groupby(ym).rank(pct=True)
days['r_strat'] = days.drho.rank(pct=True)

days['wind'] = 'mixed'
days.loc[days.calm == 1, 'wind'] = 'calm'
days.loc[days.up >= WSHARE, 'wind'] = 'up'
days.loc[days.down >= WSHARE, 'wind'] = 'down'
days.to_pickle(out_dir / 'days.p')

third = lambda r: pd.cut(r, [0, 1 / 3, 2 / 3, 1], labels=['low', 'mid', 'high'])
mon = tmid.dt.month
print('STEP 3: wind classes %s' % days.wind.value_counts().to_dict())
print('days per calendar month in each class:')
print(pd.DataFrame({'spring': (third(days.r_D2) == 'high').groupby(mon).sum(),
                    'neap': (third(days.r_D2) == 'low').groupby(mon).sum(),
                    'tropic': (third(days.r_D1) == 'high').groupby(mon).sum(),
                    'equat': (third(days.r_D1) == 'low').groupby(mon).sum(),
                    'hi_strat': (third(days.r_strat) == 'high').groupby(mon).sum(),
                    'lo_strat': (third(days.r_strat) == 'low').groupby(mon).sum(),
                    'up': (days.wind == 'up').groupby(mon).sum(),
                    'down': (days.wind == 'down').groupby(mon).sum(),
                    'calm': (days.wind == 'calm').groupby(mon).sum()}).T.to_string())
print('drho tercile limits: %.2f, %.2f kg/m3' % tuple(days.drho.quantile([1 / 3, 2 / 3])))
mid3 = (third(days.r_D2) == 'mid') & (third(days.r_D1) == 'mid') & (third(days.r_strat) == 'mid')
print('days with D2, D1, strat all middle third: %d; of those by wind %s' % (mid3.sum(), days.wind[mid3].value_counts().to_dict()))
print('saved %s' % (out_dir / 'days.p'))

"""
STEP 4: pick the baseline and the variants (one axis moved, the others held).
    Stratification and wind are seasonal (STEP 3: high strat Nov-Dec, low strat Aug-Oct,
    up-cove wind only Oct-Mar, down-cove Apr-Aug), so a variant is matched on the ranks,
    not on the date; its month is listed so the season it comes from is in view.
    BASELINE    wind 'calm'; r_D2, r_D1, r_strat all in the middle third;
                score = sum of |r - 1/2| over the three.
    VARIANTS    the moved axis in its extreme third, aimed at the MIDDLE of that third
                (rank 5/6 or 1/6: a typical spring, not the most extreme one); the other
                axes as the baseline:
                    spring / neap          r_D2 high / low        (r_D1, r_strat mid; calm)
                    tropic / equatorial    r_D1 high / low        (r_D2, r_strat mid; calm)
                    high / low strat       r_strat high / low     (r_D2, r_D1 mid; calm)
                    up- / down-cove wind   wind 'up' / 'down'     (r_D2, r_D1, r_strat mid)
                score = |r_moved - target| + sum of |r_held - 1/2|; for wind
                score = sum of |r_held - 1/2| + (1 - share of hours in the wind class).
                Cross-cove stress is not restricted on the wind days; tau_cr is listed.
    Overlapping days (consecutive days share a cycle) are thinned best first: a day is
    dropped if it overlaps one already kept. NTOP per case are kept.
    Saved: candidates.csv (case, n = 1 best, then the days.p columns)
"""
NTOP = 5
HI, LO = 5 / 6, 1 / 6
R = ['r_D2', 'r_D1', 'r_strat']

def mid(r):
    return (r > 1 / 3) & (r <= 2 / 3)

def held(d, keys):
    """sum of |r - 1/2| over the held axes."""
    return sum((d[k] - 0.5).abs() for k in keys)

def thin(d):
    """d sorted by score; keep the best, drop any later day overlapping a kept one; NTOP rows."""
    keep = []
    for i, row in d.iterrows():
        if not any((row.t0 < d.t1[j]) & (row.t1 > d.t0[j]) for j in keep):
            keep.append(i)
        if len(keep) == NTOP:
            break
    return d.loc[keep]

CASES = {}      # case: (pass mask, score)
CASES['baseline'] = ((days.wind == 'calm') & mid(days.r_D2) & mid(days.r_D1) & mid(days.r_strat), held(days, R))
for case, rk, target in [('spring', 'r_D2', HI), ('neap', 'r_D2', LO),
                         ('tropic', 'r_D1', HI), ('equatorial', 'r_D1', LO),
                         ('high strat', 'r_strat', HI), ('low strat', 'r_strat', LO)]:
    others = [k for k in R if k != rk]
    ok = (days.wind == 'calm') & (days[rk] > 2 / 3 if target == HI else days[rk] <= 1 / 3)
    for k in others:
        ok &= mid(days[k])
    CASES[case] = (ok, (days[rk] - target).abs() + held(days, others))
for case, w in [('up-cove', 'up'), ('down-cove', 'down')]:
    ok = (days.wind == w) & mid(days.r_D2) & mid(days.r_D1) & mid(days.r_strat)
    CASES[case] = (ok, held(days, R) + (1 - days[w]))

COLS = ['t0', 't1', 'score', 'r_D2', 'r_D1', 'r_strat', 'drho', 'drho_range', 'tau_al', 'tau_cr', 'calm', 'up', 'down', 'Qf1', 'Qe1', 'Qf2', 'Qe2']
out = []
print('STEP 4:')
for case, (ok, score) in CASES.items():
    d = days[ok].assign(score=score[ok]).sort_values('score')
    c = thin(d)[COLS]
    c.insert(0, 'n', np.arange(1, len(c) + 1))
    c.insert(0, 'case', case)
    out.append(c)
    print('\n%s: %d days pass, %d kept; months %s' % (case, ok.sum(), len(c), ', '.join(c.t0.dt.strftime('%b %Y'))))
    print(c.drop(columns=['case', 't1', 'calm', 'up', 'down']).to_string(index=False, float_format=lambda x: '%.3g' % x))
cand = pd.concat(out, ignore_index=True)
cand.to_csv(out_dir / 'candidates.csv', index=False, float_format='%.4g')
print('\nsaved %s' % (out_dir / 'candidates.csv'))

"""
STEP 5: hold the tide ORDER of the baseline in every variant.
    The mixed tide makes the two floods of a lunar day unequal, and the two ebbs too.
    order = 4 letters, first flood, first ebb, second flood, second ebb; capital = the
    bigger of its pair (from Qf1/Qf2, Qe1/Qe2). The baseline (STEP 4, n = 1) sets ORDER;
    every case (baseline alternates included) is re-picked from its STEP 4 passing days
    with days.order == ORDER, same score, thinned, NTOP kept. Without this, a variant
    day and the baseline would line up hour for hour with a big flood against a small
    one.
    Saved: candidates_order.csv
"""
fo = np.where(days.Qf1 > days.Qf2, 'F', 'f')
eo = np.where(days.Qe1 > days.Qe2, 'E', 'e')
days['order'] = [a + b + a.swapcase() + b.swapcase() for a, b in zip(fo, eo)]
ORDER = days.order[days.t0 == cand.t0[(cand.case == 'baseline') & (cand.n == 1)].iloc[0]].iloc[0]

out = []
print('STEP 5: baseline order %s; record shares %s' % (ORDER, days.order.value_counts(normalize=True).round(2).to_dict()))
for case, (ok, score) in CASES.items():
    ok = ok & (days.order == ORDER)
    d = days[ok].assign(score=score[ok]).sort_values('score')
    c = thin(d)[COLS + ['order']]
    c.insert(0, 'n', np.arange(1, len(c) + 1))
    c.insert(0, 'case', case)
    out.append(c)
    print('%-11s %2d pass, %d kept; best %s  r_D2 %.2f r_D1 %.2f r_strat %.2f tau_al %+.3f tau_cr %+.3f; months %s'
          % (case, ok.sum(), len(c), *(['-'] + [np.nan] * 5 if c.empty else
             [c.t0.iloc[0].strftime('%Y-%m-%d %H:%M')] + [c[k].iloc[0] for k in ['r_D2', 'r_D1', 'r_strat', 'tau_al', 'tau_cr']]),
             ', '.join(c.t0.dt.strftime('%b %Y'))))
cord = pd.concat(out, ignore_index=True)
cord.to_csv(out_dir / 'candidates_order.csv', index=False, float_format='%.4g')
print('saved %s' % (out_dir / 'candidates_order.csv'))

"""
STEP 6: the chosen day of each case = the best order-matched day (n = 1), unless a hand
pick in PICK overrides it (must be one of the case's order-matched candidates):
    spring    2024-09-02 07:00 (n = 3): the n = 1 day (2025-06-22) has drho ramping
              2.2 -> 4.2 right after it and down-cove wind 2 d before; this one is
              steady, calm, and the same season as the baseline.
    Saved: picks.csv (one row per case, the columns of candidates_order.csv)
"""
PICK = {'spring': '2024-09-02 07:00'}

best = []
for case in CASES:
    c = cord[cord.case == case]
    best.append(c[c.t0 == pd.Timestamp(PICK[case])] if case in PICK else c[c.n == 1])
    assert len(best[-1]) == 1, case
best = pd.concat(best, ignore_index=True)
best.to_csv(out_dir / 'picks.csv', index=False, float_format='%.4g')
print('STEP 6: picks\n' + best[['case', 'n', 't0', 'r_D2', 'r_D1', 'r_strat', 'drho_range', 'tau_al', 'tau_cr']].to_string(index=False, float_format=lambda x: '%.3g' % x))
print('saved %s' % (out_dir / 'picks.csv'))

"""
STEP 7: figure of the chosen day of each case (STEP 6) in context.
    Columns = cases; rows = the forcing, hours from the day's start t0, from 48 h before
    to 48 h after the day (the day shaded); every row shares its y axis across cases.
        Q_tide    flood transport -Q_tide at pc_lp, + = into the cove     [m3/s]
        D2        semidiurnal envelope (spring/neap)                      [m3/s]
        D1        diurnal envelope (tropic/equatorial)                    [m3/s]
        drho      3-section mean, hourly (grey) and Godin (black); dashed = the
                  record terciles of the daily means (STEP 3)              [kg/m3]
        tau       Godin along-cove (+ up-cove) and cross-cove (+ north) stress;
                  dashed = +-TTH                                          [Pa]
    D2 and D1 are ranked within the month, so they have no fixed tercile lines; the
    ranks are in each column title.
    Figure: candidates_order.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAD = 48
C_AL, C_CR = '#009E73', '#E69F00'
drho_raw = di[['drho_' + sn for sn in SECTS]].mean(axis=1)
qt = pd.Series(qf, index=T)
dq = days.drho.quantile([1 / 3, 2 / 3]).values

def context_fig(best, rks, title, png):
    """Columns = rows of best; rows = -Q_tide, D2, D1, drho, tau, +-PAD h around each day.
    rks = the ranks shown in the column titles."""
    fig, axs = plt.subplots(5, len(best), figsize=(2.6 * len(best) + 1, 11), sharex='col', sharey='row', layout='constrained', squeeze=False)
    for c, row in best.iterrows():
        w = (T >= row.t0 - pd.Timedelta(hours=PAD)) & (T < row.t1 + pd.Timedelta(hours=PAD))
        hr = ((T[w] - row.t0) / pd.Timedelta(hours=1)).values
        dur = (row.t1 - row.t0) / pd.Timedelta(hours=1)
        axs[0, c].plot(hr, qt[w], color='k', lw=1)
        axs[1, c].plot(hr, dh.D2[w], color='k', lw=1)
        axs[2, c].plot(hr, dh.D1[w], color='k', lw=1)
        axs[3, c].plot(hr, drho_raw[w], color='0.6', lw=0.5, label='hourly')
        axs[3, c].plot(hr, dh.drho[w], color='k', lw=1, label='Godin')
        axs[4, c].plot(hr, dh.tau_al[w], color=C_AL, lw=1.2, label='along (+ up-cove)')
        axs[4, c].plot(hr, dh.tau_cr[w], color=C_CR, lw=1.2, label='cross (+ north)')
        for y in dq:
            axs[3, c].axhline(y, color='k', lw=0.6, ls='--')
        for y in [-TTH, TTH]:
            axs[4, c].axhline(y, color='k', lw=0.6, ls='--')
        for ax in axs[:, c]:
            ax.axvspan(0, dur, color='0.9', lw=0, zorder=0)
            ax.axhline(0, color='0.5', lw=0.5)
            ax.grid(alpha=0.3)
            ax.set_xlim(-PAD, dur + PAD)
        axs[0, c].set_title('%s\n%s UTC\n%s%s'
                            % (row.case, row.t0.strftime('%Y-%m-%d %H:%M'), '  '.join('%s %.2f' % (k, row[k]) for k in rks),
                               '' if row.n == 1 else '  (n = %d)' % row.n), fontsize=9)
        axs[-1, c].set_xlabel('hours from t0')
    for ax, lab in zip(axs[:, 0], ['-Q_tide [m$^3$ s$^{-1}$]\n+ = flood', 'D2 [m$^3$ s$^{-1}$]', 'D1 [m$^3$ s$^{-1}$]',
                                   r'$\Delta\sigma_0$ [kg m$^{-3}$]', r'$\tau$ [Pa]']):
        ax.set_ylabel(lab)
    axs[3, 0].legend(fontsize=7, loc='upper left')
    axs[4, 0].legend(fontsize=7, loc='lower left')
    fig.suptitle(title)
    fig.savefig(out_dir / png, dpi=150, transparent=True)
    plt.close(fig)
    print('saved %s' % (out_dir / png))

context_fig(best, R, 'Penn Cove candidate lunar days, the chosen day of each case, all with the baseline tide order %s (%s)' % (ORDER, GTAGEX),
            'candidates_order.png')

"""
STEP 8: seasonal baselines, one per season, every axis ranked WITHIN THE SEASON
(2026-11-13): the baseline of a season is the median day of that season, not of the year.
    season    DJF, MAM, JJA, SON, from the month of the day's middle hour (YEAR only, so
              DJF = Jan, Feb and Dec 2024)
    r_D2_s, r_D1_s, r_strat_s   percentile (0-1) of the day's mean D2 / D1 / drho among
              the days of the same season. (STEP 3 ranks D2 and D1 within the month and
              drho over the year; the year rank puts almost no Nov-Dec or Aug-Sep day in
              the middle third of drho.)
    Wind stays absolute: 'calm' is |tau| <= TTH in any season.
    The STEP 4 gates (every hour calm, all three ranks in the middle third) are too few
    in some seasons, so here they are a score, not gates:
    pass      tide order == ORDER (the STEP 5 baseline order) only.
    score     sum of |r - 1/2| over r_D2_s, r_D1_s, r_strat_s, + (1 - calm): a day calm
              all 25 h and exactly the season median scores 0.
    strict    True if the day also passes every STEP 4 gate (on the seasonal ranks).
    Thinned; NTOP kept per season.
    Saved: seasonal_baselines.csv
    Figure: seasonal_baselines.png (the best of each season, as STEP 7; dashed drho
            lines are the YEAR terciles)
"""
SEASONS = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}
RS = ['r_D2_s', 'r_D1_s', 'r_strat_s']

days['season'] = ''
for sea, mm in SEASONS.items():
    days.loc[mon.isin(mm), 'season'] = sea
for rk, vn in zip(RS, ['D2', 'D1', 'drho']):
    days[rk] = days[vn].groupby(days.season).rank(pct=True)
days.to_pickle(out_dir / 'days.p')

out = []
print('STEP 8: seasonal baselines (%d)' % YEAR)
for sea in SEASONS:
    s_ = days.season == sea
    ok = s_ & (days.order == ORDER)
    strict = (days.wind == 'calm') & mid(days.r_D2_s) & mid(days.r_D1_s) & mid(days.r_strat_s)
    d = days[ok].assign(score=(held(days, RS) + 1 - days.calm)[ok], strict=strict[ok]).sort_values('score')
    c = thin(d)[COLS + ['D2', 'D1'] + RS + ['order', 'strict']]
    c.insert(0, 'n', np.arange(1, len(c) + 1))
    c.insert(0, 'case', sea + ' baseline')
    out.append(c)
    print('\n%s: %d days; season medians D2 %.0f, D1 %.0f m3/s, drho %.2f kg/m3; %d with the tide order, %d strict'
          % (sea, s_.sum(), days.D2[s_].median(), days.D1[s_].median(), days.drho[s_].median(), ok.sum(), (ok & strict).sum()))
    print(c[['n', 't0', 'score', 'strict', 'calm'] + RS + ['D2', 'D1', 'drho', 'drho_range', 'tau_al', 'tau_cr']].to_string(index=False, float_format=lambda x: '%.3g' % x))
seas = pd.concat(out, ignore_index=True)
seas.to_csv(out_dir / 'seasonal_baselines.csv', index=False, float_format='%.4g')
print('saved %s' % (out_dir / 'seasonal_baselines.csv'))

context_fig(seas[seas.n == 1].reset_index(drop=True), RS,
            'Penn Cove seasonal baselines %d: closest to calm and the season median of D2, D1, drho; tide order %s (%s)' % (YEAR, ORDER, GTAGEX),
            'seasonal_baselines.png')

"""
STEP 9: seasonal tropic days (2026-11-13), one per season, on the STEP 8 seasonal ranks:
the seasonal baseline with D1 moved to the tropic target, as the STEP 4 variants.
    pass      tide order == ORDER, and r_D1_s in the season's top third (> 2/3).
    score     |r_D1_s - HI| + |r_D2_s - 1/2| + |r_strat_s - 1/2| + (1 - calm): HI = 5/6 is
              the middle of the extreme third (a typical tropic day, not the most extreme).
    strict    True if also calm every hour and r_D2_s, r_strat_s in the middle third.
    Thinned; NTOP kept per season.
    Saved: seasonal_tropic.csv
    Figure: seasonal_tropic.png (as STEP 8)
"""
out = []
print('\nSTEP 9: seasonal tropic days (%d)' % YEAR)
for sea in SEASONS:
    ok = (days.season == sea) & (days.order == ORDER) & (days.r_D1_s > 2 / 3)
    strict = (days.wind == 'calm') & mid(days.r_D2_s) & mid(days.r_strat_s)
    score = (days.r_D1_s - HI).abs() + held(days, ['r_D2_s', 'r_strat_s']) + 1 - days.calm
    d = days[ok].assign(score=score[ok], strict=strict[ok]).sort_values('score')
    c = thin(d)[COLS + ['D2', 'D1'] + RS + ['order', 'strict']]
    c.insert(0, 'n', np.arange(1, len(c) + 1))
    c.insert(0, 'case', sea + ' tropic')
    out.append(c)
    print('\n%s: %d tropic days with the tide order, %d strict' % (sea, ok.sum(), (ok & strict).sum()))
    print(c[['n', 't0', 'score', 'strict', 'calm'] + RS + ['D2', 'D1', 'drho', 'drho_range', 'tau_al', 'tau_cr']].to_string(index=False, float_format=lambda x: '%.3g' % x))
stro = pd.concat(out, ignore_index=True)
stro.to_csv(out_dir / 'seasonal_tropic.csv', index=False, float_format='%.4g')
print('saved %s' % (out_dir / 'seasonal_tropic.csv'))

context_fig(stro[stro.n == 1].reset_index(drop=True), RS,
            'Penn Cove seasonal tropic days %d: D1 rank near 5/6 of the season, calm, season-median D2 and drho; tide order %s (%s)' % (YEAR, ORDER, GTAGEX),
            'seasonal_tropic.png')
