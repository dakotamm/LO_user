"""
Find pairs of tidal cycles that are matched on the TIDE but differ in
STRATIFICATION, so that a side-by-side animation of the two isolates the
stratification and not the forcing.

WHY MATCHING IS THE WHOLE JOB
Penn Cove stratification is seasonal (summer strong, winter weak) and the
tide is fortnightly, so the naive contrast -- a summer cycle against a winter
one -- confounds stratification with tidal amplitude, and the velocity
difference you see is mostly the difference in tidal forcing. Here a cycle is
only allowed to pair with another cycle of nearly the same tidal range, the
same diurnal inequality and the same spring/neap class; the pair is then
chosen to maximise the stratification difference that remains.

WHAT A "CYCLE" IS
One tidal DAY, not one M2 cycle: the window runs from a higher high water at
the reference section for --hours (default 25 h, a lunar day is 24.84 h).
Puget Sound is mixed semidiurnal with a large diurnal inequality, so the two
floods of a day are not interchangeable and an M2-length window would compare
a strong flood against a weak one. Higher high waters are found as the ssh
peaks separated by at least 20 h, which keeps the larger of each day's pair.
A cycle is kept only if the NEXT higher high water lands 23.5-26.5 h later,
which throws out the days where the diurnal inequality collapses and the
peak-finder has no clean choice to make.

MATCHING VARIABLES (all from the tide itself, none from the season)
    range_m     max - min ssh over the window
    ineq_m      difference between the window's two high waters
    phase       spring / neap / transition from the tidal phase calendar
Stratification is the Godin-lowpassed section-mean (bottom - surface) salinity
at the cycle midpoint -- lowpassed, so it is the SUBTIDAL state the cycle sits
in and not a value that the cycle's own tidal straining moved.

Skagit discharge and the wind are NOT matched on. They are reported for every
selected cycle instead, because they are part of why the stratification
differs -- controlling for them would remove the thing being contrasted -- but
you should know what they were before reading a movie.

Pairs are chosen greedily by stratification difference, and no cycle is used
twice, so --npairs 3 gives three independent contrasts rather than one cycle
paired against its three nearest neighbours.

Runs on the mac. Writes the pair table that 20260807_pc_cycle_movie.py reads.

run 20260807_pc_matched_cycles.py
run 20260807_pc_matched_cycles.py --dr 0.10 --npairs 5
run 20260807_pc_matched_cycles.py --strat_sect pc_cp --same_phase 0
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from scipy.signal import find_peaks

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('--gtx', default='wb1_t0_xn11abbur00')
p.add_argument('--coll', default='wb1_pc1')
p.add_argument('--ds0', default='2024.01.01')
p.add_argument('--ds1', default='2025.12.31')
p.add_argument('--ref', default='pc_lp', help='section defining tidal phase')
p.add_argument('--strat_sect', default='pc_lp',
               help='section whose stratification defines high vs low')
p.add_argument('--hours', type=float, default=25.0,
               help='length of a cycle window [h]; lunar day is 24.84')
p.add_argument('--dr', type=float, default=0.15,
               help='max |difference in tidal range| within a pair [m]')
p.add_argument('--dineq', type=float, default=0.25,
               help='max |difference in diurnal inequality| within a pair [m]')
p.add_argument('--same_phase', type=int, default=1,
               help='1 = both cycles must carry the same spring/neap label')
p.add_argument('--npairs', type=int, default=4,
               help='how many non-overlapping pairs to select')
p.add_argument('--tz', default='America/Los_Angeles')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260807_pc_matched_cycles'
Lfun.make_dir(out_dir)

GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
PHCOLOR = {'spring': '#4565e8', 'neap': '#7f7f7f', 'transition': '#b0b0b0'}


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


# ------------------------------------------------------------- the tide ---
dflux = xr.open_dataset(tef2 / ('hourly_flux_%s_%s_%s.nc'
                                % (args.ds0, args.ds1, gctag)))
tt = pd.to_datetime(dflux.time.values)                    # naive UTC
ssh = dflux.ssh.sel(sect=args.ref).values
qnet = dflux.qnet.sel(sect=args.ref).values
dflux.close()

# flood sign at the reference section, checked rather than assumed
r_flood = np.corrcoef(qnet, np.gradient(ssh))[0, 1]
fsign = -1.0 if r_flood < 0 else 1.0
print('%s flood check: corr(qnet, d(ssh)/dt) = %+.2f -> flood is q %s 0'
      % (args.ref, r_flood, '<' if r_flood < 0 else '>'))

dt_h = (tt[1] - tt[0]) / pd.Timedelta(hours=1)
nwin = int(round(args.hours / dt_h)) + 1                  # frames per cycle

# ----------------------------------------------------- stratification ---
dstr = xr.open_dataset(tef2 / ('strat_%s_%s_%s.nc' % (args.ds0, args.ds1,
                                                      gctag)))
STRAT = {sn: godin(dstr.dstrat.sel(sect=sn).values)
         for sn in ['pc_lp', 'pc_lj', 'pc_cp']}
SBOT = {sn: godin(dstr.s_bot.sel(sect=sn).values)
        for sn in ['pc_lp', 'pc_lj', 'pc_cp']}
dstr.close()

# ------------------------------------------------------ daily context ---
ph = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20260806_tidal_phase'
                 / 'phase_daily.csv', parse_dates=['date_local'])
ph = ph.set_index(ph.date_local.dt.date)

riv = pd.read_csv(Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs'
                  / 'daily_flow.csv', parse_dates=['date'])
riv = riv.set_index(riv.date.dt.date)

# wind is DAILY means, so it is looked up by date rather than sliced by hour.
# Only w_along and the speed are used: w_cross carries a known sign error in
# reduce_wind_cove.py and is not worth reporting here.
wnd = xr.open_dataset(tef2 / ('wind_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag)))
W = wnd.to_dataframe()
W['w_speed'] = np.hypot(W.Uwind, W.Vwind)
W = W.set_index(pd.DatetimeIndex(W.index).date)
wnd.close()

# ------------------------------------------------- cut into cycles ---
# distance = 20 h keeps the larger of each day's two high waters
pk, _ = find_peaks(ssh, distance=int(round(20.0 / dt_h)))
rows = []
for i0, k in enumerate(pk[:-1]):
    gap_h = (tt[pk[i0 + 1]] - tt[k]) / pd.Timedelta(hours=1)
    if not (23.5 <= gap_h <= 26.5):
        continue                       # diurnal inequality collapsed here
    if k + nwin > len(tt):
        continue
    sl = slice(k, k + nwin)
    z = ssh[sl]
    # the two high waters inside the window, for the inequality
    hi, _ = find_peaks(z, distance=int(round(6.0 / dt_h)))
    ineq = np.ptp(z[hi]) if len(hi) >= 2 else np.nan
    mid = k + nwin // 2
    d_local = (tt[k].tz_localize('UTC').tz_convert(args.tz)).date()
    prow = ph.loc[d_local] if d_local in ph.index else None
    rrow = riv.loc[d_local] if d_local in riv.index else None
    rec = dict(
        i0=k, t0=tt[k], t_mid=tt[mid], date_local=d_local,
        range_m=np.ptp(z), ineq_m=ineq, sd_ssh=np.std(z),
        phase=('unknown' if prow is None else prow.model_phase),
        days_from_syzygy=(np.nan if prow is None else prow.days_from_syzygy),
        days_from_tropic=(np.nan if prow is None else prow.days_from_tropic),
        skagit_cms=(np.nan if rrow is None else rrow.skagit),
        month=tt[mid].month, year=tt[mid].year)
    for sn in ['pc_lp', 'pc_lj', 'pc_cp']:
        rec['dstrat_' + sn] = STRAT[sn][mid]
        rec['sbot_' + sn] = SBOT[sn][mid]
    wrow = W.loc[d_local] if d_local in W.index else None
    rec['w_along'] = np.nan if wrow is None else wrow.w_along
    rec['w_speed'] = np.nan if wrow is None else wrow.w_speed
    rows.append(rec)

C = pd.DataFrame(rows).dropna(subset=['range_m', 'dstrat_' + args.strat_sect])
C = C.reset_index(drop=True)
print('\n%d usable tidal-day cycles out of %d high-water peaks'
      % (len(C), len(pk)))
print('   tidal range  %.2f - %.2f m  (median %.2f)'
      % (C.range_m.min(), C.range_m.max(), C.range_m.median()))
SV = 'dstrat_' + args.strat_sect
print('   %s  %.2f - %.2f g/kg  (median %.2f)'
      % (SV, C[SV].min(), C[SV].max(), C[SV].median()))
C.to_csv(out_dir / 'cycles.csv', index=False, float_format='%.4f')

# --------------------------------------------------------- the pairing ---
# every admissible pair, then a greedy non-overlapping selection
ii, jj = np.triu_indices(len(C), k=1)
ok = ((np.abs(C.range_m.values[ii] - C.range_m.values[jj]) <= args.dr)
      & (np.abs(np.nan_to_num(C.ineq_m.values[ii], nan=9e9)
                - np.nan_to_num(C.ineq_m.values[jj], nan=9e9)) <= args.dineq))
if args.same_phase:
    ok &= (C.phase.values[ii] == C.phase.values[jj])
ii, jj = ii[ok], jj[ok]
dS = C[SV].values[jj] - C[SV].values[ii]
# order each pair high-stratification first
hi_i = np.where(dS > 0, jj, ii)
lo_i = np.where(dS > 0, ii, jj)

P = pd.DataFrame(dict(hi=hi_i, lo=lo_i, dstrat_diff=np.abs(dS)))
P['range_diff'] = np.abs(C.range_m.values[hi_i] - C.range_m.values[lo_i])
P['ineq_diff'] = np.abs(C.ineq_m.values[hi_i] - C.ineq_m.values[lo_i])
P = P.sort_values('dstrat_diff', ascending=False).reset_index(drop=True)
print('\n%d admissible pairs (|dr| <= %.2f m, |dineq| <= %.2f m%s)'
      % (len(P), args.dr, args.dineq,
         ', same spring/neap class' if args.same_phase else ''))

used, sel = set(), []
for _, r in P.iterrows():
    if r.hi in used or r.lo in used:
        continue
    used |= {r.hi, r.lo}
    sel.append(r)
    if len(sel) == args.npairs:
        break
S = pd.DataFrame(sel).reset_index(drop=True)

out = []
for n, r in S.iterrows():
    for role, k in [('high_strat', int(r.hi)), ('low_strat', int(r.lo))]:
        c = C.loc[k]
        out.append(dict(pair=n, role=role, t0=c.t0, t_mid=c.t_mid,
                        date_local=c.date_local, range_m=c.range_m,
                        ineq_m=c.ineq_m, phase=c.phase,
                        days_from_syzygy=c.days_from_syzygy,
                        dstrat=c[SV], sbot=c['sbot_' + args.strat_sect],
                        dstrat_pc_cp=c.dstrat_pc_cp,
                        skagit_cms=c.skagit_cms,
                        w_along=c.get('w_along', np.nan),
                        w_speed=c.get('w_speed', np.nan),
                        dstrat_diff=r.dstrat_diff, range_diff=r.range_diff,
                        hours=args.hours, strat_sect=args.strat_sect,
                        ref=args.ref))
SEL = pd.DataFrame(out)
SEL.to_csv(out_dir / 'selected_pairs.csv', index=False, float_format='%.4f')

show = ['pair', 'role', 't0', 'range_m', 'ineq_m', 'phase', 'dstrat',
        'sbot', 'skagit_cms', 'w_speed']
print('\nSELECTED PAIRS  (stratification from %s)' % args.strat_sect)
print(SEL[show].round(3).to_string(index=False))
print('\nsaved %s' % (out_dir / 'selected_pairs.csv'))

# ------------------------------------------------------------- figures ---
fig = plt.figure(figsize=(15, 9.5), layout='constrained')
gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], height_ratios=[1, 1, 1])

# left column: the two years, with the selected cycles marked
tl = tt.tz_localize('UTC').tz_convert(args.tz).tz_localize(None)
cx = pd.DatetimeIndex(C.t_mid).tz_localize('UTC').tz_convert(
    args.tz).tz_localize(None)
PAIRC = plt.get_cmap('tab10')

ax = fig.add_subplot(gs[0, 0])
ax.plot(tl, ssh, lw=0.3, color='0.75')
ax.plot(cx, C.range_m, '.', ms=3, color='k')
ax.set_ylabel('ssh (m) and\ntidal range (m)')
ax.set_title('%s -- hourly ssh at %s, per-cycle tidal range over it'
             % (args.gtx, args.ref), fontsize=10)

ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
ax2.plot(tl, STRAT[args.strat_sect], lw=1.2, color='#0072B2')
ax2.set_ylabel('subtidal stratification\n%s (g kg$^{-1}$)' % args.strat_sect)

ax3 = fig.add_subplot(gs[2, 0], sharex=ax)
ax3.plot(riv.date, riv.skagit, lw=1.2, color='#009E73')
ax3.set_ylabel('Skagit (m$^3$ s$^{-1}$)')
ax3.xaxis.set_major_locator(MonthLocator(interval=2))
ax3.xaxis.set_major_formatter(DateFormatter('%b\n%Y'))

for a in [ax, ax2, ax3]:
    a.grid(**GRID)
    for n, r in S.iterrows():
        for k, ls in [(int(r.hi), '-'), (int(r.lo), '--')]:
            x = pd.Timestamp(C.loc[k, 't_mid']).tz_localize('UTC').tz_convert(
                args.tz).tz_localize(None)
            a.axvline(x, color=PAIRC(n), ls=ls, lw=1.6, alpha=0.9)

# right column: the matching, seen as a scatter
axs = fig.add_subplot(gs[0:2, 1])
sc = axs.scatter(C.range_m, C[SV], c=C.month, cmap='twilight', s=14,
                 vmin=0.5, vmax=12.5, alpha=0.75, lw=0)
plt.colorbar(sc, ax=axs, label='month', ticks=[1, 4, 7, 10])
for n, r in S.iterrows():
    hi, lo = int(r.hi), int(r.lo)
    axs.plot(C.range_m[[hi, lo]], C[SV][[hi, lo]], '-o', color=PAIRC(n),
             ms=9, lw=2.5, mec='k', mew=0.8,
             label='pair %d: $\\Delta$strat %.2f, $\\Delta$range %.2f m'
                   % (n, r.dstrat_diff, r.range_diff))
axs.set_xlabel('tidal range over the cycle (m)')
axs.set_ylabel('subtidal stratification at %s (g kg$^{-1}$)' % args.strat_sect)
axs.grid(**GRID)
axs.legend(fontsize=8, loc='best')
axs.set_title('each dot is one tidal day; a pair is a vertical jump\n'
              'at nearly fixed tidal range', fontsize=10)

axb = fig.add_subplot(gs[2, 1])
w = 0.35
xk = np.arange(len(S))
for off, role, col in [(-w / 2, 'high_strat', '#0072B2'),
                       (w / 2, 'low_strat', '#D55E00')]:
    v = SEL[SEL.role == role]
    axb.bar(xk + off, v.range_m.values, w, color=col, label=role)
    for x, r_, s_ in zip(xk + off, v.range_m.values, v.dstrat.values):
        axb.text(x, r_ + 0.03, '%.2f' % s_, ha='center', fontsize=7)
axb.set_xticks(xk)
axb.set_xticklabels(['pair %d\n%s' % (n, S.iloc[n].name) for n in xk],
                    fontsize=8)
axb.set_xticklabels(['pair %d' % n for n in xk])
axb.set_ylabel('tidal range (m)')
axb.set_title('bars = the matched quantity, labels = stratification',
              fontsize=10)
axb.grid(**GRID, axis='y')
axb.legend(fontsize=8)

fig.suptitle('Penn Cove -- tidal cycles matched on tide, contrasted in '
             'stratification\nsolid line = high-stratification cycle, '
             'dashed = its low-stratification match', fontsize=12)
fn = out_dir / 'matched_cycles.png'
fig.savefig(fn, dpi=200, bbox_inches='tight')
print('saved %s' % fn)
