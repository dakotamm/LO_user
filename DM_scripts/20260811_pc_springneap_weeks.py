"""
Pick two one-week windows for a Penn Cove particle-tracking experiment: one
around the early-August NEAP tide and one around the SPRING tide that follows
it. Companion to 20260811_pc_matched_weeks.py, and its mirror image.

WHY THIS IS THE COMPLEMENT OF THE MATCHED-WEEKS EXPERIMENT
20260811_pc_matched_weeks.py held the tide fixed and let the season vary, so
that a retention difference had to be seasonal. It answered "does a September
cove hold its bottom water longer than a February one at the same tide?" and
found ~3x. That leaves the other half of the question untested: how much of
Penn Cove's bottom-water retention is set by the tide itself. Here the season
is held fixed -- both windows are in August, ~7 days apart, so bottom DO,
stratification, Skagit discharge and daylength are as close to common-mode as
a spring-neap contrast allows -- and the TIDE is the thing that varies.

The pair is deliberately adjacent, not drawn from different months. Neap and
spring are half a spring-neap cycle apart (7.38 d), so consecutive windows are
the closest in season that a full tidal contrast can ever be. Any wider
separation would start putting the seasonal drawdown back into the number.

WHY THE WINDOWS ARE CENTRED ON THE EXTREMUM
The tidal envelope is roughly sinusoidal in the spring-neap cycle, so the
7-day mean of the envelope is most different between the two windows when each
window is CENTRED on its extremum. Releasing AT the neap and tracking forward
a week would instead walk the cohort straight into the following spring, and
the two runs would then see nearly the same mean forcing. Centring costs
something -- the release instant is ~3.5 d before the extremum, so the first
day is a transition -- but it is the arrangement that maximises the contrast
over the tracked week, which is what the retention curves integrate.

THE RELEASE INSTANT IS STILL PHASE-LOCKED
Each window start is snapped to the nearest higher high water (peaks >= 20 h
apart, so the larger of each day's pair). Both cohorts therefore begin at the
same point of the tidal DAY even though they sit at opposite points of the
spring-neap cycle. Without this the first hours of the two runs would differ
in semidiurnal phase as well as in amplitude, which is the error that made the
original pcret pair uncomparable (see 20260806 notes).

WHAT IS *NOT* CONTROLLED
Wind and the residual seasonal DO drawdown across the ~7 days between the
windows. Both are printed for each window so they can be read alongside any
result; neither is matched, because there is nothing to match them against
once the windows are forced to be adjacent.

Runs on the mac; everything it reads is already local. Writes a CSV and a
figure, and prints the two tracker commands.

run 20260811_pc_springneap_weeks.py
run 20260811_pc_springneap_weeks.py -days 7 -dtt 10
run 20260811_pc_springneap_weeks.py -year 2024 -month 8 -half first
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from lo_tools import Lfun, zfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-coll', default='wb1_pc1')
p.add_argument('-ds0', default='2024.01.01')
p.add_argument('-ds1', default='2025.12.31')
p.add_argument('-ref', default='pc_cp',
               help='section whose ssh defines the tide (sections agree to <1 cm)')
p.add_argument('-days', type=float, default=7.0, help='window length [days]')
p.add_argument('-year', type=int, default=2024)
p.add_argument('-month', type=int, default=8)
p.add_argument('-half', default='first', choices=['first', 'second'],
               help='which half of the month the NEAP must fall in')
p.add_argument('-start_phase', default='HHW', choices=['HHW', 'LLW'],
               help='tidal phase both releases start on')
p.add_argument('-dtt', type=int, default=7,
               help='tracker run length to print in the commands [days]')
p.add_argument('-exp', default='pcbot', help='tracker2 experiment name')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
hyp_dir = Ldir['LOo'] / 'DM_outs' / '20260806_hypoxia'
riv_fn = Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs' / 'daily_flow.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_springneap_weeks'
Lfun.make_dir(out_dir)

GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
C_NEAP = '#2b8cbe'    # weak tide
C_SPRING = '#d94801'  # strong tide


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


# ------------------------------------------------------------------ tide ---
fn_flux = tef2 / ('hourly_flux_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag))
d = xr.open_dataset(fn_flux)
tt = pd.to_datetime(d.time.values)                 # naive UTC
ssh = d.ssh.sel(sect=args.ref).values
d.close()
dt_h = (tt[1] - tt[0]) / pd.Timedelta(hours=1)
nwin = int(round(args.days * 24 / dt_h))
print('tide: %s ssh, %d samples at %.2f h, %s to %s (UTC)'
      % (args.ref, len(tt), dt_h, tt[0], tt[-1]))

# Spring-neap envelope. 2 x Godin|ssh'| is a smooth amplitude measure: the
# Godin filter kills the semidiurnal and diurnal bands outright and leaves the
# fortnightly modulation, so its extrema ARE the neaps and springs. The daily
# max-min range would do the same job at 24 h resolution but is noisy enough
# near the turning points to move a chosen date by a day.
env = 2 * godin(np.abs(ssh - np.nanmean(ssh)))

# ------------------------------------------------- neap, then next spring ---
lo = pd.Timestamp(year=args.year, month=args.month, day=1)
if args.half == 'first':
    hi = lo + pd.Timedelta(days=16)
else:
    lo, hi = lo + pd.Timedelta(days=14), lo + pd.offsets.MonthEnd(1)
ok = np.isfinite(env)
ineap_all, _ = find_peaks(np.where(ok, -env, -np.inf), distance=int(round(9 * 24 / dt_h)))
ispr_all, _ = find_peaks(np.where(ok, env, -np.inf), distance=int(round(9 * 24 / dt_h)))

cand = ineap_all[(tt[ineap_all] >= lo) & (tt[ineap_all] < hi)]
if len(cand) == 0:
    raise SystemExit('no neap found in %s to %s -- widen -half/-month'
                     % (lo.date(), hi.date()))
i_neap = cand[0]
nxt = ispr_all[ispr_all > i_neap]
if len(nxt) == 0:
    raise SystemExit('no spring after the neap in the record')
i_spr = nxt[0]
print('neap   envelope minimum  %s   env %.2f m' % (tt[i_neap], env[i_neap]))
print('spring envelope maximum  %s   env %.2f m' % (tt[i_spr], env[i_spr]))
print('separation %.2f d (half a spring-neap cycle is 7.38 d)'
      % ((tt[i_spr] - tt[i_neap]) / pd.Timedelta(days=1)))

# ----------------------------------- candidate starts at one tidal phase ---
sgn = 1.0 if args.start_phase == 'HHW' else -1.0
ipk, _ = find_peaks(sgn * ssh, distance=int(round(20 / dt_h)))
ipk = ipk[(ipk + nwin) < len(tt)]


def snap_start(i_ext):
    """HHW nearest to the start that centres a window on i_ext."""
    i_want = i_ext - nwin // 2
    return int(ipk[np.argmin(np.abs(ipk - i_want))])


i0_neap, i0_spr = snap_start(i_neap), snap_start(i_spr)

# ------------------------------------------------------------ context ------
hyp = pd.read_csv(hyp_dir / ('hypoxia_series_%s_pc.csv' % args.gtx),
                  parse_dates=['time_local'])
hyp['time'] = pd.to_datetime(hyp.time_local, utc=True).dt.tz_localize(None)
hyp = hyp.set_index('time').sort_index()
do_bot = np.interp(tt.values.astype('int64'),
                   hyp.index.values.astype('int64'), hyp.do_bot_mean.values)

ds = xr.open_dataset(tef2 / ('strat_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag)))
dstrat = godin(ds.dstrat.sel(sect=args.ref).values)
ds.close()

dw = xr.open_dataset(tef2 / ('wind_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag)))
wday = pd.to_datetime(dw.day.values)
# NOTE w_cross in this file is mislabelled at source: positive is SOUTHWARD.
w_along = np.interp(tt.values.astype('int64'), wday.values.astype('int64'),
                    dw.w_along.values)
dw.close()

riv = pd.read_csv(riv_fn, parse_dates=['date']).set_index('date')
skagit = np.interp(tt.values.astype('int64'),
                   riv.index.values.astype('int64'), riv['skagit'].values)


def window_stats(i0, name):
    sl = slice(i0, i0 + nwin)
    s = ssh[sl]
    nd = int(args.days)
    rng = [np.ptp(s[k * 24:(k + 1) * 24]) for k in range(nd)]
    return dict(run=name, i0=i0, t0=tt[i0], t1=tt[i0 + nwin - 1],
                env_mean=np.nanmean(env[sl]), rng_mean=np.mean(rng),
                rng_max=np.max(rng), rng_min=np.min(rng), sstd=s.std(),
                do_bot=do_bot[sl].mean(), dstrat=dstrat[sl].mean(),
                skagit=skagit[sl].mean(), w_along=w_along[sl].mean())


W = pd.DataFrame([window_stats(i0_neap, 'neap'), window_stats(i0_spr, 'spring')])
n, s = W.iloc[0], W.iloc[1]

print('\n---- the two windows (UTC, start = %s) ----' % args.start_phase)
for r, c in [(n, 'NEAP  '), (s, 'SPRING')]:
    print('%s  %s -> %s' % (c, pd.Timestamp(r.t0).strftime('%Y.%m.%d %H:%M'),
                            pd.Timestamp(r.t1).strftime('%Y.%m.%d %H:%M')))
    print('          envelope %.2f m | mean daily range %.2f m (%.2f-%.2f) | '
          'bottom DO %.2f mg/L' % (r.env_mean, r.rng_mean, r.rng_min, r.rng_max,
                                   r.do_bot))
    print('          dstrat %.2f g/kg | Skagit %.0f m3/s | w_along %+.1f m/s'
          % (r.dstrat, r.skagit, r.w_along))
print('\nTIDAL CONTRAST (the variable): envelope %.2f -> %.2f m, x%.2f; '
      'mean daily range %.2f -> %.2f m, x%.2f'
      % (n.env_mean, s.env_mean, s.env_mean / n.env_mean,
         n.rng_mean, s.rng_mean, s.rng_mean / n.rng_mean))
print('SEASONAL RESIDUAL (what is left over, and should be small):')
print('  bottom DO %.2f -> %.2f mg/L (%+.2f) | dstrat %.2f -> %.2f g/kg (%+.2f)'
      % (n.do_bot, s.do_bot, s.do_bot - n.do_bot,
         n.dstrat, s.dstrat, s.dstrat - n.dstrat))
print('  Skagit %.0f -> %.0f m3/s | w_along %+.1f -> %+.1f m/s'
      % (n.skagit, s.skagit, n.w_along, s.w_along))
W.to_csv(out_dir / 'springneap_weeks.csv', index=False)

# How long the contrast survives past the window. The two windows are adjacent,
# so a run extended much past -days walks the neap cohort into the spring and
# vice versa and the difference in mean forcing collapses. This is the number
# that sets -dtt, and it is the opposite situation from the matched-week pair,
# where a longer run only slowly degraded the match.
print('\n---- how long the spring-neap contrast holds ----')
print('  lead    neap env   spring env   ratio')
nmax = min(len(env) - max(i0_neap, i0_spr), int(28 * 24 / dt_h))
for nd in [3, 5, 7, 10, 14, 21, 28]:
    k = int(nd * 24 / dt_h)
    if k > nmax:
        break
    en = np.nanmean(env[i0_neap:i0_neap + k])
    es = np.nanmean(env[i0_spr:i0_spr + k])
    print('  %2d d %10.2f %12.2f %7.2f%s'
          % (nd, en, es, es / en, '   <- window' if nd == int(args.days) else ''))

# ------------------------------------------------- tracker2 commands -------
print('\n---- tracker2 commands (run on apogee) ----')
for r, tag in [(n, 'neap'), (s, 'spring')]:
    ts = pd.Timestamp(r.t0)
    print('# %s window, %s at %02d:%02d UTC -> released %02d:00'
          % (tag, args.start_phase, ts.hour, ts.minute, ts.hour))
    print('python tracker.py -gtx %s -ro 2 -exp %s -d %s -dtt %d -sh %d '
          '-3d True -clb True -sub_tag %s'
          % (args.gtx, args.exp, ts.strftime('%Y.%m.%d'), args.dtt, ts.hour, tag))
print('# two separate commands on purpose, same as the matched-week pair:')
print('# -nsd/-dbs steps in whole days and would break the phase lock.')
print('# tracker2 counts -dtt from the START DAY, so -sh h costs h hours off')
print('# the end; the retention script trims both runs to a common length.')

# ------------------------------------------------------------------ plot ---
fig = plt.figure(figsize=(13, 9))

ax = fig.add_subplot(3, 1, 1)
m = (tt >= lo - pd.Timedelta(days=20)) & (tt <= tt[i_spr] + pd.Timedelta(days=25))
ax.plot(tt[m], env[m], color='0.35', lw=1.2,
        label="tidal envelope, 2 x Godin|ssh'|")
ax.plot(tt[m], np.abs(ssh - np.nanmean(ssh))[m], color='0.8', lw=0.4, zorder=0,
        label="|ssh'| (raw)")
ax.axvspan(n.t0, n.t1, color=C_NEAP, alpha=0.35, lw=0, label='neap week')
ax.axvspan(s.t0, s.t1, color=C_SPRING, alpha=0.35, lw=0, label='spring week')
ax.plot(tt[i_neap], env[i_neap], 'v', color=C_NEAP, ms=9)
ax.plot(tt[i_spr], env[i_spr], '^', color=C_SPRING, ms=9)
ax2 = ax.twinx()
ax2.plot(hyp.index, hyp.do_bot_mean, color='#2a9d4a', lw=1.2)
ax2.set_xlim(ax.get_xlim())
ax2.set_ylabel('Penn Cove bottom DO [mg/L]', color='#2a9d4a')
ax.set_ylabel('tidal envelope [m]')
ax.set_title('%s: adjacent neap and spring weeks, same season' % args.gtx)
ax.grid(**GRID); ax.legend(loc='upper left', fontsize=8)

ax = fig.add_subplot(3, 1, 2)
th = np.arange(nwin) * dt_h
for i0, c, lab in [(i0_neap, C_NEAP, 'neap  '), (i0_spr, C_SPRING, 'spring')]:
    a = ssh[i0:i0 + nwin]
    ax.plot(th, a - a.mean(), color=c, lw=1.2,
            label='%s %s' % (lab, pd.Timestamp(tt[i0]).strftime('%Y.%m.%d %H:%M')))
ax.set_xlabel('hours from release'); ax.set_ylabel("ssh' [m]")
ax.set_title('the two tides as released (both start at %s)' % args.start_phase)
ax.set_xlim(0, th[-1]); ax.grid(**GRID); ax.legend(fontsize=8)

ax = fig.add_subplot(3, 1, 3)
for i0, c, lab in [(i0_neap, C_NEAP, 'neap'), (i0_spr, C_SPRING, 'spring')]:
    ax.plot(th / 24, env[i0:i0 + nwin], color=c, lw=1.4, label=lab)
ax.set_xlabel('days from release'); ax.set_ylabel('tidal envelope [m]')
ax.set_title('the forcing contrast the experiment is built on', fontsize=10)
ax.set_xlim(0, th[-1] / 24); ax.grid(**GRID); ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(out_dir / 'springneap_weeks.png', dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % (out_dir / 'springneap_weeks.png'))
