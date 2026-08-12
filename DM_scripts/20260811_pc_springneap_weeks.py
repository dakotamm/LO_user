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
is held fixed -- both windows are in August, ~9 days apart, so bottom DO,
stratification, Skagit discharge and daylength are as close to common-mode as
a spring-neap contrast allows -- and the TIDE is the variable.

The year is 2025 by default, matching 20260811_pc_forcing_stack.py and the
rest of the recent Penn Cove tidal work. Note that the earlier pcbot pair
(20260811_pc_matched_weeks.py, hiDO/loDO) is in 2024, so the two experiments
are not in the same year -- fine for reading each on its own terms, but they
cannot be pooled into one four-run comparison without saying so.

THE WINDOWS ARE CENTRED ON QPRISM, NOT ON SEA-LEVEL RANGE
This matters more than it sounds, and the first version of this script got it
wrong. At pc_cp the diurnal and semidiurnal bands are almost exactly equal in
sea-level amplitude (std 0.77 and 0.75 m -- Puget Sound's mixed semidiurnal
regime), but they carry very different volume transport, because transport
scales with d(ssh)/dt rather than ssh: at equal amplitude a semidiurnal
constituent moves about twice the water a diurnal one does. So

  2 x Godin|ssh'|   weights the two bands about equally
  Qprism            weights the semidiurnal about twice as heavily

and the two bands have DIFFERENT fortnightly periods -- 14.77 d for the
semidiurnal (M2-S2, the spring-neap cycle proper) and 13.66 d for the diurnal
(K1-O1, the tropic/equatorial cycle, driven by lunar declination rather than
lunar phase). A sea-level envelope is therefore a blend of two cycles that
beat against each other with a period of 1/(1/13.66 - 1/14.77) = 180 days,
and its extrema wander 2-3 days either side of the true spring-neap cycle
depending on the time of year. Measured over 2024-25 at pc_cp:

  qprism vs semidiurnal envelope        r = +0.985
  qprism vs envelope of d(ssh)/dt       r = +0.994
  qprism vs diurnal envelope            r = -0.147
  qprism vs total |ssh'| envelope       r = +0.506   <- the wrong metric

and against lunar phase, qprism minima fall 0.69 +/- 0.56 d after quadrature
while the total-envelope minima fall 0.35 +/- 1.96 d after it. Qprism is the
signal that is actually locked to the moon, so centring on it is what makes
these windows a spring-neap pair rather than a pair of arbitrary weeks. It is
also the exchange that ventilates the cove and the quantity plotted in panel c
of 20260811_pc_forcing_stack.py, so the experiment varies the thing that stack
already shows. -center env reproduces the old behaviour for comparison.

WHY THE WINDOWS ARE CENTRED ON THE EXTREMUM
The fortnightly cycle is roughly sinusoidal, so the 7-day mean of the forcing
is most different between the two windows when each is CENTRED on its
extremum. Releasing AT the neap and tracking forward a week would instead walk
the cohort straight into the following spring, and the two runs would see
nearly the same mean forcing.

THE RELEASE INSTANT IS STILL PHASE-LOCKED
Each window start is snapped to the nearest higher high water (peaks >= 20 h
apart, so the larger of each day's pair). Both cohorts therefore begin at the
same point of the tidal DAY even though they sit at opposite points of the
spring-neap cycle. Without this the first hours of the two runs would differ
in semidiurnal phase as well as in amplitude, which is the error that made the
original pcret pair uncomparable (see 20260806 notes).

WHAT IS *NOT* CONTROLLED, AND MUST BE REPORTED
1. THE DIURNAL INEQUALITY RUNS THE OTHER WAY. In early August the two
   fortnightly cycles are near antiphase, so the qprism-centred neap week has
   a LARGER diurnal envelope than the spring week even though its prism is far
   smaller. The pair is a clean contrast in tidal PRISM and a poor one in
   total tidal RANGE, and the neap cohort sees more flood/ebb asymmetry than
   the spring cohort. That is inherent to this time of year -- near June or
   December the two cycles are in phase and both would move together -- and it
   is printed for both windows so it can be stated rather than discovered.
2. Wind, and the residual seasonal DO drawdown across the ~8 days between the
   windows. Printed, not matched; there is nothing to match them against once
   the windows are forced to be adjacent.

Runs on the mac; everything it reads is already local. Writes a CSV and a
figure, and prints the two tracker commands.

run 20260811_pc_springneap_weeks.py
run 20260811_pc_springneap_weeks.py -dtt 10
run 20260811_pc_springneap_weeks.py -center env
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
               help='section defining the tide (sections agree to <1 cm in ssh)')
p.add_argument('-center', default='qprism', choices=['qprism', 'env'],
               help='what the windows are centred on; qprism is the spring-neap '
                    'cycle proper, env is the old blended sea-level envelope')
p.add_argument('-days', type=float, default=7.0, help='window length [days]')
p.add_argument('-year', type=int, default=2025)
p.add_argument('-month', type=int, default=8)
p.add_argument('-half', default='first', choices=['first', 'second'],
               help='which half of the month the NEAP must fall in')
p.add_argument('-start_phase', default='HHW', choices=['HHW', 'LLW'],
               help='tidal phase both releases start on')
p.add_argument('-dtt', type=int, default=0,
               help='tracker run length to print in the commands [days]; 0 = '
                    'work out the smallest value that leaves -days of record '
                    'common to both runs after the start-hour loss')
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
C_NEAP = '#2b8cbe'
C_SPRING = '#d94801'
SYN = 29.530588853                       # mean synodic month [d]
T_NEW = pd.Timestamp('2000-01-06 18:14')  # a known new moon, UTC


def godin(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zfun.lowpass(np.asarray(a, dtype=float), f='godin')


# ------------------------------------------------------------------ tide ---
d = xr.open_dataset(tef2 / ('hourly_flux_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag)))
tt = pd.to_datetime(d.time.values)                 # naive UTC
ssh = d.ssh.sel(sect=args.ref).values
d.close()
dt_h = (tt[1] - tt[0]) / pd.Timedelta(hours=1)
nwin = int(round(args.days * 24 / dt_h))
print('tide: %s ssh, %d samples at %.2f h, %s to %s (UTC)'
      % (args.ref, len(tt), dt_h, tt[0], tt[-1]))

# Band split, so the two fortnightly cycles can be reported separately. A
# brick-wall FFT band is crude as a filter but is only ever used here to form a
# SMOOTH envelope, and the Godin that follows removes everything the sharp
# edges would have rung at.
x = ssh - np.nanmean(ssh)
NF = len(x)
fq = np.fft.rfftfreq(NF, d=dt_h)                   # cycles per hour
XF = np.fft.rfft(x)


def band(lo_h, hi_h):
    m = (fq >= 1 / hi_h) & (fq <= 1 / lo_h)
    Y = np.zeros_like(XF)
    Y[m] = XF[m]
    return np.fft.irfft(Y, NF)


env = 2 * godin(np.abs(x))                 # total sea-level envelope (blended)
env_sd = 2 * godin(np.abs(band(11.0, 13.5)))   # semidiurnal: the M2-S2 cycle
env_di = 2 * godin(np.abs(band(22.0, 28.0)))   # diurnal: the K1-O1 cycle

# Qprism from bulk_calc_avg.py -- already Godin filtered and daily-subsampled,
# so it is put on the hourly clock by interpolation rather than filtered again.
dq = xr.open_dataset(tef2 / ('bulk_avg_%s_%s' % (args.ds0, args.ds1))
                     / (args.ref + '.nc'))
tq = pd.to_datetime(dq.time.values)
qprism = np.interp(tt.values.astype('int64'), tq.values.astype('int64'),
                   dq.qprism.values)
dq.close()

ok = np.isfinite(env_sd) & np.isfinite(env)
print('band amplitude at %s: semidiurnal std %.3f m, diurnal std %.3f m'
      % (args.ref, band(11.0, 13.5).std(), band(22.0, 28.0).std()))
for nm, a in [('semidiurnal env', env_sd), ('diurnal env', env_di),
              ('total env', env)]:
    print('  qprism vs %-16s r = %+.3f' % (nm, np.corrcoef(qprism[ok], a[ok])[0, 1]))

# ------------------------------------------------- neap, then next spring ---
sig = qprism if args.center == 'qprism' else env
lo = pd.Timestamp(year=args.year, month=args.month, day=1)
if args.half == 'first':
    hi = lo + pd.Timedelta(days=16)
else:
    lo, hi = lo + pd.Timedelta(days=14), lo + pd.offsets.MonthEnd(1)
dist = int(round(9 * 24 / dt_h))
sg = np.where(ok, sig, np.nan)
ineap_all, _ = find_peaks(np.where(ok, -sg, -np.inf), distance=dist)
ispr_all, _ = find_peaks(np.where(ok, sg, -np.inf), distance=dist)

cand = ineap_all[(tt[ineap_all] >= lo) & (tt[ineap_all] < hi)]
if len(cand) == 0:
    raise SystemExit('no neap found in %s to %s -- widen -half/-month'
                     % (lo.date(), hi.date()))
i_neap = cand[0]
nxt = ispr_all[ispr_all > i_neap]
if len(nxt) == 0:
    raise SystemExit('no spring after the neap in the record')
i_spr = nxt[0]
unit = 'm3/s' if args.center == 'qprism' else 'm'
print('\ncentring on %s' % args.center)
print('neap   minimum  %s   %s = %.2f %s'
      % (tt[i_neap], args.center, sig[i_neap], unit))
print('spring maximum  %s   %s = %.2f %s'
      % (tt[i_spr], args.center, sig[i_spr], unit))
print('separation %.2f d (half a spring-neap cycle is 7.38 d)'
      % ((tt[i_spr] - tt[i_neap]) / pd.Timedelta(days=1)))


# Age of the tide: how long after the lunar phase the extremum falls. This is
# the check that the pair really is spring-neap and not some other fortnightly
# beat -- a spring-neap extremum trails syzygy/quadrature by a day or so, and
# anything landing BEFORE the lunar phase is being pulled by the declinational
# cycle instead.
def lunar(kind, t0, t1):
    off = dict(new=0.0, full=0.5, q1=0.25, q3=0.75)[kind]
    k0 = int(np.floor(((t0 - T_NEW) / pd.Timedelta(days=1) - off * SYN) / SYN)) - 1
    out = [T_NEW + pd.Timedelta(days=(k + off) * SYN) for k in range(k0, k0 + 6)]
    return [t for t in out if t0 <= t <= t1]


def age_of_tide(t, kinds):
    w0, w1 = t - pd.Timedelta(days=10), t + pd.Timedelta(days=10)
    ph = sorted(sum([lunar(k, w0, w1) for k in kinds], []))
    b = min(ph, key=lambda s: abs((t - s) / pd.Timedelta(days=1)))
    return b, (t - b) / pd.Timedelta(days=1)


for lab, i, kinds in [('neap  ', i_neap, ['q1', 'q3']),
                      ('spring', i_spr, ['new', 'full'])]:
    b, dtd = age_of_tide(tt[i], kinds)
    print('%s extremum is %+.2f d after the lunar phase at %s'
          % (lab, dtd, b.strftime('%Y.%m.%d %H:%M')))

# ----------------------------------- candidate starts at one tidal phase ---
sgn = 1.0 if args.start_phase == 'HHW' else -1.0
ipk, _ = find_peaks(sgn * ssh, distance=int(round(20 / dt_h)))
ipk = ipk[(ipk + nwin) < len(tt)]


def snap_start(i_ext):
    """HHW nearest to the start that centres a window on i_ext."""
    return int(ipk[np.argmin(np.abs(ipk - (i_ext - nwin // 2)))])


i0_neap, i0_spr = snap_start(i_neap), snap_start(i_spr)

# ------------------------------------------------------------ context ------
hyp = pd.read_csv(hyp_dir / ('hypoxia_series_%s_pc.csv' % args.gtx),
                  parse_dates=['time_local'])
hyp['time'] = pd.to_datetime(hyp.time_local, utc=True).dt.tz_localize(None)
hyp = hyp.set_index('time').sort_index()
do_bot = np.interp(tt.values.astype('int64'),
                   hyp.index.values.astype('int64'), hyp.do_bot_mean.values)

dst = xr.open_dataset(tef2 / ('strat_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag)))
dstrat = godin(dst.dstrat.sel(sect=args.ref).values)
dst.close()

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
                qprism=np.nanmean(qprism[sl]), env_sd=np.nanmean(env_sd[sl]),
                env_di=np.nanmean(env_di[sl]), env=np.nanmean(env[sl]),
                rng_mean=np.mean(rng), rng_max=np.max(rng), rng_min=np.min(rng),
                do_bot=do_bot[sl].mean(), dstrat=dstrat[sl].mean(),
                skagit=skagit[sl].mean(), w_along=w_along[sl].mean())


W = pd.DataFrame([window_stats(i0_neap, 'neap'), window_stats(i0_spr, 'spring')])
n, s = W.iloc[0], W.iloc[1]

print('\n---- the two windows (UTC, start = %s) ----' % args.start_phase)
for r, c in [(n, 'NEAP  '), (s, 'SPRING')]:
    print('%s  %s -> %s' % (c, pd.Timestamp(r.t0).strftime('%Y.%m.%d %H:%M'),
                            pd.Timestamp(r.t1).strftime('%Y.%m.%d %H:%M')))
    print('          qprism %5.0f m3/s | semidiurnal env %.2f m | diurnal env '
          '%.2f m | total env %.2f m' % (r.qprism, r.env_sd, r.env_di, r.env))
    print('          mean daily range %.2f m (%.2f-%.2f) | bottom DO %.2f mg/L'
          % (r.rng_mean, r.rng_min, r.rng_max, r.do_bot))
    print('          dstrat %.2f g/kg | Skagit %.0f m3/s | w_along %+.1f m/s'
          % (r.dstrat, r.skagit, r.w_along))

print('\nCONTRAST spring/neap:')
print('  qprism           x%.2f   <- the experiment' % (s.qprism / n.qprism))
print('  semidiurnal env  x%.2f' % (s.env_sd / n.env_sd))
print('  diurnal env      x%.2f   <- runs the OTHER way in August; the neap week'
      % (s.env_di / n.env_di))
print('                            has the larger diurnal inequality, so this')
print('                            is a prism contrast, not a range contrast')
print('  total ssh env    x%.2f' % (s.env / n.env))
print('  mean daily range x%.2f' % (s.rng_mean / n.rng_mean))
print('SEASONAL RESIDUAL (should be small):')
print('  bottom DO %.2f -> %.2f mg/L (%+.2f) | dstrat %.2f -> %.2f g/kg (%+.2f)'
      % (n.do_bot, s.do_bot, s.do_bot - n.do_bot,
         n.dstrat, s.dstrat, s.dstrat - n.dstrat))
print('  Skagit %.0f -> %.0f m3/s | w_along %+.1f -> %+.1f m/s'
      % (n.skagit, s.skagit, n.w_along, s.w_along))
gap = (pd.Timestamp(s.t0) - pd.Timestamp(n.t1)) / pd.Timedelta(hours=1)
print('  windows %s by %.0f h' % ('separated' if gap > 0 else 'OVERLAP', abs(gap)))
W.to_csv(out_dir / 'springneap_weeks.csv', index=False)

# How long the contrast survives past the window. The two windows are adjacent,
# so a run extended much past -days walks the neap cohort into the spring and
# vice versa and the difference in mean forcing collapses. This is the number
# that sets -dtt, and it is the opposite situation from the matched-week pair,
# where a longer run only slowly degraded the match.
print('\n---- how long the spring-neap contrast holds ----')
print('  lead   neap qprism  spring qprism   ratio')
nmax = min(len(qprism) - max(i0_neap, i0_spr), int(28 * 24 / dt_h))
for nd in [3, 5, 7, 10, 14, 21, 28]:
    k = int(nd * 24 / dt_h)
    if k > nmax:
        break
    qn = np.nanmean(qprism[i0_neap:i0_neap + k])
    qs = np.nanmean(qprism[i0_spr:i0_spr + k])
    print('  %2d d %11.0f %14.0f %7.2f%s'
          % (nd, qn, qs, qs / qn, '   <- window' if nd == int(args.days) else ''))

# ------------------------------------------------- tracker2 commands -------
# -dtt is counted from the START DAY, so a release at -sh h returns only
# 24*dtt - h hours. The two runs here have very different start hours, and the
# analysis has to trim them to a common length, so dtt is set from the LATER
# of the two: anything less and the whole pair is cut down to the shorter run.
sh_max = max(pd.Timestamp(n.t0).hour, pd.Timestamp(s.t0).hour)
dtt = args.dtt if args.dtt > 0 else int(np.ceil(args.days + sh_max / 24))
common_h = 24 * dtt - sh_max
print('\n---- tracker2 commands (run on apogee) ----')
print('# -dtt %d: the later release is at -sh %d, so both runs still share '
      '%d h = %.2f d' % (dtt, sh_max, common_h, common_h / 24))
for r, tag in [(n, 'neap'), (s, 'spring')]:
    ts = pd.Timestamp(r.t0)
    # -sh is an integer hour and both candidates sit at :30 (ssh samples are
    # hour-centred), so both releases are floored by the SAME 30 min and stay
    # phase-locked to each other, which is the alignment that matters.
    #
    # nohup + </dev/null + & so the run survives the ssh session: tracker2
    # takes hours on a week of hourly history, and a dropped connection would
    # otherwise SIGHUP it half way through and leave a partial release file
    # that -clb True would happily overwrite on the retry.
    log = '%s_%s_%d.log' % (args.exp, tag, args.year)
    print('# %s window, %s at %02d:%02d UTC -> released %02d:00'
          % (tag, args.start_phase, ts.hour, ts.minute, ts.hour))
    print('nohup python tracker.py -gtx %s -ro 2 -exp %s -d %s -dtt %d -sh %d '
          '-3d True -clb True -sub_tag %s < /dev/null > %s 2>&1 &'
          % (args.gtx, args.exp, ts.strftime('%Y.%m.%d'), dtt, ts.hour, tag, log))
print('# watch:  tail -f %s_neap_%d.log' % (args.exp, args.year))
print('# two separate commands on purpose, same as the matched-week pair:')
print('# -nsd/-dbs steps in whole days and would break the phase lock.')
print('# tracker2 counts -dtt from the START DAY, so -sh h costs h hours off')
print('# the end; the retention script trims both runs to a common length.')

# ------------------------------------------------------------------ plot ---
fig = plt.figure(figsize=(13, 11))

ax = fig.add_subplot(4, 1, 1)
m = (tt >= lo - pd.Timedelta(days=20)) & (tt <= tt[i_spr] + pd.Timedelta(days=25))
ax.plot(tt[m], qprism[m], color='0.25', lw=1.4, label='qprism at %s' % args.ref)
ax.axvspan(n.t0, n.t1, color=C_NEAP, alpha=0.35, lw=0, label='neap week')
ax.axvspan(s.t0, s.t1, color=C_SPRING, alpha=0.35, lw=0, label='spring week')
ax.plot(tt[i_neap], qprism[i_neap], 'v', color=C_NEAP, ms=9)
ax.plot(tt[i_spr], qprism[i_spr], '^', color=C_SPRING, ms=9)
for k, mk in [('new', 'o'), ('full', 'o'), ('q1', '|'), ('q3', '|')]:
    for t in lunar(k, tt[m][0], tt[m][-1]):
        ax.axvline(t, color='0.7', lw=0.7,
                   ls='-' if k in ('new', 'full') else ':')
ax.set_ylabel('qprism [m3/s]')
ax.set_title('%s: neap and spring weeks centred on QPRISM (grey lines = lunar '
             'phase; solid syzygy, dotted quadrature)' % args.gtx, fontsize=10)
ax.grid(**GRID); ax.legend(loc='upper left', fontsize=8)

ax = fig.add_subplot(4, 1, 2)
# The point of this panel: the semidiurnal band tracks qprism, the diurnal band
# does not, and the total envelope is a blend of the two whose extrema sit
# between them. Centring on the total envelope is what the old version did.
ax.plot(tt[m], env_sd[m], color='#7a0177', lw=1.2, label='semidiurnal envelope')
ax.plot(tt[m], env_di[m], color='#00808a', lw=1.2, label='diurnal envelope')
ax.plot(tt[m], env[m], color='0.5', lw=1.0, ls='--', label='total |ssh| envelope')
ax.axvspan(n.t0, n.t1, color=C_NEAP, alpha=0.25, lw=0)
ax.axvspan(s.t0, s.t1, color=C_SPRING, alpha=0.25, lw=0)
ax.set_ylabel('envelope [m]')
ax.set_title('the two fortnightly cycles: 14.77 d semidiurnal (spring-neap) vs '
             '13.66 d diurnal (tropic)', fontsize=10)
ax.grid(**GRID); ax.legend(loc='upper left', fontsize=8, ncol=3)

ax = fig.add_subplot(4, 1, 3)
th = np.arange(nwin) * dt_h
for i0, c, lab in [(i0_neap, C_NEAP, 'neap  '), (i0_spr, C_SPRING, 'spring')]:
    a = ssh[i0:i0 + nwin]
    ax.plot(th, a - a.mean(), color=c, lw=1.2,
            label='%s %s' % (lab, pd.Timestamp(tt[i0]).strftime('%Y.%m.%d %H:%M')))
ax.set_xlabel('hours from release'); ax.set_ylabel("ssh' [m]")
ax.set_title('the two tides as released (both start at %s)' % args.start_phase,
             fontsize=10)
ax.set_xlim(0, th[-1]); ax.grid(**GRID); ax.legend(fontsize=8)

ax = fig.add_subplot(4, 1, 4)
for i0, c, lab in [(i0_neap, C_NEAP, 'neap'), (i0_spr, C_SPRING, 'spring')]:
    ax.plot(th / 24, qprism[i0:i0 + nwin], color=c, lw=1.4, label=lab)
ax.set_xlabel('days from release'); ax.set_ylabel('qprism [m3/s]')
ax.set_title('the forcing contrast the experiment is built on (x%.2f in the '
             'window mean)' % (s.qprism / n.qprism), fontsize=10)
ax.set_xlim(0, th[-1] / 24); ax.grid(**GRID); ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(out_dir / 'springneap_weeks.png', dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % (out_dir / 'springneap_weeks.png'))
