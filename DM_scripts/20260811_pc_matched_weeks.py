"""
Pick two one-week windows for a Penn Cove particle-tracking experiment: one in
the high-DO season and one in the low-DO season, MATCHED ON THE TIDE.

WHY THE MATCHING IS THE EXPERIMENT
The thing being compared is bottom-water retention inside pc_cp in high- vs
low-oxygen conditions. Retention in a tidal cove is set mostly by the tide, so
a naive contrast -- some week in winter against some week in September -- puts
a spring-neap difference and a seasonal difference into the same number and
cannot say which one moved the particles. Here the two weeks are required to
have nearly the SAME hourly sea-surface height over their whole 168 h, so the
tidal forcing is as close to common-mode as this run allows and what is left
is the season.

HOW A WINDOW IS SCORED
The cost of a pair is the RMS of (ssh_A - ssh_B) over the 168 h, with each
window's own mean removed first. Removing the mean drops the seasonal
steric/setup offset, which is not tidal forcing and which the tracker's own
free surface handles anyway. Using the hourly ssh difference -- not the daily
range -- is deliberate: it penalises a spring-neap mismatch, an amplitude
mismatch, AND a phase mismatch in one number. Two weeks with identical daily
ranges but offset by 6 h would score well on range and terribly here, and the
6 h offset is exactly the error that would ruin the comparison.

THE RELEASE INSTANT IS PHASE-LOCKED, NOT ARBITRARY
Candidate windows may only start at a higher high water at the reference
section (peaks separated by >= 20 h, which keeps the larger of each day's
pair). The two releases therefore begin at the same point of the tidal day,
so the two particle clouds are pushed the same way in their first hours. This
is not a detail: the earlier pcret release pair was ~9 h out of tidal phase
and its early retention curves were not comparable at all.

WHAT IS *NOT* MATCHED
Stratification, Skagit discharge and wind are reported for each window but
never matched on. They are part of what makes the seasons differ -- matching
them would delete the contrast -- but you should know what they were before
reading any tracker result, so every one of them is printed and plotted.

Seasons are defined from the model's own Penn Cove bottom DO
(20260806_hypoxia_timeseries.py output), not from calendar months, and the
window's mean bottom DO is printed so the "high" and "low" labels are earned.

Runs on the mac; everything it reads is already local. Writes a CSV of the
ranked pairs plus a figure, and prints the tracker commands for the winner.

run 20260811_pc_matched_weeks.py
run 20260811_pc_matched_weeks.py -days 7 -npairs 5
run 20260811_pc_matched_weeks.py -ref pc_cp -start_phase LLW
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
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
p.add_argument('-start_phase', default='HHW', choices=['HHW', 'LLW'],
               help='tidal phase both releases start on')
p.add_argument('-do_lo', type=float, default=3.0,
               help='window-mean bottom DO at or below this = low-DO season [mg/L]')
p.add_argument('-do_hi', type=float, default=6.5,
               help='window-mean bottom DO at or above this = high-DO season [mg/L]')
p.add_argument('-tol', type=float, default=0.10,
               help='tidal cost tolerance above the best pair (normalised '
                    'units); pairs inside it are ranked by DO contrast instead')
p.add_argument('-year', type=int, default=0,
               help='0 = search the whole record; a year confines BOTH windows '
                    'to it, so the pair can be pooled with other runs of that year')
p.add_argument('-dtt', type=int, default=0,
               help='tracker run length to print [days]; 0 = the smallest value '
                    'that leaves -days of record common to both runs')
p.add_argument('-npairs', type=int, default=5, help='how many pairs to report')
p.add_argument('-exp', default='pcbot', help='tracker2 experiment name to print commands for')
p.add_argument('-tz', default='America/Los_Angeles')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
hyp_dir = Ldir['LOo'] / 'DM_outs' / '20260806_hypoxia'
riv_fn = Ldir['LOo'] / 'DM_outs' / '20260806_river_hydrographs' / 'daily_flow.csv'
out_dir = Ldir['LOo'] / 'DM_outs' / '20260811_pc_matched_weeks'
Lfun.make_dir(out_dir)

GRID = dict(color='lightgray', linestyle='--', alpha=0.5)
C_HI = '#4565e8'   # high-DO (winter) window
C_LO = '#e8455e'   # low-DO window


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
nwin = int(round(args.days * 24 / dt_h))           # samples per window
print('tide: %s ssh, %d samples at %.2f h, %s to %s (UTC)'
      % (args.ref, len(tt), dt_h, tt[0], tt[-1]))

# Band split, same construction as 20260811_pc_springneap_weeks.py. The FFT
# brick wall is crude as a filter but is only ever used to form a smooth
# envelope, and the Godin that follows removes what the sharp edges ring at.
x = ssh - np.nanmean(ssh)
NF = len(x)
fq = np.fft.rfftfreq(NF, d=dt_h)
XF = np.fft.rfft(x)


def band(lo_h, hi_h):
    m = (fq >= 1 / hi_h) & (fq <= 1 / lo_h)
    Y = np.zeros_like(XF)
    Y[m] = XF[m]
    return np.fft.irfft(Y, NF)


env_di = 2 * godin(np.abs(band(22.0, 28.0)))   # diurnal K1-O1, 13.66 d cycle
env_sd = 2 * godin(np.abs(band(11.0, 13.5)))   # semidiurnal M2-S2, 14.77 d

# Qprism from bulk_calc_avg.py: already Godin filtered and daily, so it goes on
# the hourly clock by interpolation rather than being filtered twice.
dq = xr.open_dataset(tef2 / ('bulk_avg_%s_%s' % (args.ds0, args.ds1))
                     / (args.ref + '.nc'))
tq = pd.to_datetime(dq.time.values)
qprism = np.interp(tt.values.astype('int64'), tq.values.astype('int64'),
                   dq.qprism.values)
dq.close()
qok = np.isfinite(qprism) & np.isfinite(env_di)
print('qprism vs semidiurnal env r = %+.3f, vs diurnal env r = %+.3f'
      % (np.corrcoef(qprism[qok], env_sd[qok])[0, 1],
         np.corrcoef(qprism[qok], env_di[qok])[0, 1]))

# Candidate starts: one tidal phase per tidal day. distance=20 h keeps the
# LARGER of each day's two highs, which is what "higher high water" means in a
# mixed semidiurnal regime; without it the peak-finder returns both and the two
# releases could start on opposite members of the diurnal pair.
sgn = 1.0 if args.start_phase == 'HHW' else -1.0
ipk, _ = find_peaks(sgn * ssh, distance=int(round(20 / dt_h)))
ipk = ipk[(ipk + nwin) < len(tt)]
print('%d candidate %s starts' % (len(ipk), args.start_phase))

# ------------------------------------------------------- season from DO ----
hyp = pd.read_csv(hyp_dir / ('hypoxia_series_%s_pc.csv' % args.gtx),
                  parse_dates=['time_local'])
hyp['time'] = pd.to_datetime(hyp.time_local, utc=True).dt.tz_localize(None)
hyp = hyp.set_index('time').sort_index()
# daily lowpassed already; interpolate to the hourly tide clock
do_bot = np.interp(tt.values.astype('int64'),
                   hyp.index.values.astype('int64'), hyp.do_bot_mean.values)

# ------------------------------------------------- context, not matched ----
ds = xr.open_dataset(tef2 / ('strat_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag)))
dstrat = godin(ds.dstrat.sel(sect=args.ref).values)     # bottom - surface salt
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


def window_stats(i0):
    """Everything about the window starting at index i0."""
    sl = slice(i0, i0 + nwin)
    s = ssh[sl]
    # daily range, one value per 24 h of the window
    nd = int(args.days)
    rng = [np.ptp(s[k * 24:(k + 1) * 24]) for k in range(nd)]
    return dict(i0=i0, t0=tt[i0], t1=tt[i0 + nwin - 1],
                ssh_mean=s.mean(), rng_mean=np.mean(rng), rng_max=np.max(rng),
                rng_min=np.min(rng), sstd=s.std(),
                qprism=np.nanmean(qprism[sl]), env_sd=np.nanmean(env_sd[sl]),
                env_di=np.nanmean(env_di[sl]),
                do_bot=do_bot[sl].mean(), dstrat=dstrat[sl].mean(),
                skagit=skagit[sl].mean(), w_along=w_along[sl].mean())


# A window is usable only if every matching variable is finite across it. The
# Godin filter eats 1.5 days at each end of the record and qprism is NaN there,
# so without this the end-of-record windows would score as perfect matches on
# nothing at all.
good = np.array([np.all(np.isfinite(qprism[i:i + nwin]))
                 and np.all(np.isfinite(env_di[i:i + nwin])) for i in ipk])
if args.year > 0:
    good &= np.array([(tt[i].year == args.year)
                      and (tt[i + nwin - 1].year == args.year) for i in ipk])
    print('restricted to %d: %d of %d candidate starts survive'
          % (args.year, good.sum(), len(ipk)))
ipk = ipk[good]

W = pd.DataFrame([window_stats(i) for i in ipk])
W['season'] = np.where(W.do_bot <= args.do_lo, 'low',
                       np.where(W.do_bot >= args.do_hi, 'high', ''))
n_hi = (W.season == 'high').sum(); n_lo = (W.season == 'low').sum()
print('windows: %d high-DO (>= %.1f), %d low-DO (<= %.1f), %d unused'
      % (n_hi, args.do_hi, n_lo, args.do_lo, (W.season == '').sum()))
if n_hi == 0 or n_lo == 0:
    sys.exit('no candidates on one side -- loosen -do_hi / -do_lo')

# ---------------------------------------------------------- pair search ----
# WHAT THE PAIR IS MATCHED ON, AND WHY IT IS NOT SEA LEVEL.
# The first version of this script minimised RMS(ssh'_hi - ssh'_lo), which
# weights the tidal bands by their SEA-LEVEL amplitude. That is the wrong
# weighting for a flushing experiment. At pc_cp the diurnal and semidiurnal
# bands are nearly equal in sea level (std 0.77 and 0.75 m) but transport goes
# as d(ssh)/dt, so at equal amplitude the semidiurnal moves about twice the
# water. Worse, the two bands run on DIFFERENT fortnightly cycles -- 14.77 d
# for M2-S2, 13.66 d for K1-O1 -- so a sea-level cost is a blend of two beats
# and can call two weeks "matched" when the exchange that actually ventilates
# the cove differs substantially. This is the same correction made in
# 20260811_pc_springneap_weeks.py; see its header for the correlations.
#
# So the pair is matched on the two INDEPENDENT tidal cycles directly:
#   qprism   the tidal exchange at pc_cp (bulk_calc_avg), which tracks the
#            semidiurnal spring-neap cycle (r = +0.99) and is the ventilation
#            the experiment is about
#   env_di   the diurnal envelope, which qprism is blind to (r = -0.15) and
#            which sets the flood/ebb asymmetry within the tidal day
# Each is normalised by its own record-wide std so neither dominates on units,
# and the cost is the RMS over the window of the two normalised differences.
# RMS(ssh') is still computed, but as a DIAGNOSTIC printed alongside, not as
# the thing being minimised.
ih = np.where(W.season.values == 'high')[0]
il = np.where(W.season.values == 'low')[0]
s_q = np.nanstd(qprism[qok]); s_d = np.nanstd(env_di[qok])
Q = np.array([qprism[i:i + nwin] / s_q for i in ipk])
E = np.array([env_di[i:i + nwin] / s_d for i in ipk])
A = np.array([ssh[i:i + nwin] - ssh[i:i + nwin].mean() for i in ipk])
DQ = Q[ih][:, None, :] - Q[il][None, :, :]
DE = E[ih][:, None, :] - E[il][None, :, :]
cost = np.sqrt(((DQ ** 2 + DE ** 2) / 2).mean(axis=2))     # dimensionless
cost_ssh = np.sqrt(((A[ih][:, None, :] - A[il][None, :, :]) ** 2).mean(axis=2))
print('best pair cost %.3f (normalised), worst %.3f' % (cost.min(), cost.max()))

# SELECTION RULE. The tide is a CONSTRAINT, the DO contrast is the OBJECTIVE.
# Ranking straight by tidal cost is the wrong way round: it would hand back the
# best-matched pair regardless of whether its "low-DO" week was 2.5 or 1.3
# mg/L, and near the minimum the cost surface is flat -- dozens of pairs sit
# within a whisker of the best. So admit every pair within -tol of the minimum
# cost, all of which are matched far better than the seasonal signal being
# measured, and among those take the largest DO difference.
cmax = cost.min() + args.tol
adm = np.argwhere(cost <= cmax)
ddo = W.do_bot.values[ih[adm[:, 0]]] - W.do_bot.values[il[adm[:, 1]]]
order = adm[np.argsort(-ddo)]
print('%d pairs within %.3f of the best; ranking those by DO contrast'
      % (len(adm), args.tol))
pairs = []
used_h, used_l = [], []
for a, b in order:
    th, tl = W.t0.values[ih[a]], W.t0.values[il[b]]
    # no reusing a week, and no overlapping weeks between reported pairs
    if any(abs(th - u) < np.timedelta64(int(args.days * 24), 'h') for u in used_h):
        continue
    if any(abs(tl - u) < np.timedelta64(int(args.days * 24), 'h') for u in used_l):
        continue
    used_h.append(th); used_l.append(tl)
    h = W.iloc[ih[a]]; l = W.iloc[il[b]]
    pairs.append(dict(
        rank=len(pairs) + 1, cost=cost[a, b], rms_ssh=cost_ssh[a, b],
        hi_t0=h.t0, hi_t1=h.t1, lo_t0=l.t0, lo_t1=l.t1,
        hi_qprism=h.qprism, lo_qprism=l.qprism,
        dq_pct=100 * (h.qprism - l.qprism) / l.qprism,
        hi_envdi=h.env_di, lo_envdi=l.env_di,
        ddi_pct=100 * (h.env_di - l.env_di) / l.env_di,
        hi_rng=h.rng_mean, lo_rng=l.rng_mean, d_rng=h.rng_mean - l.rng_mean,
        hi_do=h.do_bot, lo_do=l.do_bot, d_do=h.do_bot - l.do_bot,
        hi_dstrat=h.dstrat, lo_dstrat=l.dstrat,
        hi_skagit=h.skagit, lo_skagit=l.skagit,
        hi_walong=h.w_along, lo_walong=l.w_along,
        i_hi=int(h.i0), i_lo=int(l.i0)))
    if len(pairs) == args.npairs:
        break
P = pd.DataFrame(pairs)

pd.set_option('display.width', 200)
print('\n---- ranked pairs (times UTC, window start = %s) ----' % args.start_phase)
show = P[['rank', 'cost', 'hi_t0', 'lo_t0', 'dq_pct', 'ddi_pct', 'rms_ssh',
          'hi_do', 'lo_do', 'hi_dstrat', 'lo_dstrat', 'hi_skagit', 'lo_skagit']]
print(show.to_string(index=False, float_format=lambda x: '%.3f' % x))
P.to_csv(out_dir / ('matched_weeks%s.csv'
                    % ('_%d' % args.year if args.year > 0 else '')), index=False)

b = P.iloc[0]
print('\n---- best pair ----')
for lab, t0, t1, col in [('HIGH-DO', b.hi_t0, b.hi_t1, 'hi'),
                         ('LOW-DO ', b.lo_t0, b.lo_t1, 'lo')]:
    print('%s  %s -> %s' % (lab, pd.Timestamp(t0).strftime('%Y.%m.%d %H:%M'),
                            pd.Timestamp(t1).strftime('%Y.%m.%d %H:%M')))
    print('          qprism %.0f m3/s | diurnal env %.2f m | daily range %.2f m'
          % (b['%s_qprism' % col], b['%s_envdi' % col], b['%s_rng' % col]))
    print('          bottom DO %.2f mg/L | dstrat %.2f g/kg | Skagit %.0f m3/s'
          ' | w_along %+.1f m/s'
          % (b['%s_do' % col], b['%s_dstrat' % col], b['%s_skagit' % col],
             b['%s_walong' % col]))
print('tidal mismatch: qprism %+.1f%%, diurnal envelope %+.1f%%, '
      'RMS(ssh) %.3f m, daily range %+.1f%%'
      % (b.dq_pct, b.ddi_pct, b.rms_ssh, 100 * b.d_rng / b.lo_rng))
print('DO contrast: %.2f mg/L' % b.d_do)

# tracker commands. tracker2 takes the START DAY and a duration in days; the
# release hour inside that day is the phase-locked start found above.
#
# -dtt is counted from the START DAY, so a release at -sh h returns only
# 24*dtt - h hours and the analysis has to trim the pair to a common length.
# The first pcbot pair was run at -dtt 14 with -sh 13 and lost 13 h that way.
# dtt is therefore set from the LATER of the two start hours, so both runs
# still cover the whole matched window.
sh_max = max(pd.Timestamp(b.hi_t0).hour, pd.Timestamp(b.lo_t0).hour)
dtt = args.dtt if args.dtt > 0 else int(np.ceil(args.days + sh_max / 24))
common_h = 24 * dtt - sh_max
yr = '_%d' % args.year if args.year > 0 else ''
print('\n---- tracker2 commands (run on apogee) ----')
print('# -dtt %d: the later release is at -sh %d, so both runs still share '
      '%d h = %.2f d' % (dtt, sh_max, common_h, common_h / 24))
for lab, t0, col in [('high-DO', b.hi_t0, 'hi'), ('low-DO', b.lo_t0, 'lo')]:
    ts = pd.Timestamp(t0)
    # -sh is an integer hour, so both releases are floored to the hour. Both
    # candidates sit at :30 (the ssh samples are hour-centred), so the two
    # windows are shifted by the SAME 30 min and stay phase-locked to each
    # other, which is the only alignment that matters here.
    tag = ('hiDO' if col == 'hi' else 'loDO') + yr
    print('# %s window, %s at %02d:%02d UTC -> released %02d:00'
          % (lab, args.start_phase, ts.hour, ts.minute, ts.hour))
    print('nohup python tracker.py -gtx %s -ro 2 -exp %s -d %s -dtt %d -sh %d '
          '-3d True -clb True -sub_tag %s < /dev/null > %s_%s.log 2>&1 &'
          % (args.gtx, args.exp, ts.strftime('%Y.%m.%d'), dtt, ts.hour, tag,
             args.exp, tag))
print('# two separate commands on purpose: -nsd/-dbs steps in whole days and')
print('# would break the tidal phase lock between the two releases.')

# How far past the window can the run be extended before the two tides part
# company? Same RMS cost, recomputed over a growing lead time. Worth knowing
# because -dtt is free to set longer than the matched week, but the match is
# not free to assume.
print('\n---- how long the tidal match holds ----')
print('  lead   qprism hi   qprism lo   diff     RMS(ssh)')
nmax = min(len(ssh) - max(b.i_hi, b.i_lo), int(28 * 24 / dt_h))
ah = ssh[b.i_hi:b.i_hi + nmax]; al = ssh[b.i_lo:b.i_lo + nmax]
ah = ah - ah.mean(); al = al - al.mean()
for nd in [3, 7, 10, 14, 21, 28]:
    n = int(nd * 24 / dt_h)
    if n > nmax:
        break
    qh = np.nanmean(qprism[b.i_hi:b.i_hi + n])
    ql = np.nanmean(qprism[b.i_lo:b.i_lo + n])
    print('  %2d d %11.0f %11.0f %+7.1f%% %10.3f m%s'
          % (nd, qh, ql, 100 * (qh - ql) / ql,
             np.sqrt(((ah[:n] - al[:n]) ** 2).mean()),
             '   <- matched here' if nd == int(args.days) else ''))

# ------------------------------------------------------------------ plot ---
fig = plt.figure(figsize=(13, 11))
ax = fig.add_subplot(4, 1, 1)
# Qprism, because that is what the pair is matched on. The old version drew
# the blended sea-level envelope, which is a different fortnightly signal --
# it correlates with qprism at only about +0.5.
ax.plot(tt, qprism, color='0.25', lw=1.0, label='qprism at %s' % args.ref)
ax2 = ax.twinx()
ax2.plot(hyp.index, hyp.do_bot_mean, color='#2a9d4a', lw=1.2)
ax2.set_ylabel('Penn Cove bottom DO [mg/L]', color='#2a9d4a')
ax2.axhline(args.do_hi, color='#2a9d4a', ls=':', lw=0.8)
ax2.axhline(args.do_lo, color='#2a9d4a', ls=':', lw=0.8)
for _, r in P.iterrows():
    for t0, t1, c in [(r.hi_t0, r.hi_t1, C_HI), (r.lo_t0, r.lo_t1, C_LO)]:
        ax.axvspan(t0, t1, color=c, alpha=0.5 if r['rank'] == 1 else 0.15, lw=0)
ax.set_ylabel('qprism [m3/s]')
ax.set_title('%s: %d-day windows matched on qprism + diurnal envelope, '
             'split by bottom DO%s'
             % (args.gtx, int(args.days),
                '  (%d only)' % args.year if args.year > 0 else ''))
ax.grid(**GRID); ax.legend(loc='upper left', fontsize=8)

# The two matched variables, over the two chosen windows. Drawn on the same
# axes as each other so the match is read directly rather than inferred from
# the cost number.
th = np.arange(nwin) * dt_h
ax = fig.add_subplot(4, 1, 2)
for i0, c, lab in [(b.i_hi, C_HI, 'high-DO  %s'
                    % pd.Timestamp(b.hi_t0).strftime('%Y.%m.%d')),
                   (b.i_lo, C_LO, 'low-DO   %s'
                    % pd.Timestamp(b.lo_t0).strftime('%Y.%m.%d'))]:
    ax.plot(th, qprism[i0:i0 + nwin], color=c, lw=1.6, label=lab + '  qprism')
    ax.plot(th, env_di[i0:i0 + nwin] * np.nanmax(qprism) / np.nanmax(env_di),
            color=c, lw=1.0, ls='--', label=lab + '  diurnal env (scaled)')
ax.set_ylabel('qprism [m3/s]')
ax.set_title('the matched variables: qprism %+.1f%%, diurnal envelope %+.1f%%'
             % (b.dq_pct, b.ddi_pct))
ax.set_xlim(0, th[-1]); ax.grid(**GRID); ax.legend(fontsize=7, ncol=2)

ax = fig.add_subplot(4, 1, 3)
ax.plot(th, ssh[b.i_hi:b.i_hi + nwin] - ssh[b.i_hi:b.i_hi + nwin].mean(),
        color=C_HI, lw=1.2, label='high-DO')
ax.plot(th, ssh[b.i_lo:b.i_lo + nwin] - ssh[b.i_lo:b.i_lo + nwin].mean(),
        color=C_LO, lw=1.2, label='low-DO')
ax.set_ylabel("ssh' [m]")
ax.set_title("sea level, a DIAGNOSTIC not a matching variable "
             "(RMS difference %.3f m)" % b.rms_ssh)
ax.set_xlim(0, th[-1]); ax.grid(**GRID); ax.legend(fontsize=8)

ax = fig.add_subplot(4, 1, 4)
ax.plot(th, ssh[b.i_hi:b.i_hi + nwin] - ssh[b.i_hi:b.i_hi + nwin].mean()
        - (ssh[b.i_lo:b.i_lo + nwin] - ssh[b.i_lo:b.i_lo + nwin].mean()),
        color='k', lw=1)
ax.axhline(0, color='0.6', lw=0.8)
ax.set_xlabel('hours from release'); ax.set_ylabel('high - low  [m]')
ax.set_title('residual tidal forcing difference the experiment cannot remove')
ax.set_xlim(0, th[-1]); ax.grid(**GRID)

fig.tight_layout()
fn_fig = out_dir / ('matched_weeks%s.png' % yr)
fig.savefig(fn_fig, dpi=200, transparent=True)
plt.close(fig)
print('\nwrote %s' % fn_fig)
