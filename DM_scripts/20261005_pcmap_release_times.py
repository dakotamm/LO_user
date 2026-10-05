"""
Release times for the pcmap experiment: one release at the peak of the
STRONGEST EBB (set E) and one at the peak of the STRONGEST FLOOD (set F) of
each lunar day in Penn Cove through the chosen year, plus the tracker2
commands to run them.

WHY PEAK FLOW, NOT SLACK WATER
Released at high water, the cove water sits a full tidal excursion landward of
its mean position and the outer cells hold Saratoga water that came in on the
flood -- it leaves on the next ebb and the outer cove looks artificially fast
(low water biases the other way). Penn Cove is short enough to be a standing
wave: transport at pc_lp is uncorrelated with ssh at zero lag (r = -0.002,
2024-25) and is ~ -dV/dt (r = -0.998 with d(ssh)/dt). So at peak flow the
water is near its tidal-mean position, and a particle's starting cell is
roughly where that water spends the tide. Each set is then interpretable on
its own; E vs F is the effect of which way the water moves first.

WHY TIDE-LOCKED
Within a set every release starts at the same phase of the same (strongest)
half-cycle, so release-to-release differences are spring-neap, wind,
stratification and season -- not phase (see the pcret confound).

HOW THE TIMES ARE PICKED
qnet at pc_lp from the tef2 hourly_flux file; the sign is set from the data so
that positive = out of the cove (ebb). distance = 20 h in find_peaks keeps the
strongest ebb (flood) of each lunar day. The samples are hour-centred (:30),
so each peak is refined with a parabola through it and its two neighbours,
then ROUNDED to the nearest whole hour for tracker.py -sh. The flow is near
its maximum for a couple of hours either side, so +/-30 min costs little.

WHY ONE COMMAND PER RELEASE, EACH WITH ITS OWN sub_tag
-nsd/-dbs step in whole days and would break the phase lock. And tracker.py
cleans its output directory on start (Lfun.make_dir(clean=True)), so parallel
runs must not share one: every release gets sub_tag <set>_<date>, i.e. its own
directory pcmap_3d[_shN]_<set>_<date>/release_<date>.nc.

RUN LENGTH
-dtt counts whole days from the start DAY, so -sh N loses N hours off the end.
The default -dtt 15 therefore leaves every release at least 14.04 d; the
analysis trims all of them to a common 14 d. The last release is the one whose
final tracked day still exists: start day + dtt - 1 <= -ds_end.

Writes, to LO_output/DM_outs/20261005_pcmap_release_times/:
  pcmap_release_times_<year>.csv   one row per release
  pcmap_commands_<year>.txt        one tracker command per line
  pcmap_release_times_<year>.png   qnet and ssh with the picked releases (check)

On apogee, from LO/tracker2 (12 at a time, 0-20 s jitter so the runs don't
collide on the shared tracks2/exp_info.csv that trackfun.py reads on import):
  cat pcmap_commands_2025.txt | xargs -P 12 -I CMD bash -c 'sleep $((RANDOM % 20)); CMD'

run 20261005_pcmap_release_times.py
run 20261005_pcmap_release_times.py -month 1      (one month, for timing)
run 20261005_pcmap_release_times.py -every 3      (every 3rd lunar day)
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-coll', default='wb1_pc1')
p.add_argument('-ds0', default='2024.01.01', help='tef2 extraction start')
p.add_argument('-ds1', default='2025.12.31', help='tef2 extraction end')
p.add_argument('-ref', default='pc_lp', help='section whose qnet defines ebb/flood')
p.add_argument('-year', type=int, default=2025)
p.add_argument('-month', type=int, default=0, help='0 = whole year')
p.add_argument('-every', type=int, default=1,
               help='keep every Nth release of each set (N lunar days apart), '
                    'counted from the first of the year so -month subsets match')
p.add_argument('-ds_end', default='2025.12.31',
               help='last day of ROMS output on apogee')
p.add_argument('-dtt', type=int, default=15)
p.add_argument('-exp', default='pcmap')
p.add_argument('-ro', type=int, default=2)
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
gctag = 'wb1_' + args.coll.split('_')[-1]
tef2 = Ldir['LOo'] / 'extract' / args.gtx / 'tef2'
out_dir = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times'
Lfun.make_dir(out_dir)

d = xr.open_dataset(tef2 / ('hourly_flux_%s_%s_%s.nc' % (args.ds0, args.ds1, gctag)))
tt = pd.to_datetime(d.time.values)                 # naive UTC, hour-centred
qnet = d.qnet.sel(sect=args.ref).values
ssh = d.ssh.sel(sect=args.ref).values
d.close()
dt_h = (tt[1] - tt[0]) / pd.Timedelta(hours=1)
# sign from the data: positive = out of the cove (ebb), i.e. ssh falling
r = np.corrcoef(qnet, np.gradient(ssh))[0, 1]
if abs(r) < 0.9:
    raise SystemExit('qnet vs d(ssh)/dt r = %.3f at %s: not a filling-draining '
                     'section, refusing to define ebb/flood from it' % (r, args.ref))
qout = -np.sign(r) * qnet
print('%s: corr(qnet, dssh/dt) = %+.3f -> ebb = %s qnet; '
      'corr(qnet, ssh) at lag 0 = %+.3f (standing wave if ~0)'
      % (args.ref, r, '+' if r < 0 else '-', np.corrcoef(qnet, ssh)[0, 1]))

last_start_day = pd.Timestamp(args.ds_end.replace('.', '-')) - pd.Timedelta(days=args.dtt - 1)

rows = []
for set_name, sgn in [('E', 1.0), ('F', -1.0)]:
    x = sgn * qout                                   # positive in this set's direction
    ipk, _ = find_peaks(x, distance=int(round(20 / dt_h)))
    ipk = ipk[(ipk > 0) & (ipk < len(x) - 1)]
    for k in ipk:
        # parabolic refinement of the peak time, in hours
        ym, y0, yp = x[k - 1], x[k], x[k + 1]
        den = ym - 2 * y0 + yp
        off = 0.5 * (ym - yp) / den if den != 0 else 0.0
        t_pk = tt[k] + pd.Timedelta(hours=float(np.clip(off, -0.5, 0.5)) * dt_h)
        t_rel = t_pk.round('h')
        if t_rel.year != args.year or t_rel.normalize() > last_start_day:
            continue
        ds = t_rel.strftime('%Y.%m.%d')
        sub_tag = '%s_%s' % (set_name, ds)
        out_name = args.exp + '_3d' + ('_sh%d' % t_rel.hour if t_rel.hour > 0 else '') + '_' + sub_tag
        cmd = ('python tracker.py -gtx %s -ro %d -exp %s -3d True -d %s -sh %d '
               '-dtt %d -clb True -sub_tag %s > %s.log 2>&1'
               % (args.gtx, args.ro, args.exp, ds, t_rel.hour, args.dtt,
                  sub_tag, out_name))
        rows.append(dict(set=set_name, t_peak=t_pk, t_release=t_rel, date=ds,
                         sh=t_rel.hour, q_peak=y0, ssh_at_peak=ssh[k], sub_tag=sub_tag,
                         out_name=out_name, cmd=cmd))

R = pd.DataFrame(rows).sort_values('t_release').reset_index(drop=True)
if args.every > 1:
    R = pd.concat([R[R.set == s].iloc[::args.every] for s in ['E', 'F']])
    R = R.sort_values('t_release').reset_index(drop=True)
if args.month:
    R = R[R.t_release.dt.month == args.month].reset_index(drop=True)
tag = ('%d' % args.year + ('_every%d' % args.every if args.every > 1 else '')
       + ('_%02d' % args.month if args.month else ''))
R.drop(columns='cmd').to_csv(out_dir / ('pcmap_release_times_%s.csv' % tag), index=False)
with open(out_dir / ('pcmap_commands_%s.txt' % tag), 'w') as f:
    f.write('\n'.join(R.cmd) + '\n')

for s in ['E', 'F']:
    r = R[R.set == s]
    gap = r.t_release.diff().dt.total_seconds() / 3600
    print('%s: %d releases, %s -> %s, spacing median %.1f h (min %.1f, max %.1f), '
          'peak |q| %.0f +/- %.0f m3/s, ssh at peak %+.2f +/- %.2f m (record mean %+.2f)'
          % (s, len(r), r.t_release.iloc[0], r.t_release.iloc[-1],
             gap.median(), gap.min(), gap.max(), r.q_peak.mean(), r.q_peak.std(),
             r.ssh_at_peak.mean(), r.ssh_at_peak.std(), np.nanmean(ssh)))
print('no two releases share a directory: %s' % (R.out_name.nunique() == len(R)))
print('last allowed start day %s (-ds_end %s, -dtt %d)'
      % (last_start_day.date(), args.ds_end, args.dtt))
print('wrote %s' % out_dir)

# check figure: the first month of the series with the picks on it
t0 = R.t_release.iloc[0].normalize()
m = (tt >= t0) & (tt < t0 + pd.Timedelta(days=31))
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axs[0].plot(tt[m], qout[m], color='0.4', lw=0.8)
axs[1].plot(tt[m], ssh[m], color='0.4', lw=0.8)
for s, c, sg in [('E', '#e8455e', 1), ('F', '#4565e8', -1)]:
    r = R[(R.set == s) & (R.t_release < t0 + pd.Timedelta(days=31))]
    axs[0].plot(r.t_peak, sg * r.q_peak, 'o', ms=4, color=c, label='%s (peak)' % s)
    axs[1].plot(r.t_peak, r.ssh_at_peak, 'o', ms=4, color=c)
    for ax in axs:
        for t in r.t_release:
            ax.axvline(t, color=c, lw=0.4, alpha=0.6)
axs[0].set_ylabel('transport out of cove at %s [m3/s]' % args.ref)
axs[1].set_ylabel('ssh at %s [m]' % args.ref)
axs[0].set_title('pcmap releases: strongest ebb (E) and flood (F) per lunar day; '
                 'lines = rounded release hour')
for ax in axs:
    ax.grid(color='lightgray', linestyle='--', alpha=0.5)
axs[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / ('pcmap_release_times_%s.png' % tag), dpi=200, transparent=True)
plt.close(fig)

# all releases through the period: time of day, peak transport, ssh at peak
fig, axs = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for s, c, mk in [('E', '#e8455e', 'o'), ('F', '#4565e8', 's')]:
    r = R[R.set == s]
    axs[0].plot(r.t_release, r.sh, mk, ms=2.5, color=c, label='%s (%d)' % (s, len(r)))
    axs[1].plot(r.t_release, r.q_peak, mk, ms=2.5, color=c)
    axs[2].plot(r.t_release, r.ssh_at_peak, mk, ms=2.5, color=c)
axs[2].axhline(np.nanmean(ssh), color='0.3', lw=0.8, ls='--', label='record-mean ssh')
axs[0].set_ylabel('release hour [UTC]')
axs[0].set_yticks(range(0, 25, 6))
axs[1].set_ylabel('peak |transport| at %s [m3/s]' % args.ref)
axs[2].set_ylabel('ssh at peak [m]')
axs[0].set_title('pcmap release times: strongest ebb (E) and flood (F) per lunar day')
for ax in axs:
    ax.grid(color='lightgray', linestyle='--', alpha=0.5)
axs[0].legend(fontsize=8, loc='upper right')
axs[2].legend(fontsize=8, loc='lower right')
fig.tight_layout()
fig.savefig(out_dir / ('pcmap_release_times_all_%s.png' % tag), dpi=200, transparent=True)
plt.close(fig)
