"""
Run the pcsrc tracker releases on apogee, N at a time, and record how long
each one takes.

pcsrc = Penn Cove source water. -exp pcsrc_spu seeds upper Saratoga only
(tef2 segment pc_lp_p; see LO_user/tracker2/experiments.py and
pcsrc_regions.py). Its release times ARE the pcmap release times, E and F,
read from the pcmap picker table
(DM_outs/20261005_pcmap_release_times/pcmap_release_times_<year>[_every<N>].csv),
so every pcsrc release has a pcmap twin that supplies the cove cohort.

Runs LO/tracker2/tracker.py once per row (its own -d, -sh and -sub_tag) in a
pool of -nproc. Launches are spaced by at least -stagger seconds, because every
tracker run writes the shared LO_output/tracks2/exp_info.csv that trackfun.py
reads back on import. One command per release, never -nsd/-dbs: those step in
whole days and break the tidal phase lock.

-dtt 15, the same as pcmap (adopted 2026-10-12): the question is the source
of the water entering the cove, and by 7-14 d the cove is >90% new water, so
the source is fully represented by then; longer runs only trace the source of
the source (20261012_pcsrc_spu_feasibility.py). Matching pcmap also means the
cove cohort covers the whole window with no extrapolation. Releases whose last
tracked day (start day + dtt - 1) would fall after -ds_end, the end of the
model output, are dropped, so a longer -dtt is still safe.

Output, all under LO_output/DM_outs/20261012_pcsrc_launch/:
  <exp>_timing.csv   one row per finished release: sub_tag, set, date, sh,
                     start, end, seconds, returncode, nc_ok, log
  logs_<exp>/        one tracker log per release

RESTARTABLE: a release is skipped if the timing file already has it with
returncode 0 and its release_<date>.nc exists. So after a crash or a killed
session, just run the same command again. -redo ignores that.

On apogee (one month first, to get real timings, then the rest):
  nohup python 20261012_pcsrc_launch.py -month 8 > pcsrc_launch_08.log 2>&1 &
  nohup python 20261012_pcsrc_launch.py > pcsrc_launch.log 2>&1 &

run 20261012_pcsrc_launch.py -dry -nproc 40 -stagger 0      (mac test: sleeps instead of tracking)
"""
import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from lo_tools import Lfun

p = argparse.ArgumentParser()
p.add_argument('-gtx', default='wb1_t0_xn11abbur00')
p.add_argument('-year', type=int, default=2025)
p.add_argument('-month', type=int, default=0, help='0 = every release in the table')
p.add_argument('-every', type=int, default=3,
               help='use the picker table made with the same -every')
p.add_argument('-set', default='both', choices=['both', 'E', 'F'])
p.add_argument('-nproc', type=int, default=15)
p.add_argument('-stagger', type=float, default=2.0, help='min seconds between launches')
p.add_argument('-dtt', type=int, default=15)
p.add_argument('-ro', type=int, default=2)
p.add_argument('-exp', default='pcsrc_spu')
p.add_argument('-ds_end', default='2025.12.31',
               help='last day of model output; releases whose final tracked day falls after it are dropped')
p.add_argument('-redo', action='store_true')
p.add_argument('-dry', action='store_true', help='sleep 1-3 s instead of running the tracker')
args = p.parse_args()

Ldir = Lfun.Lstart(gridname='wb1')
tbl = Ldir['LOo'] / 'DM_outs' / '20261005_pcmap_release_times' / ('pcmap_release_times_%d%s.csv'
       % (args.year, '_every%d' % args.every if args.every > 1 else ''))
out_dir = Ldir['LOo'] / 'DM_outs' / '20261012_pcsrc_launch'
log_dir = out_dir / ('logs_' + args.exp)
Lfun.make_dir(log_dir)
timing_fn = out_dir / ('%s_timing%s.csv' % (args.exp, '_dry' if args.dry else ''))
trk_code = Ldir['LO'] / 'tracker2'
trk_out = Ldir['LOo'] / 'tracks2' / args.gtx

R = pd.read_csv(tbl, parse_dates=['t_release'])
if args.month:
    R = R[R.t_release.dt.month == args.month]
if args.set != 'both':
    R = R[R.set == args.set]
# the table was built for one -dtt; drop releases this -dtt would run off the end
last_day = pd.Timestamp(args.ds_end.replace('.', '-')) - pd.Timedelta(days=args.dtt - 1)
n0 = len(R)
R = R[R.t_release.dt.normalize() <= last_day]
if len(R) < n0:
    print('dropped %d releases starting after %s (-dtt %d, -ds_end %s)'
          % (n0 - len(R), last_day.date(), args.dtt, args.ds_end))


def out_name(row):
    # tracker.py's folder name: <exp>_3d[_sh<h>]_<sub_tag>
    return '%s_3d%s_%s' % (args.exp, '_sh%d' % row.sh if row.sh else '', row.sub_tag)


assert all(out_name(r) == r.out_name.replace('pcmap', args.exp, 1) for r in R.itertuples())


def nc_path(row):
    return trk_out / out_name(row) / ('release_%s.nc' % row.date)


done = set()
if timing_fn.is_file() and not args.redo:
    T = pd.read_csv(timing_fn)
    done = set(T.sub_tag[T.returncode == 0])
todo = [row for row in R.itertuples() if args.redo or args.dry
        or not (row.sub_tag in done and nc_path(row).is_file())]
print('%d releases in table selection, %d to run, nproc %d%s'
      % (len(R), len(todo), args.nproc, ' (DRY RUN)' if args.dry else ''))
sys.stdout.flush()

launch_lock = threading.Lock()
last_launch = [0.0]


def run_one(row):
    with launch_lock:                          # space out the starts
        wait = args.stagger - (time.time() - last_launch[0])
        if wait > 0:
            time.sleep(wait)
        last_launch[0] = time.time()
    if args.dry:
        cmd = [sys.executable, '-c', 'import time, random; time.sleep(random.uniform(1, 3))']
    else:
        cmd = [sys.executable, 'tracker.py', '-gtx', args.gtx, '-ro', str(args.ro),
               '-exp', args.exp, '-3d', 'True', '-d', row.date, '-sh', str(row.sh),
               '-dtt', str(args.dtt), '-clb', 'True', '-sub_tag', row.sub_tag]
    log_fn = log_dir / ('%s.log' % out_name(row))
    t0 = time.time()
    with open(log_fn, 'w') as f:
        rc = subprocess.run(cmd, cwd=trk_code, stdout=f, stderr=subprocess.STDOUT).returncode
    t1 = time.time()
    return dict(sub_tag=row.sub_tag, set=row.set, date=row.date, sh=row.sh,
                start=pd.Timestamp(t0, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                end=pd.Timestamp(t1, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                seconds=round(t1 - t0, 1), returncode=rc,
                nc_ok=args.dry or nc_path(row).is_file(), log=log_fn.name)


t_all = time.time()
n_ok = n_bad = 0
with ThreadPoolExecutor(max_workers=args.nproc) as ex:
    futs = [ex.submit(run_one, row) for row in todo]
    for k, fut in enumerate(as_completed(futs), 1):
        rec = fut.result()
        # only this (main) thread writes the timing file, so no interleaving
        pd.DataFrame([rec]).to_csv(timing_fn, mode='a', index=False,
                                   header=not timing_fn.is_file())
        ok = rec['returncode'] == 0 and rec['nc_ok']
        n_ok += ok; n_bad += not ok
        print('[%d/%d] %-14s %7.1f s  %s' % (k, len(todo), rec['sub_tag'], rec['seconds'],
                                           'ok' if ok else 'FAILED, see logs/' + rec['log']))
        sys.stdout.flush()

el = time.time() - t_all
print('done: %d ok, %d failed, %.2f h wall clock' % (n_ok, n_bad, el / 3600))
if n_ok + n_bad:
    T = pd.read_csv(timing_fn)
    T = T[T.returncode == 0]
    if len(T):
        print('per-release seconds (all ok runs so far): median %.0f, max %.0f, n %d'
              % (T.seconds.median(), T.seconds.max(), len(T)))
print('timing: %s' % timing_fn)
