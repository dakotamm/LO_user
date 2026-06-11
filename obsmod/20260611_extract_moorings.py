"""
Extract lowpassed (daily Godin-filtered) mooring time series at the King County
(kc_whidbeyBasin) and Ecology (ecology_nc) CTD/bottle station locations, for use
in obs-model validation of wb1_t0_xn11abbur00 over 2024-2025.

This is a self-contained driver (modeled on LO/extract/moor/multi_mooring_driver.py).
The station list is embedded below (the 15 in-domain stations that appear in the
combined obs-mod pickles). It calls LO/extract/moor/extract_moor.py as a series of
subprocesses, one per station, requesting all variable groups with -lt lowpass.

Output:
    LO_output/extract/<gtx>/moor/KCEcology_2024_2025/<station>_<ds0>_<ds1>.nc

Run on apogee:
    python 20260611_extract_moorings.py -gtx wb1_t0_xn11abbur00 -ro 2 > moor.log &

Test (first 2 stations, keeps originals):
    python 20260611_extract_moorings.py -test True
"""

import sys
import argparse
import shutil
from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi

from lo_tools import Lfun

# ---- embedded station list: name -> (lon, lat) -------------------------------
# These are the in-domain KC + Ecology stations present in the combined obs-mod
# pickles (combined_[ctd,bottle]_[2024,2025]_wb1_t0_xn11abbur00.p).
STATIONS = {
    # ecology_nc
    'ADM001':       (-122.616699, 48.029999),
    'ADM003':       (-122.481796, 47.879169),
    'PSS019':       (-122.300003, 48.011669),
    'PTH005':       (-122.763298, 48.083328),
    'SAR003':       (-122.489998, 48.108330),
    'SKG003':       (-122.488297, 48.296669),
    # kc_whidbeyBasin
    'PENNCOVEENT':  (-122.6550, 48.2370),
    'PENNCOVEWEST': (-122.7200, 48.2249),
    'PSUSANBUOY':   (-122.4200, 48.1750),
    'PSUSANENT':    (-122.3300, 48.0600),
    'PSUSANKP':     (-122.4000, 48.1300),
    'Poss DO-2':    (-122.3358, 47.9392),
    'SARATOGACH':   (-122.3690, 48.0440),
    'SARATOGAOP':   (-122.5500, 48.1840),
    'SARATOGARP':   (-122.5500, 48.2400),
}


def sanitize(name):
    """Make a station name safe for use as a filename / -sn argument."""
    return name.replace(' ', '_')


# ---- command line arguments --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str, default='wb1_t0_xn11abbur00')
parser.add_argument('-ro', '--roms_out_num', type=int, default=2)
parser.add_argument('-0', '--ds0', type=str, default='2024.01.02')
parser.add_argument('-1', '--ds1', type=str, default='2025.12.30')
parser.add_argument('-lt', '--list_type', type=str, default='lowpass')
parser.add_argument('-Nproc', type=int, default=10)
parser.add_argument('-job', type=str, default='KCEcology_2024_2025')
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

sta_dict = dict(STATIONS)
if args.testing:
    sta_dict = {k: sta_dict[k] for k in list(sta_dict.keys())[:2]}

# extract_moor.py must run from its own directory
moor_dir = Ldir['LO'] / 'extract' / 'moor'

# where extract_moor.py drops its raw output, and where we collect this job
out_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor'
jout_dir = out_dir / args.job
Lfun.make_dir(jout_dir)
log_dir = out_dir / 'logs'
Lfun.make_dir(log_dir)

print(' 20260611_extract_moorings '.center(60, '='))
print('gtagex   = %s' % args.gtagex)
print('period   = %s to %s  (list_type=%s)' % (args.ds0, args.ds1, args.list_type))
print('stations = %d' % len(sta_dict))
print('results  -> %s' % str(jout_dir))
sys.stdout.flush()

njobs = len(sta_dict)
for ii, sn in enumerate(sta_dict.keys(), start=1):
    tt0 = time()
    sn_safe = sanitize(sn)
    lon, lat = sta_dict[sn]
    print('Working on %s (%d of %d)' % (sn_safe, ii, njobs), end='')
    sys.stdout.flush()

    cmd_list = ['python', 'extract_moor.py',
        '-gtx', args.gtagex, '-ro', str(args.roms_out_num),
        '-0', args.ds0, '-1', args.ds1, '-lt', args.list_type,
        '-sn', sn_safe, '-lon', ' ' + str(lon), '-lat', ' ' + str(lat),
        '-Nproc', str(args.Nproc), '-get_all', 'True']
    proc = Po(cmd_list, stdout=Pi, stderr=Pi, cwd=str(moor_dir))
    stdout, stderr = proc.communicate()

    # collect the result into the job folder
    moor_fn = out_dir / ('%s_%s_%s.nc' % (sn_safe, args.ds0, args.ds1))
    job_moor_fn = jout_dir / ('%s_%s_%s.nc' % (sn_safe, args.ds0, args.ds1))
    try:
        if args.testing:
            shutil.copyfile(moor_fn, job_moor_fn)
        else:
            shutil.move(moor_fn, job_moor_fn)
    except FileNotFoundError:
        print(' - error making %s' % job_moor_fn.name)

    # write logs
    for tag_, blob in [('screen_output', stdout), ('subprocess_error', stderr)]:
        fn = log_dir / ('%s_%s.txt' % (sn_safe, tag_))
        fn.unlink(missing_ok=True)
        if len(blob) > 0:
            with open(fn, 'w') as f:
                f.write(blob.decode())
    if args.testing and len(stderr) > 0:
        print('\n' + stderr.decode())

    print(': completed in %d sec' % (time() - tt0))
    sys.stdout.flush()

print('Done.')
