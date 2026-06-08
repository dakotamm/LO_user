"""
Driver to compare multiple model versions: run cast extractions and combine
obs+mod for each, then plot obs vs. all models for bottom DO time series.

Testing on mac:
run two_model_compare -gtx wb1_t0_xn11abbur00 wb1_t1_xn11abbur00 wb1_t0_xn11ab -lt hourly -year0 2024 -test True

Production:
python two_model_compare.py -gtx wb1_t0_xn11abbur00 wb1_t1_xn11abbur00 wb1_t0_xn11ab -lt average -year0 2024 -year1 2025
"""

import argparse
from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi
import sys

from lo_tools import Lfun
import obsmod_functions as omfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', type=str, nargs='+')   # one or more gtagex values
parser.add_argument('-ro', '--roms_out_num', type=int, default=0)
parser.add_argument('-lt', '--list_type', type=str, default='average')
parser.add_argument('-sources', type=str, default='all')
parser.add_argument('-otype', type=str, default='all')
parser.add_argument('-stations', type=str, default='')
parser.add_argument('-year0', type=int)
parser.add_argument('-year1', type=int, default=0)
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)

args = parser.parse_args()

if args.gtx is None or len(args.gtx) == 0:
    print('*** Missing required argument: -gtx')
    sys.exit()
if args.year0 is None:
    print('*** Missing required argument: -year0')
    sys.exit()

gtx_list = args.gtx

# Initialize Ldir from first gtx (for shared paths: LO, LOo, LOu)
gridname, tag, ex_name = gtx_list[0].split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

argsd = args.__dict__
for a in argsd.keys():
    if a not in Ldir.keys():
        Ldir[a] = argsd[a]

if Ldir['roms_out_num'] == 0:
    pass
elif Ldir['roms_out_num'] > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(Ldir['roms_out_num'])]

if args.year1 == 0:
    args.year1 = args.year0
year_list = list(range(args.year0, args.year1 + 1))
Ldir['year0'] = args.year0
Ldir['year1'] = args.year1

# Hook to LO_user for obsmod_functions
ff = 'obsmod_functions.py'
fn = Ldir['LO'] / 'obsmod' / ff
ufn = Ldir['LOu'] / 'obsmod' / ff
if ufn.is_file():
    print('Importing %s from LO_user' % ff)
    omfun = Lfun.module_from_file('obsmod_functions', ufn)
else:
    omfun = Lfun.module_from_file('obsmod_functions', fn)

try:
    source_list = omfun.parse_sources_arg(args.sources)
except ValueError as e:
    print('*** Bad -sources argument: ' + str(e))
    sys.exit()

if args.otype == 'all':
    otype_list = ['bottle', 'ctd']
else:
    otype_list = [args.otype]

# Hook to extract_casts_fast.py
fn_extract = Ldir['LO'] / 'extract' / 'cast' / 'extract_casts_fast.py'
ufn_extract = Ldir['LOu'] / 'extract' / 'cast' / 'extract_casts_fast.py'
if ufn_extract.is_file():
    fn_extract = ufn_extract

# Hook to combine_obs_mod.py
fn_combine = Ldir['LO'] / 'obsmod' / 'combine_obs_mod.py'
ufn_combine = Ldir['LOu'] / 'obsmod' / 'combine_obs_mod.py'
if ufn_combine.is_file():
    fn_combine = ufn_combine

fn_plot_DO = Ldir['LOu'] / 'obsmod' / 'plot_bottom_DO_ts_multimodel.py'

def run_subprocess(cmd_list):
    proc = Po(cmd_list, stdout=Pi, stderr=Pi)
    stdout, stderr = proc.communicate()
    if len(stdout) > 0:
        print(stdout.decode())
    if len(stderr) > 0:
        print(' stderr '.center(20, '-'))
        print(stderr.decode())

for otype in otype_list:
  for year in year_list:

    # ---- Cast extractions for all model versions ----
    for gtx in gtx_list:
        print('\n' + (' otype=%s %d %s Cast Extractions ' % (otype, year, gtx)).center(60, '*') + '\n')
        tt0 = time()
        for source in source_list:
            cmd_list = ['python', str(fn_extract),
                '-gtx', gtx,
                '-ro', str(args.roms_out_num),
                '-lt', args.list_type,
                '-source', source,
                '-otype', otype,
                '-year', str(year)]
            if args.testing:
                print(cmd_list)
            else:
                proc = Po(cmd_list, stdout=Pi, stderr=Pi)
                stdout, stderr = proc.communicate()
                print(' ' + source)
                if len(stdout) > 0:
                    aa = stdout.decode().split('\n')
                    print('  %d casts processed' % (len(aa) - 1))
                else:
                    print('  no casts found')
                if len(stderr) > 0:
                    print('\n' + ' stderr '.center(60, '-'))
                    print(stderr.decode())
                sys.stdout.flush()
        print('---Time for cast extractions (%s) = %0.1f sec' % (gtx, time() - tt0))

    # ---- Combine obs+mod for all model versions ----
    for gtx in gtx_list:
        print('\n' + (' otype=%s %d %s Combining obs+mod ' % (otype, year, gtx)).center(60, '*') + '\n')
        tt0 = time()
        cmd_list = ['python', str(fn_combine),
            '-gtx', gtx,
            '-sources', args.sources,
            '-otype', otype,
            '-year', str(year)]
        if args.testing:
            print(cmd_list)
        else:
            run_subprocess(cmd_list)
        print('---Time to combine (%s) = %0.1f sec' % (gtx, time() - tt0))
        sys.stdout.flush()

# ---- Comparison plots (once per otype, spanning full year range) ----
def run_plot(label, cmd_list):
    print('\n' + (' %s ' % label).center(60, '*') + '\n')
    tt0 = time()
    if args.testing:
        print(cmd_list)
    else:
        run_subprocess(cmd_list)
    print('---Time = %0.1f sec' % (time() - tt0))
    sys.stdout.flush()

for otype in otype_list:
    base_args = (['-gtx'] + gtx_list +
                 ['-otype', otype,
                  '-year0', str(args.year0),
                  '-year1', str(args.year1)])
    if args.stations:
        base_args += ['-stations', args.stations]

    run_plot('otype=%s Bottom DO Comparison Plot' % otype,
             ['python', str(fn_plot_DO)] + base_args)
