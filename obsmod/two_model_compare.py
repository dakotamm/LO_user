"""
Driver to compare two model versions: run cast extractions and combine obs+mod
for both, then plot obs vs. both models for bottom DO and bottom detritus.

Testing on mac:
run two_model_compare -gtx0 wb1_t0_xn11abbur00 -gtx1 wb1_t1_xn11abbur00 -lt hourly -year0 2022 -test True

Production:
python two_model_compare.py -gtx0 wb1_t0_xn11abbur00 -gtx1 wb1_t1_xn11abbur00 -lt average -year0 2022 -year1 2024
"""

import argparse
from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi
import sys

from lo_tools import Lfun
import obsmod_functions as omfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx0', type=str)            # e.g. wb1_t0_xn11abbur00
parser.add_argument('-gtx1', type=str)            # e.g. wb1_t1_xn11abbur00
parser.add_argument('-ro', '--roms_out_num', type=int, default=0)
parser.add_argument('-lt', '--list_type', type=str, default='average')
parser.add_argument('-sources', type=str, default='all')
parser.add_argument('-otype', type=str, default='all')
parser.add_argument('-stations', type=str, default='')  # comma-separated station names; empty = all
parser.add_argument('-year0', type=int)
parser.add_argument('-year1', type=int, default=0)
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)

args = parser.parse_args()
argsd = args.__dict__

for a in ['gtx0', 'gtx1', 'year0']:
    if argsd[a] is None:
        print('*** Missing required argument: ' + a)
        sys.exit()

# Initialize Ldir from gtx0 (for shared paths: LO, LOo, LOu)
gridname, tag, ex_name = args.gtx0.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
for a in argsd.keys():
    if a not in Ldir.keys():
        Ldir[a] = argsd[a]

if Ldir['roms_out_num'] == 0:
    pass
elif Ldir['roms_out_num'] > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(Ldir['roms_out_num'])]

if Ldir['year1'] == 0:
    Ldir['year1'] = Ldir['year0']
year_list = list(range(Ldir['year0'], Ldir['year1'] + 1))

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
    source_list = omfun.parse_sources_arg(Ldir['sources'])
except ValueError as e:
    print('*** Bad -sources argument: ' + str(e))
    sys.exit()

if Ldir['otype'] == 'all':
    otype_list = ['bottle', 'ctd']
else:
    otype_list = [Ldir['otype']]

gtx_list = [args.gtx0, args.gtx1]

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

# Paths to comparison plot scripts (live in LO_user)
fn_plot_DO  = Ldir['LOu'] / 'obsmod' / 'plot_bottom_DO_ts_compare.py'
fn_plot_det = Ldir['LOu'] / 'obsmod' / 'plot_bottom_detritus_compare.py'

for otype in otype_list:
  for year in year_list:

    # ---- Cast extractions for both model versions ----
    for gtx in gtx_list:
        print('\n' + (' otype=%s %d %s Cast Extractions ' % (otype, year, gtx)).center(60, '*') + '\n')
        tt0 = time()
        for source in source_list:
            cmd_list = ['python', str(fn_extract),
                '-gtx', gtx,
                '-ro', str(Ldir['roms_out_num']),
                '-lt', Ldir['list_type'],
                '-source', source,
                '-otype', otype,
                '-year', str(year)]
            if Ldir['testing']:
                print(cmd_list)
            else:
                proc = Po(cmd_list, stdout=Pi, stderr=Pi)
                stdout, stderr = proc.communicate()
                print(' ' + source)
                if len(stdout) > 0:
                    a = stdout.decode()
                    aa = a.split('\n')
                    print('  %d casts processed' % (len(aa) - 1))
                else:
                    print('  no casts found')
                if len(stderr) > 0:
                    print('\n' + ' stderr '.center(60, '-'))
                    print(stderr.decode())
                sys.stdout.flush()
        print('---Time for cast extractions (%s) = %0.1f sec' % (gtx, time() - tt0))

    # ---- Combine obs+mod for both model versions ----
    for gtx in gtx_list:
        print('\n' + (' otype=%s %d %s Combining obs+mod ' % (otype, year, gtx)).center(60, '*') + '\n')
        tt0 = time()
        cmd_list = ['python', str(fn_combine),
            '-gtx', gtx,
            '-sources', Ldir['sources'],
            '-otype', otype,
            '-year', str(year)]
        if Ldir['testing']:
            print(cmd_list)
        else:
            proc = Po(cmd_list, stdout=Pi, stderr=Pi)
            stdout, stderr = proc.communicate()
            if len(stdout) > 0:
                print(' stdout '.center(20, '-'))
                print(stdout.decode())
            else:
                print('  no stdout')
            if len(stderr) > 0:
                print(' stderr '.center(20, '-'))
                print(stderr.decode())
        print('---Time to combine (%s) = %0.1f sec' % (gtx, time() - tt0))
        sys.stdout.flush()

    # ---- Bottom DO comparison plot ----
    print('\n' + (' otype=%s %d Bottom DO Comparison Plot ' % (otype, year)).center(60, '*') + '\n')
    tt0 = time()
    cmd_list = ['python', str(fn_plot_DO),
        '-gtx0', args.gtx0,
        '-gtx1', args.gtx1,
        '-otype', otype,
        '-year', str(year)]
    if args.stations:
        cmd_list += ['-stations', args.stations]
    if Ldir['testing']:
        print(cmd_list)
    else:
        proc = Po(cmd_list, stdout=Pi, stderr=Pi)
        stdout, stderr = proc.communicate()
        if len(stdout) > 0:
            print(' stdout '.center(20, '-'))
            print(stdout.decode())
        else:
            print('  no stdout')
        if len(stderr) > 0:
            print(' stderr '.center(20, '-'))
            print(stderr.decode())
    print('---Time for DO comparison plot = %0.1f sec' % (time() - tt0))
    sys.stdout.flush()

    # ---- Bottom detritus comparison plots (small and large) ----
    for det_var in ['detritus', 'Ldetritus']:
        print('\n' + (' otype=%s %d Bottom %s Comparison Plot ' % (otype, year, det_var)).center(60, '*') + '\n')
        tt0 = time()
        cmd_list = ['python', str(fn_plot_det),
            '-gtx0', args.gtx0,
            '-gtx1', args.gtx1,
            '-otype', otype,
            '-year', str(year),
            '-var', det_var]
        if args.stations:
            cmd_list += ['-stations', args.stations]
        if Ldir['testing']:
            print(cmd_list)
        else:
            proc = Po(cmd_list, stdout=Pi, stderr=Pi)
            stdout, stderr = proc.communicate()
            if len(stdout) > 0:
                print(' stdout '.center(20, '-'))
                print(stdout.decode())
            else:
                print('  no stdout')
            if len(stderr) > 0:
                print(' stderr '.center(20, '-'))
                print(stderr.decode())
        print('---Time for %s comparison plot = %0.1f sec' % (det_var, time() - tt0))
        sys.stdout.flush()
