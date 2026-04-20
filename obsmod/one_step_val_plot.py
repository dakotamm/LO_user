"""
Driver code to do a cast extraction, combine the results with bottle data, and plot the results to a png.

For -test True it plots to the screen instead of a png.

Testing on mac:
run one_step_bottle_val_plot -gtx cas7_t0_x4b -lt hourly -year0 2017 -test True

Production runs on apogee:
python one_step_bottle_val_plot.py -gtx cas7_t0_x4b -lt hourly -year0 2017 > one_step_test.log &
python one_step_bottle_val_plot.py -gtx cas7_t1_x11ab -lt average -year0 2017 > one_step_test.log &
python one_step_bottle_val_plot.py -gtx cas7_t1_x11ab -lt average -year0 2013 -year1 2024 > one_step_test.log &
"""

import argparse
from time import time
from subprocess import Popen as Po
from subprocess import PIPE as Pi
import sys

from lo_tools import Lfun
import obsmod_functions as omfun

parser = argparse.ArgumentParser()
# which run to use
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t1_x11ab
parser.add_argument('-ro', '--roms_out_num', type=int, default=0) # 1 = Ldir['roms_out1'], etc.
parser.add_argument('-lt', '--list_type', type=str, default='average') # hourly, average
parser.add_argument('-sources', type=str, default='all') # e.g. all, or other user-defined list
parser.add_argument('-otype', type=str, default='all') # observation type: bottle, ctd, or all (=bottle,ctd)
parser.add_argument('-year0', type=int) # e.g. 2019
parser.add_argument('-year1', type=int, default=0) # will set to year0 if not specified
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)

# get the args and put into Ldir
args = parser.parse_args()
argsd = args.__dict__
for a in ['gtagex', 'roms_out_num', 'list_type', 'year0']:
    if argsd[a] == None:
        print('*** Missing required argument to extract_argfun.intro(): ' + a)
        sys.exit()
gridname, tag, ex_name = args.gtagex.split('_')
# get the dict Ldir
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
# add more entries to Ldir
for a in argsd.keys():
    if a not in Ldir.keys():
        Ldir[a] = argsd[a]
# set where to look for model output
if Ldir['roms_out_num'] == 0:
    pass
elif Ldir['roms_out_num'] > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(Ldir['roms_out_num'])]
# set year range
if Ldir['year1'] == 0:
    Ldir['year1'] = Ldir['year0']
year_list = list(range(Ldir['year0'],Ldir['year1']+1))

# Set up hooks to look in LO_user
ff = 'obsmod_functions.py'
fn = Ldir['LO'] / 'obsmod' / ff
ufn = Ldir['LOu'] / 'obsmod' / ff
if ufn.is_file():
    print('Importing %s from LO_user' % (ff))
    omfun = Lfun.module_from_file('obsmod_functions', ufn)
else:
    omfun = Lfun.module_from_file('obsmod_functions', fn)

# Get the list of obs sources to use
source_list = omfun.source_dict[Ldir['sources']]

# Build list of observation types to process
if Ldir['otype'] == 'all':
    otype_list = ['bottle', 'ctd']
else:
    otype_list = [Ldir['otype']]

# Loop over otypes and years:
for otype in otype_list:
  for year in year_list:

    print('\n' + (' otype=' + otype + ' ' + str(year) + ' Cast Extractions ').center(50,'*') + '\n')

    tt0 = time()
    # - Do the cast extractions for each source
    # Hook to LO_user
    fn = Ldir['LO'] / 'extract' / 'cast' / 'extract_casts_fast.py'
    ufn = Ldir['LOu'] / 'extract' / 'cast' / 'extract_casts_fast.py'
    if ufn.is_file():
        fn = ufn
    # End Hook
    for source in source_list:
        cmd_list = ['python', str(fn),
        '-gtx', Ldir['gtagex'],
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
                print('\n' + ' stderr '.center(60,'-'))
                print(stderr.decode())
            sys.stdout.flush()
    print('\n---Time for all cast extractions = %0.1f sec' % (time()-tt0))

    # - Combine the cast extractions with obs values into a single DataFrame
    print('\n' + (' otype=' + otype + ' ' + str(year) + ' Combining obs+mod ').center(50,'*') + '\n')

    tt0 = time()
    # Hook to LO_user
    fn = Ldir['LO'] / 'obsmod' / 'combine_obs_mod.py'
    ufn = Ldir['LOu'] / 'obsmod' / 'combine_obs_mod.py'
    if ufn.is_file():
        fn = ufn
    # End Hook
    cmd_list = ['python', str(fn),
        '-gtx', Ldir['gtagex'],
        '-sources', Ldir['sources'],
        '-otype', otype,
        '-year', str(year)]

    if Ldir['testing']:
        print(cmd_list)
    else:
        proc = Po(cmd_list, stdout=Pi, stderr=Pi)
        stdout, stderr = proc.communicate()
        if len(stdout) > 0:
            print(' stdout '.center(20,'-'))
            a = stdout.decode()
            print(a)
        else:
            print('  no stdout')
        if len(stderr) > 0:
            print(' stderr '.center(20,'-'))
            print(stderr.decode())
    print('---Time to combine model and obs casts = %0.1f sec' % (time()-tt0))
    sys.stdout.flush()

    # - Plot the results and save as a png.
    print('\n' + (' otype=' + otype + ' ' + str(year) + ' Making Val Plot ').center(50,'*') + '\n')

    tt0 = time()
    # Hook to LO_user
    fn = Ldir['LO'] / 'obsmod' / 'plot_val.py'
    ufn = Ldir['LOu'] / 'obsmod' / 'plot_val.py'
    if ufn.is_file():
        fn = ufn
    # End Hook
    cmd_list = ['python', str(fn),
        '-gtx', Ldir['gtagex'],
        '-otype', otype,
        '-year', str(year)]
    if Ldir['testing']:
        print(cmd_list)
    else:
        proc = Po(cmd_list, stdout=Pi, stderr=Pi)
        stdout, stderr = proc.communicate()
        if len(stdout) > 0:
            print(' stdout '.center(20,'-'))
            a = stdout.decode()
            print(a)
        else:
            print('  no stdout')
        if len(stderr) > 0:
            print(' stderr '.center(20,'-'))
            print(stderr.decode())
    print('---Time to make val plot = %0.1f sec' % (time()-tt0))
    sys.stdout.flush()

    # - Plot DO bias maps
    print('\n' + (' otype=' + otype + ' ' + str(year) + ' DO Bias Map ').center(50,'*') + '\n')

    tt0 = time()
    # Hook to LO_user
    fn = Ldir['LO'] / 'obsmod' / 'plot_DO_bias_map.py'
    ufn = Ldir['LOu'] / 'obsmod' / 'plot_DO_bias_map.py'
    if ufn.is_file():
        fn = ufn
    # End Hook
    cmd_list = ['python', str(fn),
        '-gtx', Ldir['gtagex'],
        '-otype', otype,
        '-year', str(year)]
    if Ldir['testing']:
        print(cmd_list)
    else:
        proc = Po(cmd_list, stdout=Pi, stderr=Pi)
        stdout, stderr = proc.communicate()
        if len(stdout) > 0:
            print(' stdout '.center(20,'-'))
            a = stdout.decode()
            print(a)
        else:
            print('  no stdout')
        if len(stderr) > 0:
            print(' stderr '.center(20,'-'))
            print(stderr.decode())
    print('---Time to make DO bias map = %0.1f sec' % (time()-tt0))
    sys.stdout.flush()
