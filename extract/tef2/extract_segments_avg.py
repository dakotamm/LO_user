"""
A tool to extract hourly time series of volume and selected tracers and derived quantities
for budgets in the segments.

To test on mac:
run extract_segments.py -gtx cas7_trapsV00_meV00 -ctag c0 -get_bio True -riv trapsV00 -0 2017.07.04 -1 2017.07.06

Performance: 2.5 minutes for test

Use -get_bio True to get all bio tracers
Use -test True for more screen output and a shorter extraction

"""
from lo_tools import Lfun, zrfun
from lo_tools import extract_argfun as exfun
Ldir = exfun.intro() # this handles the argument passing

import sys
from time import time
import numpy as np
import pickle
import pandas as pd
from subprocess import Popen as Po
from subprocess import PIPE as Pi
import xarray as xr

from datetime import datetime, timedelta

ds_fmt = '%Y.%m.%d'


tt00 = time()

# some names for convenience
Ldir['gctag'] = Ldir['gridname'] + '_' + Ldir['collection_tag']
Ldir['date_range'] = Ldir['ds0'] + '_' + Ldir['ds1']
long_tag = Ldir['date_range'] + '_' + Ldir['gctag'] + '_' + Ldir['riv']

# get segment info
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'
seg_info_dict_fn = tef2_dir / ('seg_info_dict_' + Ldir['gctag'] + '_' + Ldir['riv'] + '.p')

# output names and places
out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
temp_dir = out_dir0 / ('segments_temp_' + long_tag)
out_name = 'segments_avg_' + long_tag + '.nc'
out_fn = out_dir0 / out_name
out_fn.unlink(missing_ok=True) # make sure output file does not exist
Lfun.make_dir(out_dir0)
Lfun.make_dir(temp_dir, clean=True)

print(' Doing segment extraction for '.center(60,'='))
print(' out_dir0 = ' + str(out_dir0))
print(' out_name = ' + out_name)
print(' temp dir = ' + temp_dir.name)

### DM modified 20260320 for average hourly from Lfun

def date_list_utility(dt0, dt1, daystep=1):
    """
    INPUT: start and end datetimes
    OUTPUT: list of LiveOcean formatted dates
    """
    date_list = []
    dt = dt0
    while dt <= dt1:
        date_list.append(dt.strftime(ds_fmt))
        dt = dt + timedelta(days=daystep)
    return date_list

def fn_list_utility(dt0, dt1, Ldir, hourmax=24, his_num=2):
    """
    INPUT: start and end datetimes
    OUTPUT: list of all history files expected to span the dates
    - list items are Path objects
    """
    dir0 = Ldir['roms_out'] / Ldir['gtagex']
    fn_list = []
    date_list = date_list_utility(dt0, dt1)
    if his_num == 1:
        # New scheme 2023.10.05 to work with new or continuation start_type,
        # by assuming we want to start with ocean_his_0001.nc of dt0
        fn_list.append(dir0 / ('f'+dt0.strftime(ds_fmt)) / 'ocean_his_0001.nc')
        for dl in date_list:
            f_string = 'f' + dl
            hourmin = 1
            for nhis in range(hourmin+1, hourmax+2):
                nhiss = ('0000' + str(nhis))[-4:]
                fn = dir0 / f_string / ('ocean_his_' + nhiss + '.nc')
                fn_list.append(fn)
    else:
        # For any other value of his_num we assume this is a perfect start_type
        # and so there is no ocean_his_0001.nc on any day and we start with
        # ocean_his_0025.nc of the day before.
        dt00 = (dt0 - timedelta(days=1))
        fn_list.append(dir0 / ('f'+dt00.strftime(ds_fmt)) / 'ocean_his_0025.nc')
        for dl in date_list:
            f_string = 'f' + dl
            hourmin = 1
            for nhis in range(hourmin+1, hourmax+2):
                nhiss = ('0000' + str(nhis))[-4:]
                fn = dir0 / f_string / ('ocean_his_' + nhiss + '.nc')
                fn_list.append(fn)
    return fn_list

# DM created
def fn_list_utility_avg(dt0, dt1, Ldir, hourmax=23, his_num=2): #DM modified  20260320
    """
    INPUT: start and end datetimes
    OUTPUT: list of all history files expected to span the dates
    - list items are Path objects
    """
    dir0 = Ldir['roms_out'] / Ldir['gtagex']
    fn_list = []
    date_list = date_list_utility(dt0, dt1)
    if his_num == 1:
        # New scheme 2023.10.05 to work with new or continuation start_type,
        # by assuming we want to start with ocean_his_0001.nc of dt0
        fn_list.append(dir0 / ('f'+dt0.strftime(ds_fmt)) / 'ocean_avg_0001.nc')
        for dl in date_list:
            f_string = 'f' + dl
            hourmin = 1
            for nhis in range(hourmin+1, hourmax+2):
                nhiss = ('0000' + str(nhis))[-4:]
                fn = dir0 / f_string / ('ocean_avg_' + nhiss + '.nc')
                fn_list.append(fn)
    else:
        # For any other value of his_num we assume this is a perfect start_type
        # and so there is no ocean_his_0001.nc on any day and we start with
        # ocean_his_0025.nc of the day before.
        dt00 = (dt0 - timedelta(days=1))
        fn_list.append(dir0 / ('f'+dt00.strftime(ds_fmt)) / 'ocean_avg_0025.nc')
        for dl in date_list:
            f_string = 'f' + dl
            hourmin = 1
            for nhis in range(hourmin+1, hourmax+2):
                nhiss = ('0000' + str(nhis))[-4:]
                fn = dir0 / f_string / ('ocean_avg_' + nhiss + '.nc')
                fn_list.append(fn)
    return fn_list

def get_fn_list(list_type, Ldir, ds0, ds1, his_num=2):
    """
    INPUT:
    A function for getting lists of history files.
    List items are Path objects
    
    NEW 2023.10.05: for list_type = 'hourly', if you pass his_num = 1
    it will start with ocean_his_0001.nc on the first day instead of the default which
    is to start with ocean_his_0025.nc on the day before.

    NEW 2025.06.20: for list_type = 'hourly0'
    which will start with ocean_his_0001.nc on the first day instead of the default which
    is to start with ocean_his_0025.nc on the day before.
    This is identical to passing his_num = 1, but may be more convenient, especially
    as we move to "continuation" start_type, which always writes an 0001 file.
    """
    dt0 = datetime.strptime(ds0, ds_fmt)
    dt1 = datetime.strptime(ds1, ds_fmt)
    dir0 = Ldir['roms_out'] / Ldir['gtagex']
    if list_type == 'snapshot':
        # a single file name in a list
        his_string = ('0000' + str(his_num))[-4:]
        fn_list = [dir0 / ('f' + ds0) / ('ocean_his_' + his_string + '.nc')]
    elif list_type == 'hourly':
        # list of hourly files over a date range
        fn_list = fn_list_utility(dt0,dt1,Ldir,his_num=his_num)
    elif list_type == 'hourly0':
        # list of hourly files over a date range, starting with 0001 of dt0.
        fn_list = fn_list_utility(dt0,dt1,Ldir,his_num=1)
    elif list_type == 'daily':
        # list of history file 21 (Noon PST) over a date range
        fn_list = []
        date_list = date_list_utility(dt0, dt1)
        for dl in date_list:
            f_string = 'f' + dl
            fn = dir0 / f_string / 'ocean_his_0021.nc'
            fn_list.append(fn)
    elif list_type == 'lowpass':
        # list of lowpassed files (Noon PST) over a date range
        fn_list = []
        date_list = date_list_utility(dt0, dt1)
        for dl in date_list:
            f_string = 'f' + dl
            fn = dir0 / f_string / 'lowpassed.nc'
            fn_list.append(fn)
    elif list_type == 'average':
        # list of daily averaged files (Noon PST) over a date range
        fn_list = []
        date_list = date_list_utility(dt0, dt1)
        for dl in date_list:
            f_string = 'f' + dl
            fn = dir0 / f_string / 'ocean_avg_0001.nc'
            fn_list.append(fn)
    elif list_type == 'hourlyaverage':
        # DM created 
        fn_list = fn_list_utility_avg(dt0,dt1,Ldir,his_num=his_num)
    elif list_type == 'weekly':
        # like "daily" but at 7-day intervals
        fn_list = []
        date_list = date_list_utility(dt0, dt1, daystep=7)
        for dl in date_list:
            f_string = 'f' + dl
            fn = dir0 / f_string / 'ocean_his_0021.nc'
            fn_list.append(fn)
    elif list_type == 'allhours':
        # a list of all the history files in a directory
        # (this is the only list_type that actually finds files)
        in_dir = dir0 / ('f' + ds0)
        fn_list = [ff for ff in in_dir.glob('ocean_his*nc')]
        fn_list.sort()

    return fn_list

fn_list = get_fn_list('hourlyaverage', Ldir, Ldir['ds0'], Ldir['ds1'], his_num=Ldir['his_num']) #DM modified 20260320 for averageif Ldir['testing']:
if Ldir['testing']:
    fn_list = fn_list[:3]
    
print('Doing initial data extraction:')
# We do extractions one hour at a time, as separate subprocess jobs.
# Files are saved to temp_dir.
tt000 = time()
proc_list = []
N = len(fn_list)
for ii in range(N):
    fn = fn_list[ii]
    
    d = fn.parent.name.replace('f','')
    nhis = int(fn.name.split('.')[0].split('_')[-1])
    
    cmd_list = ['python3', 'extract_segments_one_time.py',
            '-in_fn',str(fn),
            '-out_dir',str(temp_dir),
            '-file_num',str(ii),
            '-seg_fn',str(seg_info_dict_fn),
            '-get_bio', str(Ldir['get_bio']),
            '-test', str(Ldir['testing'])]
    proc = Po(cmd_list, stdout=Pi, stderr=Pi)
    proc_list.append(proc)

    Nproc = Ldir['Nproc']
    if ((np.mod(ii,Nproc) == 0) and (ii > 0)) or (ii == N-1):
        tt0 = time()
        for proc in proc_list:
            stdout, stderr = proc.communicate()
            if Ldir['testing']:
                print(' sdtout '.center(60,'-'))
                print(stdout.decode())
            if len(stderr) > 0:
                print(' stderr '.center(60,'-'))
                print(stderr.decode())
        print(' - %d out of %d: %d took %0.2f sec' % (ii, N, Nproc, time()-tt0))
        sys.stdout.flush()
        proc_list = []
print('Total elapsed time = %0.2f sec' % (time()-tt000))

"""
Now we repackage all these single-time extractions into an xarray Dataset.  This is
a little tricky because for the Dataset we want to have it composed of DataArrays that
are variable(time, segment), but what we are starting from is a collection of
time(segment, variable) pandas DataFrames.

We will proceed by using a two step process, first concatenating all the pandas DataFrames
into an xarray DataArray with dimensions(time, segment, variable).  Then we will
convert this to an xarray Dataset with a collection of variable(time,segment) DataArrays.

I'm sure there is a more clever way to do this in xarray, but I am not yet proficient
enough with that module.
"""

# get a list of all our pandas DataFrames
A_list = list(temp_dir.glob('A*.p'))
A_list.sort()
# make a list of the datetimes and form a time index
ot_list = []
for fn in fn_list:
    ds = xr.open_dataset(fn)
    ot = ds['ocean_time'].values[0]
    ot_list.append(ot)
ot_ind = pd.Index(ot_list)
# make a list of the output as DataArrays
x_list = []
for A_fn in A_list:
    A = pd.read_pickle(A_fn)
    x_list.append(xr.DataArray(A, dims=('seg','vn')))
# and concatenate that list into a single DataArray, with time as the concatenating dimension
da = xr.concat(x_list, pd.Index(ot_ind, name='time'))
# repackage the DataArray as a Dataset
vns = da.coords['vn'].values
segs = da.coords['seg'].values
times = da.coords['time'].values
ds = xr.Dataset(coords={'time': times,'seg': segs})
for vn in vns:
    v = da.sel(vn=vn).values
    ds[vn] = (('time','seg'), v)
# save it to NetCDF
ds.to_netcdf(out_fn)
ds.close()

# Clean up
if not Ldir['testing']:
    Lfun.make_dir(temp_dir, clean=True)
    temp_dir.rmdir()

if Ldir['testing']:
    # check results
    seg_info_dict = pd.read_pickle(seg_info_dict_fn) # same as using pickle.load()
    dd = xr.open_dataset(out_fn)
    print(dd.salt.sel(seg=list(seg_info_dict.keys())[0]).values)
    dd.close()


