"""
Finish an extract_box.py job that was interrupted AFTER its ncks loop, from the
per-time temp files it left behind, instead of re-running the extraction.

DO NOT re-run extract_box.py for such a job: its first step is
make_dir(temp_dir, clean=True), which deletes the finished extractions.

extract_box.py, after ncks, does (and this script repeats, in order):
  1. rewrite every temp file with _FillValue = 1e20 (so ncrcat sees one fill value)
  2. ncrcat the temp files into <job>_<ds0>_<ds1>.nc
  3. add z_rho, z_w (unless -surf / -bot)
  4. compress (zlib, complevel 1)
  5. delete the temp directory
(Its "add attributes" block opens the file and closes it without writing, so it
changes nothing and is not repeated.)

What this script does differently, so it is safe to interrupt and re-run:
  0. CHECKS every temp file first; any that will not open (e.g. the one being
     written when the job was killed) or is missing is re-extracted with the same
     ncks command extract_box.py uses.
  1. runs the fill-value rewrite in parallel (-Nproc), each file written to a
     temporary name and renamed over the original, so a kill can never leave a
     half-written file. The rewrite is idempotent, so files already done are
     simply redone.
  3. adds z_rho / z_w only with -add_z True (default False: for a box that shares
     its footprint and times with another box that already carries z, e.g.
     pc_cove_mix with pc_cove, they would be ~6 GB of duplicates).
  5. deletes the temp directory only after the final file has all N times.

Run with the same arguments as the interrupted extract_box.py job, e.g. on apogee:
  python finish_box.py -gtx wb1_t0_xn11abbur00 -ro 2 -0 2024.01.01 -1 2025.12.31 -lt hourly0 -job pc_cove_mix
"""
import sys
import argparse
import multiprocessing as mp
from subprocess import Popen as Po
from subprocess import PIPE as Pi
from time import time

import numpy as np
import xarray as xr

from lo_tools import Lfun, zfun, zrfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str)
parser.add_argument('-ro', '--roms_out_num', type=int, default=0)
parser.add_argument('-0', '--ds0', type=str)
parser.add_argument('-1', '--ds1', type=str)
parser.add_argument('-lt', '--list_type', type=str)
parser.add_argument('-job', type=str)
parser.add_argument('-Nproc', type=int, default=10)
parser.add_argument('-add_z', default=False, type=Lfun.boolean_string)
parser.add_argument('-keep_temp', default=False, type=Lfun.boolean_string)   # True: leave the temp dir in place
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
for a, v in args.__dict__.items():
    if a not in Ldir.keys():
        Ldir[a] = v
if Ldir['roms_out_num'] > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(Ldir['roms_out_num'])]

out_dir = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'box'
bname = Ldir['job'] + '_' + Ldir['ds0'] + '_' + Ldir['ds1']
box_fn = out_dir / (bname + '.nc')
temp_dir = out_dir / ('temp_' + bname)
if not temp_dir.is_dir():
    print('ERROR: no temp directory ' + str(temp_dir))
    sys.exit()

# ------------------------------------------------------------------ the same file list, box indices and vn_list as extract_box.py
fn_list = Lfun.get_fn_list(Ldir['list_type'], Ldir, Ldir['ds0'], Ldir['ds1'])
N = len(fn_list)
G, S, T = zrfun.get_basic_info(fn_list[0])
Lon, Lat = G['lon_rho'][0, :], G['lat_rho'][:, 0]
upth = Ldir['LOu'] / 'extract' / 'box'
job_definitions = Lfun.module_from_file('job_definitions', upth / 'job_definitions.py')
aa, vn_list = job_definitions.get_box(Ldir['job'], Lon, Lat)
ilon0, ilon1 = zfun.find_nearest_ind(Lon, aa[0]), zfun.find_nearest_ind(Lon, aa[1])
ilat0, ilat1 = zfun.find_nearest_ind(Lat, aa[2]), zfun.find_nearest_ind(Lat, aa[3])
ds = xr.open_dataset(fn_list[0])
vn_list = ','.join([vn for vn in vn_list.split(',') if vn in ds.data_vars])
ds.close()
print('%s: %d times, vn_list %s' % (bname, N, vn_list))


def temp_fn(ii):
    return temp_dir / ('box_' + ('000000' + str(ii))[-6:] + '.nc')


def ncks(ii):
    """The extract_box.py ncks command for time ii."""
    cmd = ['ncks', '-v', vn_list,
           '-d', 'xi_rho,%d,%d' % (ilon0, ilon1), '-d', 'eta_rho,%d,%d' % (ilat0, ilat1),
           '-d', 'xi_u,%d,%d' % (ilon0, ilon1 - 1), '-d', 'eta_u,%d,%d' % (ilat0, ilat1),
           '-d', 'xi_v,%d,%d' % (ilon0, ilon1), '-d', 'eta_v,%d,%d' % (ilat0, ilat1 - 1),
           '--mk_rec_dim', 'ocean_time', '-O', str(fn_list[ii]), str(temp_fn(ii))]
    stdout, stderr = Po(cmd, stdout=Pi, stderr=Pi).communicate()
    if len(stderr) > 0:
        print(stderr.decode())


def is_bad(ii):
    """True if temp file ii is missing or will not open with every variable."""
    fn = temp_fn(ii)
    if not fn.is_file():
        return True
    try:
        with xr.open_dataset(fn) as d:
            return not all(vn in d.data_vars for vn in vn_list.split(',')) or d.sizes['ocean_time'] != 1
    except Exception:
        return True


def refill(ii):
    """extract_box.py step 1 for one file, written to a temp name then renamed (atomic)."""
    fn = temp_fn(ii)
    tmp = fn.with_suffix('.tmp.nc')
    d = xr.load_dataset(fn)
    d.to_netcdf(tmp, encoding={vn: {'_FillValue': 1e20} for vn in vn_list.split(',')})
    d.close()
    tmp.replace(fn)


if __name__ == '__main__':
    # ------------------------------------------------------------------ 0. check the temp files, re-extract any bad or missing ones
    tt0 = time()
    for f in temp_dir.glob('*.tmp.nc'):       # leftovers of an interrupted run of THIS script
        f.unlink()
    with mp.Pool(Ldir['Nproc']) as pool:
        bad = [ii for ii, b in zip(range(N), pool.map(is_bad, range(N), chunksize=200)) if b]
    extra = [f.name for f in temp_dir.glob('box_*.nc') if int(f.stem[4:]) >= N]
    print('0. checked %d temp files in %.0f s: %d bad or missing %s' % (N, time() - tt0, len(bad), bad[:20]))
    if extra:
        print('ERROR: temp files beyond the %d expected (%s ...); check -0/-1/-lt match the original job' % (N, extra[0]))
        sys.exit()
    for ii in bad:
        ncks(ii)
    if bad:
        still = [ii for ii in bad if is_bad(ii)]
        print('   re-extracted %d; still bad: %s' % (len(bad), still))
        if still:
            sys.exit()

    # ------------------------------------------------------------------ 1. one fill value everywhere
    tt0 = time()
    with mp.Pool(Ldir['Nproc']) as pool:
        pool.map(refill, range(N), chunksize=100)
    print('1. fill value rewritten in %.0f s' % (time() - tt0))

    # ------------------------------------------------------------------ 2. concatenate (exactly as extract_box.py)
    tt0 = time()
    box_fn.unlink(missing_ok=True)
    p1 = Po(['ls', str(temp_dir)], stdout=Pi)
    p2 = Po(['grep', 'box'], stdin=p1.stdout, stdout=Pi)
    stdout, stderr = Po(['ncrcat', '-p', str(temp_dir), '-O', str(box_fn)], stdin=p2.stdout, stdout=Pi, stderr=Pi).communicate()
    if len(stderr) > 0:
        print(stderr.decode())
    print('2. ncrcat in %.0f s' % (time() - tt0))

    # ------------------------------------------------------------------ 3. z variables (optional)
    ds = xr.load_dataset(box_fn)
    if Ldir['add_z']:
        tt0 = time()
        NT, NZ, NR, NC = ds.salt.shape
        ds['z_rho'] = (('ocean_time', 's_rho', 'eta_rho', 'xi_rho'), np.nan * np.ones((NT, NZ, NR, NC)))
        ds['z_w'] = (('ocean_time', 's_w', 'eta_rho', 'xi_rho'), np.nan * np.ones((NT, NZ + 1, NR, NC)))
        ds.z_rho.attrs = {'units': 'm', 'long_name': 'vertical position on s_rho grid, positive up'}
        ds.z_w.attrs = {'units': 'm', 'long_name': 'vertical position on s_w grid, positive up'}
        h = ds.h.values
        for it in range(NT):
            z_rho, z_w = zrfun.get_z(h, ds.zeta[it, :, :].values, S)
            ds['z_rho'][it, :, :, :] = z_rho
            ds['z_w'][it, :, :, :] = z_w
        print('3. z variables added in %.0f s' % (time() - tt0))
    else:
        print('3. z variables skipped (-add_z False)')

    # ------------------------------------------------------------------ 4. squeeze and compress (as extract_box.py)
    tt0 = time()
    ds = ds.squeeze()
    enc = {vn: {'zlib': True, 'complevel': 1, '_FillValue': 1e20} for vn in ds.data_vars if 'ocean_time' in ds[vn].dims}
    tmp = box_fn.with_suffix('.tmp.nc')
    ds.to_netcdf(tmp, encoding=enc)
    ds.close()
    tmp.replace(box_fn)
    print('4. compressed in %.0f s' % (time() - tt0))

    # ------------------------------------------------------------------ 5. check, then clean up
    with xr.open_dataset(box_fn) as d:
        nt = d.sizes['ocean_time']
        print('final file: %d times (expected %d), %s to %s' % (nt, N, str(d.ocean_time.values[0])[:19], str(d.ocean_time.values[-1])[:19]))
        for vn in d.data_vars:
            print('  %s %s' % (vn, str(d[vn].shape)))
    if nt == N and not Ldir['keep_temp']:
        Lfun.make_dir(temp_dir, clean=True)
        temp_dir.rmdir()
        print('5. temp directory removed')
    else:
        print('5. temp directory KEPT (%s)' % ('-keep_temp True' if nt == N else 'time count mismatch'))
    print('\nPath to file:\n' + str(box_fn))
