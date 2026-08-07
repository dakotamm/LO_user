"""
Extract BOTH velocity components at every tef2 section face, all depths, hourly.

Same driver pattern as extract_sections_avg.py, but calling the uv worker, so
the result carries the velocity VECTOR at each face rather than only the volume
flux through it:

    u_norm(time, z, p)   normal to the section, pm applied   [m s-1]
    u_tan (time, z, p)   tangential, t_hat = z_hat x n_hat   [m s-1]
    q     (time, z, p)   volume flux from Huon/Hvom          [m3 s-1]
    DZ    (time, z, p)   cell thickness                      [m]
    zeta  (time, p), h(p), dd(p)

u_norm is recoverable from the existing extractions_avg as q/(dd*DZ) -- it is
included here anyway, both as a check on that identity and so this file stands
alone. u_tan is the part that genuinely needs a new pass through the average
files: the along-section component is never stored by the standard tef2 chain.

Speed at a face is then hypot(u_norm, u_tan), and the direction relative to the
section follows from arctan2. That is what makes it possible to say whether a
face is being flushed through or swept along.

SIZE. The 3-D fields are written float32: 2 MB per face per variable for a
two-year hourly record, four such variables, so about 8 MB per face. For
wb1_pc1 that is

    pc_cp  8 faces   67 MB      sp_mid    19 faces  160 MB
    pc_lj  8 faces   67 MB      skagit_sp 32 faces  270 MB
    pc_lp 12 faces  101 MB      -------------------------
                                total              665 MB

Extraction cost is dominated by opening 17544 average files, which is the same
whether one section is wanted or all of them, so everything is extracted and
the per-section files are subset afterwards -- scp back only the sections
actually needed. The three Penn Cove sections together are 235 MB.

Output: LO_output/extract/<gtagex>/tef2/extractions_uv_<ds0>_<ds1>/<sect>.nc

To test on the mac:
run extract_sections_avg_uv.py -gtx wb1_r0_xn11ab -ctag pc0 \
    -0 2017.01.01 -1 2017.01.01 -test True

On apogee:
python extract_sections_avg_uv.py -gtx wb1_t0_xn11abbur00 -ctag pc1 \
    -0 2024.01.01 -1 2025.12.31 -Nproc 20
"""

from lo_tools import Lfun, zrfun
from lo_tools import extract_argfun as exfun
Ldir = exfun.intro()  # this handles the argument passing

from tef2_avg_fun import get_avg_fn_list
from subprocess import Popen as Po
from subprocess import PIPE as Pi
from time import time
import sys
import pandas as pd
import xarray as xr
import numpy as np

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'

sect_df_fn = tef2_dir / ('sect_df_' + gctag + '.p')
sect_df = pd.read_pickle(sect_df_fn)

fn_list = get_avg_fn_list(Ldir, Ldir['ds0'], Ldir['ds1'])
print(f'\nNumber of avg files: {len(fn_list)}')
print(f'First: {fn_list[0]}')
print(f'Last: {fn_list[-1]}')
print(f'First file exists: {fn_list[0].is_file()}\n')

out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
out_dir = out_dir0 / ('extractions_uv_' + Ldir['ds0'] + '_' + Ldir['ds1'])
temp_dir = out_dir0 / ('temp_uv_' + Ldir['ds0'] + '_' + Ldir['ds1'])
Lfun.make_dir(out_dir, clean=True)
Lfun.make_dir(temp_dir, clean=True)

if Ldir['testing']:
    fn_list = fn_list[:3]

# loop over all jobs
tt0 = time()
N = len(fn_list)
proc_list = []
for ii in range(N):
    fn = fn_list[ii]
    ii_str = ('0000' + str(ii))[-5:]
    out_fn = temp_dir / ('CC_' + ii_str + '.nc')
    cmd_list = ['python3', 'extract_sections_one_time_avg_uv.py',
                '-sect_df_fn', str(sect_df_fn),
                '-in_fn', str(fn),
                '-out_fn', str(out_fn)]
    proc = Po(cmd_list, stdout=Pi, stderr=Pi)
    proc_list.append(proc)
    if ((np.mod(ii, Ldir['Nproc']) == 0) and (ii > 0)) or (ii == N - 1):
        for proc in proc_list:
            stdout, stderr = proc.communicate()
            if len(stdout) > 0:
                print('\nSTDOUT:')
                print(stdout.decode())
                sys.stdout.flush()
            if len(stderr) > 0:
                print('\nSTDERR:')
                print(stderr.decode())
                sys.stdout.flush()
        proc_list = []
    if (np.mod(ii, 10) == 0) and ii > 0:
        print(str(ii), end=', ')
        sys.stdout.flush()
    if (np.mod(ii, 50) == 0) and (ii > 0):
        print('')
        sys.stdout.flush()
    if (ii == N - 1):
        print(str(ii))
        sys.stdout.flush()

print('Total processing time = %0.2f sec' % (time() - tt0))

# concatenate the records into one file
pp1 = Po(['ls', str(temp_dir)], stdout=Pi)
pp2 = Po(['grep', 'CC'], stdin=pp1.stdout, stdout=Pi)
temp_fn = str(temp_dir) + '/all.nc'
cmd_list = ['ncrcat', '-p', str(temp_dir), '-O', temp_fn]
proc = Po(cmd_list, stdin=pp2.stdout, stdout=Pi, stderr=Pi)
stdout, stderr = proc.communicate()
if len(stdout) > 0:
    print('\nSTDOUT:')
    print(stdout.decode())
    sys.stdout.flush()
if len(stderr) > 0:
    print('\nSTDERR:')
    print(stderr.decode())
    sys.stdout.flush()

# add DZ, exactly as extract_sections_avg.py does, then split by section
ds1 = xr.open_dataset(temp_fn)
S = zrfun.get_basic_info(fn_list[0], only_S=True)
eta = ds1.zeta.values.squeeze()                    # packed (t, p)
if eta.ndim == 1:                                  # a single time step
    eta = eta.reshape(1, -1)
NT, NP = eta.shape
hh = ds1.h.values.squeeze().reshape(1, NP) * np.ones((NT, 1))
zw = zrfun.get_z(hh, eta, S, only_w=True)
dz = np.diff(zw, axis=0)                           # packed (z, t, p)
DZ = np.transpose(dz, (1, 0, 2))                   # packed (t, z, p)

sect_list = list(sect_df.sn.unique())
sect_list.sort()
for sn in sect_list:
    ii = np.where(sect_df.sn == sn)[0]
    this_ds = ds1.isel(p=ii)
    this_ds['DZ'] = (('time', 'z', 'p'), DZ[:, :, ii].astype('float32'))
    this_ds.attrs['note'] = (
        'both velocity components at each face of section ' + sn + '. u_norm '
        'is normal to the section with pm applied; u_tan is tangential with '
        't_hat = z_hat x n_hat. Hourly, NOT tidally filtered. z index 0 = bed, '
        '-1 = surface. speed = hypot(u_norm, u_tan).')
    this_fn = out_dir / (sn + '.nc')
    this_ds.to_netcdf(this_fn)
    print('saved %s  (%d faces)' % (this_fn.name, len(ii)))
ds1.close()

# clean up the temp dir
if not Ldir['testing']:
    Lfun.make_dir(temp_dir, clean=True)
    temp_dir.rmdir()
