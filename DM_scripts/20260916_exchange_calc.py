"""
Eulerian and isohaline (TEF) exchange flow at the wb1_pc1 sections, following
Chen et al. 2012 (doi:10.1175/JPO-D-11-086.1).

All the machinery is in 20260916_exchange_fun.py -- read its docstring for the
definitions. This script just runs it for every section and packs the daily
time series into one file:

    LO_output/extract/[gtagex]/tef2/exchange_[ds0]_[ds1]_wb1_pc1.nc

Three exchange flows are reported for each section, all on the same daily
(Godin averaged, noon subsampled) axis as the existing bulk_avg_* files:

    tef_*    isohaline, Q(S) divided into layers by the Lorenz method. This is
             a reimplementation of process_sections_avg + bulk_calc_avg in one
             pass; it reproduces the existing bulk_avg_* output to machine
             precision, which is the check that it is wired up right.
    eu9_*    Eulerian in isohaline coordinates -- Chen et al. Eq. (9), and the
             comparison the paper actually makes. Same Q(S) machinery as tef_*
             but fed the SUBTIDAL velocity and salinity, so the difference
             between tef_ and eu9_ is exactly the tidal flux FT.
    eulz_*   Eulerian, section summed over p then split by the sign of <q>(z).
             The textbook two-layer Eulerian.
    eul_*    Eulerian, split by the sign of <q> in each (z,p) cell. Penn Cove's
             mouth exchange is largely lateral, so this is the honest Eulerian
             here -- but it is an upper bound, since it counts every cell of a
             given sign no matter how small, and unlike eulz_Qin it is NOT
             bounded by Qprism.

Sign convention: positive is INTO Penn Cove everywhere in the output, which is
a flip of the section's own positive direction for pc_lp/pc_cp/pc_lj. See
INFLOW_SIGN in the function module.

run 20260916_exchange_calc.py
"""
import argparse
import importlib.util
import sys
from pathlib import Path
from time import time

import numpy as np
import xarray as xr

from lo_tools import Lfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', default='wb1_t0_xn11abbur00', type=str)
parser.add_argument('-ctag', default='pc1', type=str)
parser.add_argument('-0', '--ds0', default='2024.01.01', type=str)
parser.add_argument('-1', '--ds1', default='2025.12.31', type=str)
parser.add_argument('-sect', default='pc_lp,pc_lj,pc_cp', type=str,
                    help='comma separated section names')
parser.add_argument('-NS', default=1000, type=int, help='number of salinity bins')
args = parser.parse_args()

# load the sibling function module by path, since DM_scripts is not a package
# and the file name starts with a digit
_fn = Path(__file__).parent / '20260916_exchange_fun.py'
_spec = importlib.util.spec_from_file_location('exchange_fun', _fn)
xfun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xfun)

Ldir = Lfun.Lstart(gridname='wb1')
in_dir = xfun.section_dir(args.gtagex, args.ds0, args.ds1, Ldir=Ldir)
out_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'tef2'
          / ('exchange_' + args.ds0 + '_' + args.ds1 + '_wb1_' + args.ctag + '.nc'))

sect_list = [s.strip() for s in args.sect.split(',') if s.strip()]

# scalars carried straight through from the calculation
VEC = ['qnet', 'qprism', 'F_total', 'F_mean', 'F_exch', 'F_tidal', 's0', 'area']
BULK = ['Qin', 'Qout', 'sin', 'sout']

R = dict()
time_lp = None
good = []

for sn in sect_list:
    if not (in_dir / (sn + '.nc')).is_file():
        print('SKIP %s -- no extraction on this machine (it is on apogee)' % sn)
        continue
    tt0 = time()
    print('Working on ' + sn)
    sys.stdout.flush()

    S = xfun.load_section(sn, args.gtagex, args.ds0, args.ds1, Ldir=Ldir)
    T = xfun.tef_bulk(S, NS=args.NS)
    E9 = xfun.eulerian_isohaline(S, NS=args.NS)
    Ec = xfun.eulerian_bulk(S, mode='cell')
    Ez = xfun.eulerian_bulk(S, mode='vertical')

    if time_lp is None:
        time_lp = T['time']
    elif not np.array_equal(time_lp, T['time']):
        raise ValueError('time axes differ between sections')

    d = dict()
    for k in BULK:
        d['tef_' + k] = T[k]
        d['eu9_' + k] = E9[k]
        d['eul_' + k] = Ec[k]
        d['eulz_' + k] = Ez[k]
    d['qnet'] = T['qnet']
    d['qprism'] = T['qprism']
    for k in ['F_total', 'F_mean', 'F_exch', 'F_tidal', 's0', 'area']:
        d[k] = Ec[k]
    R[sn] = d
    good.append(sn)
    print('  %d days, elapsed %d sec' % (len(T['time']), time() - tt0))
    sys.stdout.flush()

if len(good) == 0:
    print('Nothing to do.')
    sys.exit()

ds = xr.Dataset(coords={'time': time_lp, 'sect': good})
for k in [p + v for p in ['tef_', 'eu9_', 'eul_', 'eulz_'] for v in BULK] + VEC:
    ds[k] = (('time', 'sect'), np.stack([R[sn][k] for sn in good], axis=1))

# Chen et al.'s tidal conversion parameter, the fraction of the tidal inflow
# that ends up as net exchange
for p in ['tef_', 'eu9_', 'eul_', 'eulz_']:
    ds[p + 'conv'] = ds[p + 'Qin'] / ds['qprism']
# and the delta s that goes with each exchange flow
for p in ['tef_', 'eu9_', 'eul_', 'eulz_']:
    ds[p + 'ds'] = ds[p + 'sin'] - ds[p + 'sout']

ds['qnet'].attrs['note'] = 'subtidal net transport, positive into Penn Cove'
ds['qprism'].attrs['note'] = '1/2 <|qnet - lowpass(qnet)|>, as in bulk_calc_avg.py'
ds['F_tidal'].attrs['note'] = 'subtidal salt flux missed by the Eulerian analysis'
ds.attrs['reference'] = 'Chen et al. 2012, doi:10.1175/JPO-D-11-086.1'
ds.attrs['sign'] = 'positive is INTO Penn Cove'
ds.attrs['gtagex'] = args.gtagex

ds.to_netcdf(out_fn)
ds.close()
print('\nSaved ' + str(out_fn))
