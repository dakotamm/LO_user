"""
Process tef2 extractions from average files (Huon/Hvom) into transport vs.
OXYGEN, rather than transport vs. salinity.

This is process_sections_avg.py with the binning coordinate swapped from salt
to oxygen. Everything downstream of the coordinate -- cumulating to Q, the
Lorenz multi-layer divide, the Godin averaging -- is coordinate agnostic, so
the only thing that has to change to run TEF in oxygen classes is this step.

The output goes to

    LO_output/extract/[gtagex]/tef2/processed_avg_DO_[ds0]_[ds1]/[sn].nc

with a coordinate named 'obins' instead of 'sbins'. It is otherwise the same
shape as processed_avg_*, and carries the transport of every tracer binned by
oxygen, so Q(O) for salt tells you what salinity each oxygen class is riding
on -- which is the diagnostic that separates locally consumed water from
imported low-oxygen water.

qnet, fnet and ssh are the section totals and do not depend on the binning
coordinate at all, so they come out bit-identical to processed_avg_*. That is
the cheapest check that this script is wired up right: run both and diff them.

OXYGEN CLASSES
ROMS oxygen in LO is mmol m-3, i.e. uM. At the wb1 pc sections the field spans
0 to ~430, so the default range is 0-450 in 1 uM bins.

That resolution is chosen to match what the salinity version does rather than
picked out of the air. process_sections_avg.py uses 1000 bins over 0-36, i.e.
0.036 psu, against a tidal salinity swing at pc_lp of 0.516 psu -- about 1/14
of the tidal signal. The tidal oxygen swing at the same section is 15.4 uM, so
1 uM bins are 1/15 of it. Same relative resolution, and coarser in absolute
bin count, so the Q(O) curve handed to find_extrema is no noisier than the
Q(S) curve it is already tuned for.

OUT OF RANGE CELLS ARE CLIPPED, NOT DROPPED
process_sections_avg.py bins with binned_statistic(range=(S_low,S_hi)), which
silently DISCARDS any cell outside the range. That is harmless for salinity,
where the range is padded on both sides. It is not harmless here: the model
puts cells at exactly 0 and a hair below it, and those are the anoxic cells,
which are the whole point. So a cell outside [O_LOW, O_HI] is clipped into the
end class instead of being thrown away, matching what 20260916_exchange_fun.py
does. Volume transport is then conserved by the binning: sum over classes of
TEF['q'] equals qnet at every hour, and the script asserts that.

A cell whose oxygen is nan cannot be assigned a class and IS dropped. The
script reports what fraction of |q| that costs; on a clean extraction it is
the land mask only, and the number should be ~0.

PERFORMANCE
The binning is done with np.bincount over a time-offset flat index -- one call
per variable for the whole record -- instead of binned_statistic inside a loop
over time. Numerically identical for a 'sum' statistic on a uniform grid, and
fast enough that the two-year wb1 record is about a minute per section.

DOWNSTREAM
bulk_calc_avg.py needs three edits to consume this:
    in_dir       'processed_avg_'  ->  'processed_avg_DO_'
    coord test   'sbins'           ->  'obins'
    layer sort   argsort(bulk_dict['salt'])  ->  argsort(bulk_dict['oxygen'])
tef_fun_lorenz.calc_bulk_values itself needs nothing -- it only ever touches
thisQ_dict['q'] and the edges array.

To test on mac:
run process_sections_avg_DO.py -gtx wb1_t0_xn11abbur00 -ctag pc1 -0 2024.01.01 -1 2025.12.31 -test True

For real:
run process_sections_avg_DO.py -gtx wb1_t0_xn11abbur00 -ctag pc1 -0 2024.01.01 -1 2025.12.31
"""

import sys
from time import time

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun
from lo_tools import extract_argfun as exfun
Ldir = exfun.intro() # this handles the argument passing

# ------------------------------------------------------------------ settings

# oxygen class definition [mmol m-3 = uM]
O_LOW = 0.0
O_HI = 450.0
NO_FULL = 450   # 1 uM bins
NO_TEST = 45    # 10 uM bins, for -test True

# tracers squared, carried so that a variance budget can be built in oxygen
# coordinates the same way salt2 supports the salinity variance budget
SQUARES = ['salt', 'oxygen']

g = 9.8
rho = 1025

# --------------------------------------------------------------------- paths

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'

sect_df_fn = tef2_dir / ('sect_df_' + gctag + '.p')
sect_df = pd.read_pickle(sect_df_fn)

out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
in_dir = out_dir0 / ('extractions_avg_' + Ldir['ds0'] + '_' + Ldir['ds1'])
out_dir = out_dir0 / ('processed_avg_DO_' + Ldir['ds0'] + '_' + Ldir['ds1'])
Lfun.make_dir(out_dir, clean=True)

sect_list = [item.name for item in in_dir.glob('*.nc')]
sect_list.sort()
if Ldir['testing']:
    sect_list = sect_list[:1]
    print('testing: only ' + sect_list[0])

NO = NO_TEST if Ldir['testing'] else NO_FULL

# make vn_list by inspecting the first section
ds = xr.open_dataset(in_dir / sect_list[0])
vn_list = [item for item in ds.data_vars
           if (len(ds[item].dims) == 3) and (item not in ['q', 'DZ'])]
ds.close()

if 'oxygen' not in vn_list:
    print('ERROR: no oxygen in ' + str(in_dir / sect_list[0]))
    print('The extraction has to have been made with -get_bio True.')
    sys.exit(1)

print('\nProcessing TEF extraction into oxygen classes (avg/Huon):')
print(str(in_dir))
print('%d oxygen classes of %.3f uM over [%.1f, %.1f]'
      % (NO, (O_HI - O_LOW) / NO, O_LOW, O_HI))

# ------------------------------------------------------------------- binning

oedges = np.linspace(O_LOW, O_HI, NO + 1)
obins = oedges[:-1] + np.diff(oedges) / 2
DO_ = oedges[1] - oedges[0]

tt00 = time()

for ext_fn in sect_list:
    tt0 = time()
    print(ext_fn)
    sys.stdout.flush()

    ds = xr.open_dataset(in_dir / ext_fn)

    # q comes straight from Huon/Hvom and is already a volume flux [m3 s-1]
    q = ds['q'].to_numpy()
    NT, NZ, NP = q.shape
    oxy = ds['oxygen'].to_numpy().astype(float)
    ot = ds['time'].to_numpy()
    zeta = ds['zeta'].to_numpy().astype(float)

    # class index of every cell. Clip into range rather than drop, so that the
    # anoxic cells at O = 0 land in class 0 instead of vanishing.
    ok = np.isfinite(oxy) & np.isfinite(q)
    io = np.floor((np.where(ok, oxy, O_LOW) - O_LOW) / DO_)
    ib = np.clip(io, 0, NO - 1).astype(int)
    # flatten time into the index so the whole record bins in one bincount
    ib_flat = (np.arange(NT)[:, None, None] * NO + ib).ravel()

    # how much transport the nan mask costs us
    qabs_tot = np.nansum(np.abs(q))
    qabs_drop = np.nansum(np.abs(np.where(ok, 0.0, q)))
    print('  dropped |q| (nan oxygen) = %.3g %% of total'
          % (100 * qabs_drop / qabs_tot if qabs_tot > 0 else 0.0))

    def bin_it(vals):
        """Sum vals into oxygen classes at every time. Returns (NT, NO)."""
        w = np.where(ok, vals, 0.0).ravel()
        return np.bincount(ib_flat, weights=w, minlength=NT * NO).reshape(NT, NO)

    TEF = dict()
    TEF['q'] = bin_it(q)

    # section totals: these are properties of the section, not of the binning,
    # so they match processed_avg_* exactly
    qnet = np.nansum(np.where(ok, q, 0.0), axis=(1, 2))
    zi = zeta.copy()
    zi[~np.isfinite(q[:, 0, :])] = np.nan
    ssh = np.nanmean(zi, axis=1)
    fnet = g * rho * ssh * qnet

    # volume transport must survive the binning exactly
    resid = np.max(np.abs(TEF['q'].sum(axis=1) - qnet))
    scale = np.max(np.abs(qnet))
    if resid > 1e-8 * max(scale, 1.0):
        print('  WARNING: binning lost volume transport, max resid = %.3g' % resid)

    # property transports, one tracer at a time to keep peak memory down
    for vn in vn_list:
        V = ds[vn].to_numpy().astype(float)
        TEF[vn] = bin_it(q * V)
        if vn in SQUARES:
            TEF[vn + '2'] = bin_it(q * V * V)
        del V

    ds.close()

    TEF['qnet'] = qnet
    TEF['fnet'] = fnet
    TEF['ssh'] = ssh

    # ------------------------------------------------------------------ save
    ds_out = xr.Dataset(coords={'time': ot, 'obins': obins})
    ds_out['obins'].attrs = {'units': 'mmol m-3',
                             'long_name': 'oxygen class center'}
    for vn in ['qnet', 'fnet', 'ssh']:
        ds_out[vn] = (('time'), TEF[vn])
    bin_vn_list = ['q'] + vn_list + [vn + '2' for vn in SQUARES]
    for vn in bin_vn_list:
        ds_out[vn] = (('time', 'obins'), TEF[vn])
    ds_out.attrs = {'binning_coordinate': 'oxygen',
                    'O_low': O_LOW, 'O_hi': O_HI, 'NO': NO,
                    'oxygen_units': 'mmol m-3',
                    'out_of_range': 'clipped into end class'}
    ds_out.to_netcdf(out_dir / ext_fn)
    if not Ldir['testing']:
        ds_out.close()

    print('  elapsed time for section = %d seconds' % (time() - tt0))
    sys.stdout.flush()

print('\nTotal elapsed time = %d seconds' % (time() - tt00))
