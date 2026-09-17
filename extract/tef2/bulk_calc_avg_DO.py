"""
Calculate a TEF time series in OXYGEN coordinates using Marvin Lorenz'
multi-layer code.

This is bulk_calc_avg.py reading processed_avg_DO_* instead of processed_avg_*.
tef_fun_lorenz.calc_bulk_values needs no changes at all -- it only ever touches
thisQ_dict['q'] and the edges array, so the coordinate it is dividing in is
whatever the upstream binning used. The edits here are bookkeeping:

    in_dir      'processed_avg_'  ->  'processed_avg_DO_'
    coord test  'sbins'           ->  'obins'
    layer sort  argsort('salt')   ->  argsort('oxygen')

out_dir is bulk_avg_DO_[ds0]_[ds1], so this sits alongside the salinity
bulk_avg_* rather than overwriting it.

SIGN CONVENTION
Raw section frame, positive in the section's own direction (pm = +1), exactly
as bulk_avg_* is. For the wb1 pc sections that points OUT of Penn Cove, so the
'in' layers of this file are the EASTWARD ones. The INFLOW_SIGN flip lives in
DM_scripts/20260916_exchange_fun.py, not here.

LAYER AXIS IS GROWN TO FIT
bulk_calc_avg.py preallocates 31 layers and raises a broadcast error if the
divider ever finds more. Q(O) at pc_lp runs 4 layers median and 19 max, so 31
is not tight, but the failure mode is a crash partway through a two-year run
rather than a warning, so the rows are collected first and the layer axis is
sized afterwards. nlay is the floor, not the cap -- same choice already made in
20260916_exchange_fun.divide_Q_of_S.

QPRISM IS UNCHANGED
qnet, fnet, ssh and qprism are section totals and do not depend on the binning
coordinate, so they come out identical to bulk_avg_*. Another free wiring check.

DEBUG FIGURE IS SAVED, NOT SHOWN
-test True still builds the Q(O) / -dQ/dO diagnostic figure, but writes it to
bulk_avg_DO_*/debug_Q_of_O_[sn].png instead of calling plt.show().

To test on mac (deep debugging: first day only, with a saved diagnostic figure):
run bulk_calc_avg_DO.py -gtx wb1_t0_xn11abbur00 -ctag pc1 -0 2024.01.01 -1 2025.12.31 -test True

For real:
run bulk_calc_avg_DO.py -gtx wb1_t0_xn11abbur00 -ctag pc1 -0 2024.01.01 -1 2025.12.31
"""

import sys
import matplotlib
matplotlib.use('Agg')  # debug figure is saved, never shown -- see below
import matplotlib.pyplot as plt
import numpy as np
from time import time
import pandas as pd
import xarray as xr

from lo_tools import Lfun, zfun
import extract.tef2.archive.tef_fun_lorenz as tfl

from lo_tools import extract_argfun as exfun
Ldir = exfun.intro() # this handles the argument passing

# the coordinate the upstream binning used
COORD = 'obins'
# the tracer the layers get sorted by, which has to be the binning coordinate
SORT_VN = 'oxygen'

gctag = Ldir['gridname'] + '_' + Ldir['collection_tag']
tef2_dir = Ldir['LOo'] / 'extract' / 'tef2'

sect_df_fn = tef2_dir / ('sect_df_' + gctag + '.p')
sect_df = pd.read_pickle(sect_df_fn)

out_dir0 = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'
in_dir = out_dir0 / ('processed_avg_DO_' + Ldir['ds0'] + '_' + Ldir['ds1'])
out_dir = out_dir0 / ('bulk_avg_DO_' + Ldir['ds0'] + '_' + Ldir['ds1'])
# NOTE clean only on a real run. The house scripts always clean, but here a
# -test True pass after a real pass would wipe the two-year output to write one
# debug figure, and a real pass after a test pass wipes the figure. Testing
# writes into the existing directory instead.
Lfun.make_dir(out_dir, clean=not Ldir['testing'])

if not in_dir.is_dir():
    print('ERROR: no ' + str(in_dir))
    print('Run process_sections_avg_DO.py first.')
    sys.exit(1)

sect_list = [item.name for item in in_dir.glob('*.nc')]
sect_list.sort()
if Ldir['testing']:
    sect_list = sect_list[:1]
    print('testing: only ' + sect_list[0])

# ---------

tt00 = time()

# setting Ldir['testing'] = True runs a deep debugging step, in which you only process
# the first day, and look at the details of the multi-layer bulk calculation,
# both graphically and as screen output.

for snp in sect_list:
    tt0 = time()

    print('Working on ' + snp)
    sys.stdout.flush()
    out_fn = out_dir / snp

    # load the processed Dataset for this section

    ds = xr.open_dataset(in_dir / snp)

    # Create the absolute value of the net transport (to make Qprism)
    # but first remove the low-passed transport (like Qr)
    qnet_lp = zfun.lowpass(ds.qnet.values, f='godin', nanpad=False)
    qabs = np.abs(ds.qnet.values - qnet_lp)

    # Tidal averaging, subsample, and cut off nans
    pad = 36
    # this pad is more than is required for the nans from the godin filter (35),
    # but, when combined with the subsampling we end up with fields at Noon of
    # each day (excluding the first and last days of the record)
    TEF_lp = dict() # temporary storage
    vn_list = []
    vec_list = []
    for vn in ds.data_vars:
        if ('time' in ds[vn].coords) and (COORD in ds[vn].coords):
            TEF_lp[vn] = zfun.lowpass(ds[vn].values, f='godin')[pad:-pad+1:24, :]
            vn_list.append(vn)
        elif ('time' in ds[vn].coords) and (COORD not in ds[vn].coords):
            TEF_lp[vn] = zfun.lowpass(ds[vn].values, f='godin')[pad:-pad+1:24]
            vec_list.append(vn)
    time_lp = ds.time.values[pad:-pad+1:24]
    obins = ds[COORD].values
    TEF_lp['qabs'] = zfun.lowpass(qabs, f='godin')[pad:-pad+1:24]
    # Add the qprism time series.
    # Conceptually, qprism is the maximum possible exchange flow if all
    # the flood tide made Qin and all the ebb tide made Qout.
    # If you go through the trigonometry you find that qprism = 1/2 <qabs>.
    TEF_lp['qprism'] = TEF_lp['qabs'].copy()/2
    vec_list += ['qabs', 'qprism']
    O_units = ds[COORD].attrs.get('units', 'mmol m-3')
    ds.close()

    if SORT_VN not in vn_list:
        print('ERROR: no ' + SORT_VN + ' in ' + str(in_dir / snp))
        sys.exit(1)

    # get sizes and make oedges (the edges of obins)
    DO_ = obins[1] - obins[0]
    oedges = np.concatenate((obins - DO_/2, [obins[-1] + DO_/2]))
    NT = len(time_lp)
    NO = len(oedges)

    # calculate all transports integrated over oxygen, e.g. Q(O) = integral(q dO)
    omat = np.zeros((NT, NO))
    Q_dict = dict()
    for vn in vn_list:
        Q_dict[vn] = omat.copy()
        Q_dict[vn][:,:-1] = np.fliplr(np.cumsum(np.fliplr(TEF_lp[vn]), axis=1))

    nlay = 31 # floor on the layer axis, not a cap -- see docstring

    if Ldir['testing']:
        plt.close('all')
        dd_list = [0]
        print_info = True
    else:
        dd_list = range(NT)
        print_info = False

    # collect the layers first, size the layer axis afterwards
    rows = dict()

    for dd in dd_list:

        thisQ_dict = dict()
        for vn in vn_list:
            thisQ_dict[vn] = Q_dict[vn][dd,:]

        if print_info == True:
            print('\n**** dd = %d ***' % (dd))

        out_tup = tfl.calc_bulk_values(oedges, thisQ_dict, vn_list, print_info=print_info)
        in_dict, out_dict, div_oxy, ind, minmax = out_tup

        if print_info == True:
            print(' ind = %s' % (str(ind)))
            print(' minmax = %s' % (str(minmax)))
            print(' div_oxy = %s' % (str(div_oxy)))
            print(' Q_in_m = %s' % (str(in_dict['q'])))
            print(' O_in_m = %s' % (str(in_dict[SORT_VN])))
            print(' Q_out_m = %s' % (str(out_dict['q'])))
            print(' O_out_m = %s' % (str(out_dict[SORT_VN])))

            fig = plt.figure(figsize=(12,8))

            ax = fig.add_subplot(121)
            ax.plot(Q_dict['q'][dd,:], oedges,'.k')
            min_mask = minmax=='min'
            max_mask = minmax=='max'
            ax.plot(Q_dict['q'][dd,ind[min_mask]], oedges[ind[min_mask]],'*b')
            ax.plot(Q_dict['q'][dd,ind[max_mask]], oedges[ind[max_mask]],'*r')
            ax.grid(True)
            ax.set_title('Q(O) Time index = %d' % (dd))
            ax.set_ylim(oedges[0]-1, oedges[-1]+1)
            ax.set_ylabel('Oxygen [' + O_units + ']')

            ax = fig.add_subplot(122)
            ax.plot(TEF_lp['q'][dd,:], obins)
            ax.grid(True)
            ax.set_title('-dQ/dO')
            ax.set_ylabel('Oxygen [' + O_units + ']')

            fig_fn = out_dir / ('debug_Q_of_O_' + snp.replace('.nc', '') + '.png')
            fig.savefig(fig_fn, transparent=True)
            plt.close(fig)
            print(' saved ' + str(fig_fn))

        bulk_dict = dict()
        for vn in vn_list:
            bulk_dict[vn] = np.array(in_dict[vn] + out_dict[vn])
        ii = np.argsort(bulk_dict[SORT_VN])
        rows[dd] = {vn: bulk_dict[vn][ii] for vn in vn_list}

    # now size the layer axis to whatever the divider actually found
    NL = max(nlay, max([len(r['q']) for r in rows.values()] + [0]))
    if NL > nlay:
        print('  note: grew layer axis to %d' % (NL))
    MLO = {vn: np.nan * np.ones((NT, NL)) for vn in vn_list}
    for dd, r in rows.items():
        n = len(r['q'])
        if n > 0:
            for vn in vn_list:
                MLO[vn][dd, :n] = r[vn]

    for vn in vec_list:
        MLO[vn] = TEF_lp[vn].copy()

    # Pack results in a Dataset and then save to NetCDF
    ds = xr.Dataset(coords={'time': time_lp,'layer': np.arange(NL)})
    for vn in vn_list:
        ds[vn] = (('time','layer'), MLO[vn])
    for vn in vec_list:
        ds[vn] = (('time'), MLO[vn])
    ds.attrs = {'binning_coordinate': 'oxygen',
                'oxygen_units': O_units,
                'sign_convention': 'raw section frame (pm=+1), NOT flipped to inflow'}
    # save it to NetCDF -- but NOT when testing, where only dd=0 was processed
    # and every other day is nan. The house script writes it anyway, which
    # leaves a file that looks like a finished run and is not one.
    if Ldir['testing']:
        print('  testing: NOT writing ' + str(out_fn) + ' (only dd=0 computed)')
    else:
        ds.to_netcdf(out_fn)
        ds.close()
    print('  elapsed time for section = %d seconds' % (time()-tt0))
    sys.stdout.flush()

print('\nTotal elapsed time = %d seconds' % (time()-tt00))
