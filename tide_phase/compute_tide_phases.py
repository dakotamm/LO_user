"""
Compute tidal phase labels from an extracted zeta time series
and (optionally) qprism from TEF bulk output.

Reads output from extract_zeta_ts.py, runs tide_phase_fun detectors,
and saves a phase-label NetCDF.

Usage examples
--------------
    # Signal-based detection using zeta and qprism:
    python compute_tide_phases.py -gtx wb1_r0_xn11b -ctag pc0 \
        -sect_name pc0 -0 2017.09.01 -1 2017.09.30 -method signal

    # UTide harmonic detection:
    python compute_tide_phases.py -gtx wb1_r0_xn11b -ctag pc0 \
        -sect_name pc0 -0 2017.09.01 -1 2017.09.30 -method utide \
        -lat 48.23

    # Both methods:
    python compute_tide_phases.py -gtx wb1_r0_xn11b -ctag pc0 \
        -sect_name pc0 -0 2017.09.01 -1 2017.09.30 -method both \
        -lat 48.23
"""

import argparse
import sys
import numpy as np
import xarray as xr
from time import time as timer

from lo_tools import Lfun

# Import from local DM_scripts
import tide_phase_fun as tpf


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='Compute tidal phase labels from zeta/qprism.')
    parser.add_argument('-gtx', '--gtagex', type=str, required=True)
    parser.add_argument('-ro', '--roms_out_num', type=int, default=0)
    parser.add_argument('-0', '--ds0', type=str, required=True)
    parser.add_argument('-1', '--ds1', type=str, required=True)
    parser.add_argument('-ctag', '--collection_tag', type=str, default=None)
    parser.add_argument('-sect_name', type=str, default=None,
                        help='Section name (matches extract_zeta_ts label)')
    parser.add_argument('-label', type=str, default=None,
                        help='Custom label (for point extractions, e.g. lon-122.7_lat48.23)')
    parser.add_argument('-method', type=str, default='signal',
                        choices=['signal', 'utide', 'both'])
    parser.add_argument('-lat', type=float, default=None,
                        help='Latitude for UTide (required if method=utide or both)')
    parser.add_argument('-fe_method', type=str, default='zeta',
                        choices=['qnet', 'zeta'],
                        help='Flood/ebb detection variable for signal method')
    parser.add_argument('-sn_percentile', type=float, default=50,
                        help='Percentile threshold for spring/neap (default 50)')
    parser.add_argument('-test', '--testing', type=Lfun.boolean_string,
                        default=False)

    args = parser.parse_args()

    # Determine the label for finding input files
    if args.sect_name is not None:
        args.label = args.sect_name
    elif args.label is None:
        print('ERROR: provide -sect_name or -label')
        sys.exit(1)

    if args.method in ('utide', 'both') and args.lat is None:
        print('ERROR: -lat is required for UTide method')
        sys.exit(1)

    # Build Ldir
    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    for k, v in vars(args).items():
        if k not in Ldir:
            Ldir[k] = v
    if Ldir['roms_out_num'] > 0:
        Ldir['roms_out'] = Ldir['roms_out' + str(Ldir['roms_out_num'])]

    return Ldir


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == '__main__':
    tt0 = timer()
    Ldir = get_args()

    label = Ldir['label']
    ds0 = Ldir['ds0']
    ds1 = Ldir['ds1']

    # ----- Load zeta time series -----
    zeta_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
                / ('zeta_ts_' + ds0 + '_' + ds1))
    zeta_fn = zeta_dir / (label + '.nc')
    if not zeta_fn.is_file():
        print(f'ERROR: zeta file not found: {zeta_fn}')
        print('Run extract_zeta_ts.py first.')
        sys.exit(1)

    ds_in = xr.open_dataset(zeta_fn)
    zeta = ds_in['zeta'].values
    time_zeta = ds_in['time'].values

    # Load qnet if available (from bulk output, attached by extract_zeta_ts)
    has_qnet = 'qnet' in ds_in
    has_qprism = 'qprism' in ds_in
    if has_qnet:
        qnet = ds_in['qnet'].values
        qnet_time = ds_in['qnet_time'].values
    if has_qprism:
        qprism_bulk = ds_in['qprism'].values
    ds_in.close()

    print(f'Loaded zeta: {len(zeta)} timesteps, {time_zeta[0]} to {time_zeta[-1]}')
    if has_qnet:
        print(f'Loaded qnet: {len(qnet)} timesteps (daily, from bulk)')
    if has_qprism:
        print(f'Loaded qprism: {len(qprism_bulk)} timesteps (daily, from bulk)')

    # ----- Signal-based detection -----
    results = {}

    if Ldir['method'] in ('signal', 'both'):
        print('\n--- Signal-based detection ---')

        # Flood/ebb
        if Ldir['fe_method'] == 'qnet' and has_qnet:
            # qnet is on daily bulk time — need hourly for alignment
            # Use zeta derivative instead if qnet is daily
            print('  Note: qnet from bulk is daily-averaged. '
                  'Using zeta for hourly flood/ebb detection.')
            is_flood, is_ebb = tpf.detect_flood_ebb(zeta, time_zeta,
                                                     method='zeta')
        elif Ldir['fe_method'] == 'zeta':
            is_flood, is_ebb = tpf.detect_flood_ebb(zeta, time_zeta,
                                                     method='zeta')
        else:
            # Fallback to zeta if qnet not available
            print('  qnet not available; falling back to zeta for flood/ebb.')
            is_flood, is_ebb = tpf.detect_flood_ebb(zeta, time_zeta,
                                                     method='zeta')

        # Slack
        slack_hi, slack_lo = tpf.detect_slack(zeta, time_zeta, method='zeta')

        # Spring/neap — needs qprism
        if has_qprism:
            # qprism is on daily time; interpolate to hourly zeta time
            import pandas as pd
            qp_series = pd.Series(qprism_bulk,
                                  index=pd.DatetimeIndex(ds_in['qnet_time'].values
                                                         if has_qnet
                                                         else time_zeta[:len(qprism_bulk)]))
            # Reopen to get qnet_time
            ds_tmp = xr.open_dataset(zeta_fn)
            if 'qnet_time' in ds_tmp:
                qp_time = ds_tmp['qnet_time'].values
            else:
                qp_time = time_zeta[:len(qprism_bulk)]
            ds_tmp.close()

            qp_series = pd.Series(qprism_bulk, index=pd.DatetimeIndex(qp_time))
            qp_hourly = qp_series.reindex(
                pd.DatetimeIndex(time_zeta)).interpolate(method='time')
            is_spring, is_neap = tpf.detect_spring_neap(
                qp_hourly.values,
                percentile_thresh=Ldir['sn_percentile'])
        else:
            # Compute qprism from zeta-based proxy (tidal range envelope)
            print('  No qprism available; using tidal range envelope for spring/neap.')
            is_spring, is_neap = tpf._spring_neap_from_envelope(zeta)

        results['signal'] = {
            'is_flood': is_flood,
            'is_ebb': is_ebb,
            'is_spring': is_spring,
            'is_neap': is_neap,
            'slack_hi': slack_hi,
            'slack_lo': slack_lo,
        }
        n_flood = is_flood.sum()
        n_ebb = is_ebb.sum()
        n_spring = is_spring.sum()
        n_neap = is_neap.sum()
        print(f'  flood: {n_flood}, ebb: {n_ebb}, '
              f'spring: {n_spring}, neap: {n_neap}, '
              f'slack_hi: {slack_hi.sum()}, slack_lo: {slack_lo.sum()}')

    # ----- UTide harmonic detection -----
    if Ldir['method'] in ('utide', 'both'):
        print('\n--- UTide harmonic detection ---')
        phase_dict = tpf.detect_phases_utide(zeta, time_zeta, Ldir['lat'])
        results['utide'] = phase_dict
        print(f'  flood: {phase_dict["is_flood"].sum()}, '
              f'ebb: {phase_dict["is_ebb"].sum()}, '
              f'spring: {phase_dict["is_spring"].sum()}, '
              f'neap: {phase_dict["is_neap"].sum()}')

    # ----- Choose primary result -----
    if Ldir['method'] == 'signal':
        primary = results['signal']
    elif Ldir['method'] == 'utide':
        primary = results['utide']
    else:
        # 'both': use UTide for flood/ebb/slack, signal for spring/neap
        primary = {
            'is_flood': results['utide']['is_flood'],
            'is_ebb': results['utide']['is_ebb'],
            'slack_hi': results['utide']['slack_hi'],
            'slack_lo': results['utide']['slack_lo'],
            'is_spring': results['signal']['is_spring'],
            'is_neap': results['signal']['is_neap'],
        }

    # ----- Build phase labels -----
    labels_df = tpf.get_phase_labels(
        time_zeta,
        primary['is_flood'],
        primary['is_spring'],
        primary['slack_hi'],
        primary['slack_lo'],
    )

    # Report phase counts
    print('\n--- Phase label counts ---')
    print(labels_df['phase'].value_counts().to_string())

    # ----- Save to NetCDF -----
    out_dir = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('tide_phases_' + ds0 + '_' + ds1))
    Lfun.make_dir(out_dir)
    out_fn = out_dir / (label + '.nc')

    ds_out = xr.Dataset(
        coords={'time': time_zeta},
        attrs={
            'gridname': Ldir['gridname'],
            'gtagex': Ldir['gtagex'],
            'ds0': ds0,
            'ds1': ds1,
            'label': label,
            'method': Ldir['method'],
            'fe_method': Ldir['fe_method'],
            'sn_percentile': Ldir['sn_percentile'],
        },
    )

    ds_out['zeta'] = ('time', zeta)
    ds_out['zeta'].attrs['units'] = 'm'

    for vn in ['is_flood', 'is_ebb', 'is_spring', 'is_neap',
               'slack_hi', 'slack_lo']:
        ds_out[vn] = ('time', primary[vn].astype(np.int8))

    # Encode composite phase as integer for NetCDF compatibility
    phase_map = {
        'spring_flood': 1, 'spring_ebb': 2,
        'neap_flood': 3, 'neap_ebb': 4,
        'slack_high': 5, 'slack_low': 6,
        'unclassified': 0,
    }
    phase_int = np.array([phase_map.get(p, 0) for p in labels_df['phase'].values],
                         dtype=np.int8)
    ds_out['phase'] = ('time', phase_int)
    ds_out['phase'].attrs['flag_values'] = list(phase_map.values())
    ds_out['phase'].attrs['flag_meanings'] = ' '.join(phase_map.keys())

    # Include UTide prediction if available
    if 'utide' in results:
        ds_out['zeta_pred'] = ('time', results['utide']['pred'])
        ds_out['zeta_pred'].attrs['long_name'] = 'UTide tidal prediction'

    ds_out.to_netcdf(out_fn)
    ds_out.close()

    print(f'\nSaved: {out_fn}')
    print(f'Total time: {timer() - tt0:.1f} s')
