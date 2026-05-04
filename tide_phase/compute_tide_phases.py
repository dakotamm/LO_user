"""
Compute tidal phase labels from a zeta time series via UTide.

Reads output from extract_zeta_ts.py, runs harmonic analysis with UTide,
classifies flood/ebb from d(prediction)/dt and spring/neap from the
M2+S2 beat envelope.

Usage
-----
    python compute_tide_phases.py -gtx wb1_t0_xn11ab \
        -0 2024.01.01 -1 2024.06.30 -label penn_cove
"""

import argparse
import sys
import numpy as np
import xarray as xr
from time import time as timer

from lo_tools import Lfun

import tide_phase_fun as tpf


# -----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='Compute UTide-based tidal phase labels.')
    parser.add_argument('-gtx', '--gtagex', type=str, required=True)
    parser.add_argument('-0', '--ds0', type=str, required=True)
    parser.add_argument('-1', '--ds1', type=str, required=True)
    parser.add_argument('-label', type=str, required=True,
                        help='Label matching extract_zeta_ts output filename')
    parser.add_argument('-sn_percentile', type=float, default=50,
                        help='Percentile threshold for spring/neap split')

    args = parser.parse_args()

    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
    for k, v in vars(args).items():
        if k not in Ldir:
            Ldir[k] = v
    return Ldir


# -----------------------------------------------------------------------
if __name__ == '__main__':
    tt0 = timer()
    Ldir = get_args()
    label = Ldir['label']
    ds0 = Ldir['ds0']
    ds1 = Ldir['ds1']

    # Load zeta
    zeta_fn = (Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
               / ('zeta_ts_' + ds0 + '_' + ds1) / (label + '.nc'))
    if not zeta_fn.is_file():
        print(f'ERROR: zeta file not found: {zeta_fn}')
        print('Run extract_zeta_ts.py first.')
        sys.exit(1)

    ds_in = xr.open_dataset(zeta_fn)
    zeta = ds_in['zeta'].values
    time_zeta = ds_in['time'].values
    lat = ds_in.attrs.get('lat_actual', ds_in.attrs.get('lat_target'))
    lon = ds_in.attrs.get('lon_actual', ds_in.attrs.get('lon_target'))
    ds_in.close()

    print(f'Loaded zeta: {len(zeta)} timesteps, {time_zeta[0]} to {time_zeta[-1]}')
    print(f'Location: ({lon:.4f}, {lat:.4f})')

    # UTide harmonic analysis
    print('\n--- UTide harmonic analysis ---')
    phase_dict = tpf.detect_phases_utide(zeta, time_zeta, lat)

    # Optionally re-do spring/neap with custom percentile
    if Ldir['sn_percentile'] != 50:
        print(f'Recomputing spring/neap at percentile={Ldir["sn_percentile"]}')
        is_spring, is_neap, _ = tpf.spring_neap_from_m2s2(
            time_zeta, phase_dict['coef'],
            percentile_thresh=Ldir['sn_percentile'])
        phase_dict['is_spring'] = is_spring
        phase_dict['is_neap'] = is_neap

    # Build composite phase labels
    labels_df = tpf.get_phase_labels(
        time_zeta,
        phase_dict['is_flood'],
        phase_dict['is_spring'],
        phase_dict['slack_hi'],
        phase_dict['slack_lo'],
    )

    print('\n--- Phase counts ---')
    print(f'  flood:  {phase_dict["is_flood"].sum()}')
    print(f'  ebb:    {phase_dict["is_ebb"].sum()}')
    print(f'  spring: {phase_dict["is_spring"].sum()}')
    print(f'  neap:   {phase_dict["is_neap"].sum()}')
    print(f'  slack_hi: {phase_dict["slack_hi"].sum()}, '
          f'slack_lo: {phase_dict["slack_lo"].sum()}')
    print('\n--- Composite phase counts ---')
    print(labels_df['phase'].value_counts().to_string())

    # Save
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
            'method': 'utide_m2s2',
            'sn_percentile': Ldir['sn_percentile'],
            'lat': lat,
            'lon': lon,
        },
    )
    ds_out['zeta'] = ('time', zeta)
    ds_out['zeta'].attrs['units'] = 'm'
    ds_out['zeta_pred'] = ('time', phase_dict['pred'])
    ds_out['zeta_pred'].attrs['long_name'] = 'UTide tidal prediction (all constituents)'
    ds_out['zeta_m2s2'] = ('time', phase_dict['pred_m2s2'])
    ds_out['zeta_m2s2'].attrs['long_name'] = 'UTide M2+S2 reconstruction'

    for vn in ['is_flood', 'is_ebb', 'is_spring', 'is_neap',
               'slack_hi', 'slack_lo']:
        ds_out[vn] = ('time', phase_dict[vn].astype(np.int8))

    phase_map = {
        'unclassified': 0,
        'spring_flood': 1, 'spring_ebb': 2,
        'neap_flood': 3, 'neap_ebb': 4,
        'slack_high': 5, 'slack_low': 6,
    }
    phase_int = np.array([phase_map.get(p, 0) for p in labels_df['phase'].values],
                         dtype=np.int8)
    ds_out['phase'] = ('time', phase_int)
    ds_out['phase'].attrs['flag_values'] = list(phase_map.values())
    ds_out['phase'].attrs['flag_meanings'] = ' '.join(phase_map.keys())

    ds_out.to_netcdf(out_fn)
    ds_out.close()

    print(f'\nSaved: {out_fn}')
    print(f'Total time: {timer() - tt0:.1f} s')
