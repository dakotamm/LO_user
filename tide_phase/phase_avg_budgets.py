"""
Compute TEF transports and tracer budgets grouped by tidal phase.

Reads tide phase labels from compute_tide_phases.py output, then loads
hourly TEF section data (from extract_sections_avg.py or extract_sections.py)
and groups transports by phase.

Usage examples
--------------
    # From avg-pipeline extractions:
    python phase_avg_budgets.py -gtx wb1_r0_xn11b -ctag pc0 \
        -sect_name pc0 -0 2017.09.01 -1 2017.09.30 -file_type avg

    # From his-pipeline extractions:
    python phase_avg_budgets.py -gtx wb1_r0_xn11b -ctag pc0 \
        -sect_name pc0 -0 2017.09.01 -1 2017.09.30 -file_type his
"""

import argparse
import sys
import numpy as np
import xarray as xr
import pandas as pd
from time import time as timer

from lo_tools import Lfun

import tide_phase_fun as tpf


# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(
        description='Compute phase-grouped TEF transports and budgets.')
    parser.add_argument('-gtx', '--gtagex', type=str, required=True)
    parser.add_argument('-ro', '--roms_out_num', type=int, default=0)
    parser.add_argument('-0', '--ds0', type=str, required=True)
    parser.add_argument('-1', '--ds1', type=str, required=True)
    parser.add_argument('-ctag', '--collection_tag', type=str, default=None)
    parser.add_argument('-sect_name', type=str, required=True,
                        help='Section name for both phase labels and TEF data')
    parser.add_argument('-file_type', type=str, default='avg',
                        choices=['avg', 'his'],
                        help='Which extraction pipeline was used: avg or his')
    parser.add_argument('-phases', type=str,
                        default='spring_flood,spring_ebb,neap_flood,neap_ebb',
                        help='Comma-separated phase names')
    parser.add_argument('-test', '--testing', type=Lfun.boolean_string,
                        default=False)

    args = parser.parse_args()

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

    sect_name = Ldir['sect_name']
    ds0 = Ldir['ds0']
    ds1 = Ldir['ds1']
    phase_names = [s.strip() for s in Ldir['phases'].split(',')]

    tp_base = Ldir['LOo'] / 'tide_phase' / Ldir['gtagex']
    tef2_base = Ldir['LOo'] / 'extract' / Ldir['gtagex'] / 'tef2'

    # ----- Load phase labels -----
    phase_fn = (tp_base / ('tide_phases_' + ds0 + '_' + ds1)
                / (sect_name + '.nc'))
    if not phase_fn.is_file():
        print(f'ERROR: phase file not found: {phase_fn}')
        sys.exit(1)

    ds_phase = xr.open_dataset(phase_fn)
    phase_time = ds_phase['time'].values
    phase_int = ds_phase['phase'].values
    flag_meanings = ds_phase['phase'].attrs.get('flag_meanings', '').split()
    flag_values = ds_phase['phase'].attrs.get('flag_values', [])
    val_to_name = dict(zip(flag_values, flag_meanings))
    phase_str = np.array([val_to_name.get(int(v), 'unclassified')
                          for v in phase_int])
    ds_phase.close()

    # ----- Load hourly TEF extraction -----
    # Try avg pipeline first, then his pipeline
    if Ldir['file_type'] == 'avg':
        ext_dir = tef2_base / ('extractions_avg_' + ds0 + '_' + ds1)
    else:
        ext_dir = tef2_base / ('extractions_' + ds0 + '_' + ds1)

    ext_fn = ext_dir / (sect_name + '.nc')
    if not ext_fn.is_file():
        print(f'ERROR: extraction file not found: {ext_fn}')
        print(f'Run extract_sections{"_avg" if Ldir["file_type"]=="avg" else ""}.py first.')
        sys.exit(1)

    ds_ext = xr.open_dataset(ext_fn)
    ext_time = ds_ext['time'].values
    print(f'Extraction: {len(ext_time)} timesteps, {ext_time[0]} to {ext_time[-1]}')

    # Identify available variables
    # q = volume transport, salt, temp, oxygen, etc.
    vn_3d = [vn for vn in ds_ext.data_vars
             if ds_ext[vn].dims == ('time', 'z', 'p')]
    vn_2d_tp = [vn for vn in ds_ext.data_vars
                if ds_ext[vn].dims == ('time', 'p')]
    print(f'3D vars (time,z,p): {vn_3d}')
    print(f'2D vars (time,p): {vn_2d_tp}')

    # ----- Map extraction times to phase labels -----
    # Find nearest phase label for each extraction timestep
    ext_phases = np.full(len(ext_time), 'unclassified', dtype=object)
    for ii, et in enumerate(ext_time):
        idx = np.argmin(np.abs(phase_time - et))
        dt_diff = abs((phase_time[idx] - et) / np.timedelta64(1, 'h'))
        if dt_diff <= 1.5:
            ext_phases[ii] = phase_str[idx]

    # ----- Compute section-integrated transport per timestep -----
    # q is volume flux [m3/s] at each (z, p) point
    if 'q' in ds_ext:
        q = ds_ext['q'].values  # (time, z, p)
    else:
        print('WARNING: no "q" variable in extraction; skipping transport.')
        q = None

    # Get DZ for flux-weighted tracer calculations
    if 'DZ' in ds_ext:
        DZ = ds_ext['DZ'].values  # (time, z, p)
    if 'dd' in ds_ext:
        dd = ds_ext['dd'].values  # (p,) — section width

    # Compute qnet (section-integrated volume transport) per timestep
    if q is not None:
        qnet_hourly = np.nansum(q, axis=(1, 2))  # (time,)

    # Compute section-integrated tracer transport per timestep
    tracer_transport = {}
    for vn in vn_3d:
        if vn in ('q', 'DZ'):
            continue
        fld = ds_ext[vn].values  # (time, z, p)
        if q is not None:
            tracer_transport[vn] = np.nansum(q * fld, axis=(1, 2))  # (time,)

    ds_ext.close()

    # ----- Group by phase and compute statistics -----
    results = {}
    for pn in phase_names:
        mask = ext_phases == pn
        n = mask.sum()
        results[pn] = {'n_timesteps': int(n)}

        if n == 0:
            print(f'  {pn}: no timesteps')
            continue

        print(f'  {pn}: {n} timesteps')

        if q is not None:
            qnet_phase = qnet_hourly[mask]
            results[pn]['qnet_mean'] = float(np.nanmean(qnet_phase))
            results[pn]['qnet_std'] = float(np.nanstd(qnet_phase))
            # Separate into Qin (positive) and Qout (negative)
            results[pn]['Qin_mean'] = float(np.nanmean(
                np.where(qnet_phase > 0, qnet_phase, 0)))
            results[pn]['Qout_mean'] = float(np.nanmean(
                np.where(qnet_phase < 0, qnet_phase, 0)))

        for vn, transport in tracer_transport.items():
            transport_phase = transport[mask]
            results[pn][f'{vn}_flux_mean'] = float(np.nanmean(transport_phase))
            results[pn][f'{vn}_flux_std'] = float(np.nanstd(transport_phase))

    # ----- Save to NetCDF -----
    out_dir = tp_base / ('phase_budget_' + ds0 + '_' + ds1)
    Lfun.make_dir(out_dir)
    out_fn = out_dir / (sect_name + '.nc')

    # Build output dataset
    ds_out = xr.Dataset(
        coords={'phase': phase_names},
        attrs={
            'gridname': Ldir['gridname'],
            'gtagex': Ldir['gtagex'],
            'ds0': ds0,
            'ds1': ds1,
            'sect_name': sect_name,
            'file_type': Ldir['file_type'],
        },
    )

    # Pack scalar stats per phase
    n_arr = np.array([results[pn]['n_timesteps'] for pn in phase_names])
    ds_out['n_timesteps'] = ('phase', n_arr)

    for stat_key in ['qnet_mean', 'qnet_std', 'Qin_mean', 'Qout_mean']:
        arr = np.array([results[pn].get(stat_key, np.nan)
                        for pn in phase_names])
        ds_out[stat_key] = ('phase', arr)
        if 'qnet' in stat_key or 'Q' in stat_key:
            ds_out[stat_key].attrs['units'] = 'm3/s'

    for vn in tracer_transport:
        for suffix in ['_flux_mean', '_flux_std']:
            key = f'{vn}{suffix}'
            arr = np.array([results[pn].get(key, np.nan)
                            for pn in phase_names])
            ds_out[key] = ('phase', arr)

    ds_out.to_netcdf(out_fn)
    ds_out.close()

    print(f'\nSaved: {out_fn}')
    print(f'Total time: {timer() - tt0:.1f} s')
