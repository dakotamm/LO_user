"""
Phase 1 driver: build a per-mooring wind / flow / stratification / DO
NetCDF for downstream plotting and stats.

For each hourly time step in the input mooring file, computes:
  - 10-m wind components, speed, met direction, bulk wind stress (tau_x/y/mag)
  - Surface, bottom, and depth-averaged (u, v); plus along/across channel
    rotations (one rotation angle per layer, derived from the layer's own
    data-driven principal axis)
  - Top-bottom potential-density difference (gsw sigma0)
  - Bottom DO in mg/L
Each quantity is saved at hourly resolution plus a Godin (24-24-25)
tidal-averaged version (suffix '_lp').

Usage on apogee (after the hourly extraction is finished):
    conda activate loenv
    python compute_wind_flow_M1.py \
        -gtx wb1_r0_xn11b -mooring M1 -job pc0 \
        -ds0 2017.01.01 -ds1 2017.12.31

Hourly extraction prerequisite (run once on apogee, takes ~1 hr at Nproc=10):
    cd ~/LO/extract/moor
    python multi_mooring_driver.py -gtx wb1_r0_xn11b -ro 2 \
        -0 2017.01.01 -1 2017.12.31 -lt hourly -job pc0 -get_all True \
        > pc0_hourly.log &

Output:
    LOo/wind_flow/<gtx>/<mooring>_wind_flow_<ds0>_<ds1>.nc
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from lo_tools import Lfun

from wind_flow_utils import (
    load_moor, get_wind, get_flow_profile, principal_axis, rotate,
    strat_delta_rho, get_bottom_do_mgl, godin_lowpass_df,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-gtx', '--gtagex', type=str, default='wb1_r0_xn11b')
    p.add_argument('-mooring', type=str, default='M1')
    p.add_argument('-job', type=str, default='pc0')
    p.add_argument('-ds0', type=str, default='2017.01.01')
    p.add_argument('-ds1', type=str, default='2017.12.31')
    p.add_argument('-moor_fn', type=str, default=None,
                   help='Override path to the hourly mooring NetCDF.')
    p.add_argument('-out_fn', type=str, default=None,
                   help='Override output NetCDF path.')
    return p.parse_args()


def main():
    args = parse_args()
    gridname, tag, ex_name = args.gtagex.split('_')
    Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)

    if args.moor_fn is not None:
        moor_fn = Path(args.moor_fn)
    else:
        moor_fn = (Ldir['LOo'] / 'extract' / args.gtagex / 'moor' / args.job
                   / f'{args.mooring}_{args.ds0}_{args.ds1}.nc')
    if not moor_fn.exists():
        raise FileNotFoundError(f'Mooring file not found: {moor_fn}')
    print(f'Reading {moor_fn}')

    ds = load_moor(moor_fn)

    # --- check hourly cadence ---
    times = pd.to_datetime(ds.ocean_time.values)
    if len(times) > 1:
        dt_sec = (times[1] - times[0]).total_seconds()
        if abs(dt_sec - 3600.0) > 1.0:
            raise ValueError(
                f'Expected hourly cadence; got dt = {dt_sec:.1f} s. '
                f'Re-extract with -lt hourly.')

    # --- raw hourly diagnostics ---
    wind = get_wind(ds)
    flow = get_flow_profile(ds)
    strat = strat_delta_rho(ds)
    do = get_bottom_do_mgl(ds).to_frame()

    # --- principal-axis rotation per layer (data-driven) ---
    theta = {}
    for layer in ('surface', 'depthavg', 'bottom'):
        u = flow[f'u_{layer}'].values
        v = flow[f'v_{layer}'].values
        theta[layer] = principal_axis(u, v)
        along, across = rotate(u, v, theta[layer])
        flow[f'along_{layer}']  = along
        flow[f'across_{layer}'] = across

    # Wind rotated onto the depth-avg flow axis (single common reference)
    theta_w = theta['depthavg']
    tau_along, tau_across = rotate(wind.tau_x.values, wind.tau_y.values, theta_w)
    wind['tau_along']  = tau_along
    wind['tau_across'] = tau_across
    u_w_along, u_w_across = rotate(wind.Uwind.values, wind.Vwind.values, theta_w)
    wind['Uwind_along']  = u_w_along
    wind['Uwind_across'] = u_w_across

    raw = pd.concat([wind, flow, strat, do], axis=1)
    lp = godin_lowpass_df(raw)
    lp.columns = [c + '_lp' for c in lp.columns]
    full = pd.concat([raw, lp], axis=1)

    print(f'Principal-axis angles (deg CCW from east):')
    for layer, th in theta.items():
        print(f'  {layer:9s}: {np.degrees(th):+7.2f}')

    # --- save ---
    out = xr.Dataset(
        {c: (('ocean_time',), full[c].values) for c in full.columns},
        coords={'ocean_time': full.index.values},
    )
    out.attrs['gtagex']            = args.gtagex
    out.attrs['mooring']           = args.mooring
    out.attrs['job']               = args.job
    out.attrs['source_moor_fn']    = str(moor_fn)
    out.attrs['theta_surface_deg'] = np.degrees(theta['surface'])
    out.attrs['theta_depthavg_deg'] = np.degrees(theta['depthavg'])
    out.attrs['theta_bottom_deg']  = np.degrees(theta['bottom'])
    out.attrs['rotation_convention'] = (
        'theta is the principal-axis angle (CCW from east, radians) of the '
        'detrended (u, v) covariance for each layer. along = +cos*theta * u + '
        'sin*theta * v ; across = -sin*theta * u + cos*theta * v. Wind/stress '
        'are rotated onto the depth-avg theta.')
    out.attrs['lowpass'] = 'Godin 24-24-25 (suffix _lp); NaN-padded.'

    if args.out_fn is not None:
        out_fn = Path(args.out_fn)
    else:
        out_dir = Ldir['LOo'] / 'wind_flow' / args.gtagex
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fn = out_dir / (f'{args.mooring}_wind_flow_'
                            f'{args.ds0}_{args.ds1}.nc')
    out_fn.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {'zlib': True, 'complevel': 4} for v in out.data_vars}
    out.to_netcdf(out_fn, encoding=encoding)
    print(f'Wrote {out_fn}')


if __name__ == '__main__':
    main()
