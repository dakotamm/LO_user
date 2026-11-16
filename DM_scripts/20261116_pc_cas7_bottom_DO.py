"""
Bottom DO time series at one Penn Cove point in cas7_t1_x11ab (Parker's run on apogee /dat1), from the daily
averages f<YYYY.MM.DD>/ocean_avg_0001.nc. Lightweight: one value per file, oxygen[0, 0, j, i] (bottom layer).

Run on apogee (-ro 1 = Ldir['roms_out1'] = /dat1/parker/LO_roms on dm_apogee):
    python 20261116_pc_cas7_bottom_DO.py -gtx cas7_t1_x11ab -ro 1 -0 2012.10.07 -1 2025.06.12 > pc_cas7_bottom_DO.log &
Re-plot from the saved csv (e.g. on the Mac after copying it over):
    python 20261116_pc_cas7_bottom_DO.py -gtx cas7_t1_x11ab -0 2012.10.07 -1 2025.06.12 -plot_only True

POINT   western Penn Cove: the centre of cas7 rho cell j=882, i=557 (h = 17.0 m), one cell south and one west of the
        mid-channel cell j=883, i=558. Nearest wb1 cell j=217, i=47 (h = 17.2 m): bottom DO < 2 mg/L on 19% of 2024-25
        days in wb1_t0_xn11abbur00 (20260806_hypoxia RUN['fhyp_bot_2']). Snapped to the nearest wet cas7 rho cell
        (mask_rho from the first avg file, grid.nc if the avg file has no grid variables); the cell is printed.
        Map: 20261116_pc_cas7_point_map.py.
BOTTOM  s_rho index 0. oxygen is mmol m-3; mg/L = mmol m-3 * 32/1000.
MISSING days with no ocean_avg_0001.nc are skipped and counted.
OUT     LO_output/DM_outs/20261116_pc_cas7_bottom_DO/  (csv of daily bottom DO + png)
"""

import argparse
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun

parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str, default='cas7_t1_x11ab')
parser.add_argument('-ro', '--roms_out_num', type=int, default=1)  # 1 = Ldir['roms_out1']
parser.add_argument('-0', '--ds0', type=str, default='2012.10.07')
parser.add_argument('-1', '--ds1', type=str, default='2025.06.12')
parser.add_argument('-lon', type=float, default=-122.70878)
parser.add_argument('-lat', type=float, default=48.22710)
parser.add_argument('-sn', type=str, default='pc_west')  # station name, used in the output file names
parser.add_argument('-plot_only', type=zfun.boolean_string, default=False)
args = parser.parse_args()

gridname, tag, ex_name = args.gtagex.split('_')
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
if args.roms_out_num > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]

DO_MMOL_TO_MGL = 32.0 / 1000.0

out_dir = Ldir['LOo'] / 'DM_outs' / '20261116_pc_cas7_bottom_DO'
Lfun.make_dir(out_dir)
out_name = args.gtagex + '_' + args.sn + '_' + args.ds0 + '_' + args.ds1
csv_fn = out_dir / (out_name + '.csv')

if not args.plot_only:
    fn_list = Lfun.get_fn_list('average', Ldir, args.ds0, args.ds1)
    fn_have = [fn for fn in fn_list if fn.exists()]
    print(f'files  : {len(fn_have)} of {len(fn_list)} days have ocean_avg_0001.nc')
    if len(fn_have) == 0:
        print(f'*** no files under {Ldir["roms_out"] / args.gtagex}')
        sys.exit()

    # ---- nearest wet rho cell ----
    with nc.Dataset(fn_have[0]) as ds:
        if 'mask_rho' in ds.variables:
            G = {vn: ds[vn][:].data for vn in ['lon_rho', 'lat_rho', 'mask_rho', 'h']}
        else:
            with nc.Dataset(Ldir['grid'] / 'grid.nc') as dg:
                G = {vn: dg[vn][:].data for vn in ['lon_rho', 'lat_rho', 'mask_rho', 'h']}
    dist2 = ((G['lon_rho'] - args.lon) * np.cos(np.deg2rad(args.lat)))**2 + (G['lat_rho'] - args.lat)**2
    dist2[G['mask_rho'] != 1] = np.inf
    j, i = np.unravel_index(np.argmin(dist2), dist2.shape)
    lon, lat, h = G['lon_rho'][j, i], G['lat_rho'][j, i], G['h'][j, i]
    print(f'cell   : j={j} i={i}  lon {lon:.5f}  lat {lat:.5f}  h {h:.1f} m  '
          f'({np.sqrt(dist2[j, i]) * 111:.2f} km from the requested point)')

    # ---- one value per file ----
    tt, oo = [], []
    for nn, fn in enumerate(fn_have):
        with nc.Dataset(fn) as ds:
            tt.append(nc.num2date(ds['ocean_time'][0], ds['ocean_time'].units,
                                  only_use_cftime_datetimes=False, only_use_python_datetimes=True))
            oo.append(float(ds['oxygen'][0, 0, j, i]))
        if nn % 500 == 0:
            print(f'  {nn} / {len(fn_have)}  {fn.parent.name}', flush=True)

    df = pd.DataFrame({'time_utc': pd.to_datetime(tt), 'DO_bot_mgL': np.array(oo) * DO_MMOL_TO_MGL})
    df.attrs = {}
    with open(csv_fn, 'w') as f:
        f.write(f'# {args.gtagex} ocean_avg_0001 bottom DO, cell j={j} i={i} lon={lon:.5f} lat={lat:.5f} h={h:.1f}\n')
        df.to_csv(f, index=False)
    print(f'saved  : {csv_fn}')

# ---- plot ----
with open(csv_fn) as f:
    header = f.readline().strip('# \n')
df = pd.read_csv(csv_fn, comment='#', parse_dates=['time_utc'])
t, do = df['time_utc'], df['DO_bot_mgL']
print(f'bottom DO: min {do.min():.2f}  mean {do.mean():.2f}  max {do.max():.2f} mg/L;  '
      f'days < 2 mg/L: {int((do < 2).sum())} of {len(do)}')

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t, do, '-', color='tab:blue', lw=0.8)
ax.axhline(2, color='gray', ls=':', lw=1)
ax.set_ylabel('Bottom DO [mg/L]')
ax.set_ylim(bottom=0)
ax.set_xlim(t.iloc[0], t.iloc[-1])
ax.grid(True, alpha=0.3)
ax.set_title(header, fontsize=10)
fig.tight_layout()
fig_fn = out_dir / (out_name + '.png')
fig.savefig(fig_fn, dpi=200, transparent=True)
plt.close(fig)
print(f'saved  : {fig_fn}')
