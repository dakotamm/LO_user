"""
Plot map of model hypoxic days over a year, with obs hypoxic cast counts overlaid.

A grid cell is "hypoxic on day D" if model DO < 2 mg/L (62.5 uM) anywhere in
the water column (any z level) on that day.
A cast is "hypoxic" if obs DO < 2 mg/L at any sampled depth.

Reads daily ocean_avg_0001.nc files for the year, plus the combined pickle(s)
for obs hypoxic counts per station.

Testing on mac:
run plot_hypoxic_days -gtx wb1_t0_xn11ab -year 2024 -ro 2 -test True

"""
import sys
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str)
parser.add_argument('-year', type=int)
parser.add_argument('-ro', '--roms_out_num', type=int, default=0)
parser.add_argument('-otype', type=str, default='all') # bottle, ctd, all
parser.add_argument('-hypoxic_threshold_mgL', type=float, default=2.0)
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

Ldir = Lfun.Lstart()
if args.roms_out_num > 0:
    Ldir['roms_out'] = Ldir['roms_out' + str(args.roms_out_num)]

if '_mac' in Ldir['lo_env']:
    pass
else:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

gtx = args.gtagex
year = args.year
year_str = str(year)

# threshold in uM (model + obs are stored in uM in the pickle / model output)
DO_UM_TO_MGL = 32.0 / 1000.0
hyp_uM = args.hypoxic_threshold_mgL / DO_UM_TO_MGL  # 2 mg/L = 62.5 uM

out_dir = Ldir['LOo'] / 'obsmod_val_plots'
Lfun.make_dir(out_dir)

# ===== Build list of daily avg files =====
roms_dir = Ldir['roms_out'] / gtx
start = datetime(year, 1, 1)
end = datetime(year, 12, 31)
day_list = []
fn_list = []
d = start
while d <= end:
    date_string = d.strftime('%Y.%m.%d')
    fn = roms_dir / ('f' + date_string) / 'ocean_avg_0001.nc'
    if fn.is_file():
        day_list.append(d)
        fn_list.append(fn)
    d += timedelta(days=1)

if len(fn_list) == 0:
    print('No model files found in %s' % roms_dir)
    sys.exit()

print('Found %d daily avg files for %d' % (len(fn_list), year))

# ===== Compute hypoxic-day count at each grid cell =====
# Open first file to get grid shape
ds0 = xr.open_dataset(fn_list[0])
lon_rho = ds0.lon_rho.values
lat_rho = ds0.lat_rho.values
mask_rho = ds0.mask_rho.values
ny, nx = lon_rho.shape
ds0.close()

hyp_count = np.zeros((ny, nx), dtype=int)

for i, fn in enumerate(fn_list):
    if (i % 30) == 0:
        print('  processing %s (%d/%d)' % (fn.parent.name, i+1, len(fn_list)))
        sys.stdout.flush()
    ds = xr.open_dataset(fn)
    # oxygen has dims (ocean_time, s_rho, eta_rho, xi_rho), single time step
    oxy = ds.oxygen.values  # shape (1, N, ny, nx)
    # min along z axis, drop time dim
    oxy_min = np.nanmin(oxy[0], axis=0)  # (ny, nx)
    hyp_count += (oxy_min < hyp_uM).astype(int)
    ds.close()

# mask land
hyp_count_masked = np.where(mask_rho > 0, hyp_count, np.nan)

print('Max hypoxic days at any cell: %d / %d' % (int(np.nanmax(hyp_count_masked)), len(fn_list)))

# ===== Obs: count hypoxic casts per station =====
otype_list = ['bottle', 'ctd'] if args.otype == 'all' else [args.otype]

obs_records = []
for otype in otype_list:
    in_fn = Ldir['LOo'] / 'obsmod' / ('combined_' + otype + '_' + year_str + '_' + gtx + '.p')
    if not in_fn.is_file():
        print('Missing combined pickle: %s' % in_fn)
        continue
    df_dict = pickle.load(open(in_fn, 'rb'))
    obs = df_dict['obs']
    if 'DO (uM)' not in obs.columns:
        continue
    sub = obs[['cid', 'lon', 'lat', 'name', 'DO (uM)']].copy()
    sub['otype'] = otype
    obs_records.append(sub)

obs_stn = None
if len(obs_records) > 0:
    all_obs = pd.concat(obs_records, ignore_index=True)
    all_obs = all_obs.dropna(subset=['DO (uM)'])
    # per-cast hypoxic flag = any depth below threshold
    cast_hyp = all_obs.groupby('cid').agg(
        hyp=('DO (uM)', lambda x: int(np.any(x < hyp_uM))),
        lon=('lon', 'mean'),
        lat=('lat', 'mean'),
        name=('name', 'first'),
    ).reset_index()
    # group by station (use name if available, otherwise rounded coords)
    cast_hyp['lon_r'] = cast_hyp['lon'].round(3)
    cast_hyp['lat_r'] = cast_hyp['lat'].round(3)
    if cast_hyp['name'].astype(str).str.len().sum() > 0:
        cast_hyp['station'] = cast_hyp['name']
    else:
        cast_hyp['station'] = cast_hyp['lon_r'].astype(str) + '_' + cast_hyp['lat_r'].astype(str)
    obs_stn = cast_hyp.groupby('station').agg(
        n_hyp=('hyp', 'sum'),
        n_casts=('hyp', 'count'),
        lon=('lon', 'mean'),
        lat=('lat', 'mean'),
    ).reset_index()
    print('%d unique obs stations; %d had at least one hypoxic cast' %
          (len(obs_stn), int((obs_stn['n_hyp'] > 0).sum())))

# ===== Auto-zoom extent =====
# default to model grid where mask is valid; if obs is present, include those too
water_lon = lon_rho[mask_rho > 0]
water_lat = lat_rho[mask_rho > 0]
lon_min, lon_max = water_lon.min(), water_lon.max()
lat_min, lat_max = water_lat.min(), water_lat.max()
pad_lon = 0.05 * (lon_max - lon_min)
pad_lat = 0.05 * (lat_max - lat_min)
extent = [lon_min - pad_lon, lon_max + pad_lon,
          lat_min - pad_lat, lat_max + pad_lat]

# ===== Plot =====
pfun.start_plot(figsize=(11, 9), fs=12)
fig, ax = plt.subplots(1, 1, figsize=(11, 9))

vmax = max(int(np.nanmax(hyp_count_masked)), 1)
pcm = ax.pcolormesh(lon_rho, lat_rho, hyp_count_masked,
                    cmap='magma_r', vmin=0, vmax=vmax, shading='auto')
cbar = plt.colorbar(pcm, ax=ax, shrink=0.8)
cbar.set_label('Model hypoxic days (DO<%.1f mg/L anywhere in column)' %
               args.hypoxic_threshold_mgL)

# Overlay obs stations: dot size proportional to n_hyp, color by n_hyp
if obs_stn is not None and len(obs_stn) > 0:
    no_hyp = obs_stn[obs_stn['n_hyp'] == 0]
    has_hyp = obs_stn[obs_stn['n_hyp'] > 0]
    if len(no_hyp) > 0:
        ax.scatter(no_hyp['lon'], no_hyp['lat'],
                   marker='o', s=25, facecolors='none', edgecolors='cyan',
                   linewidths=1.0, zorder=5,
                   label='station, no hypoxic casts')
    if len(has_hyp) > 0:
        sc = ax.scatter(has_hyp['lon'], has_hyp['lat'],
                        c=has_hyp['n_hyp'], cmap='YlGn',
                        s=40 + 20*has_hyp['n_hyp'], edgecolors='k',
                        linewidths=0.6, zorder=6,
                        label='station with hypoxic casts')
        cbar2 = plt.colorbar(sc, ax=ax, shrink=0.5, location='left',
                             pad=0.08)
        cbar2.set_label('Obs hypoxic casts at station')

pfun.add_coast(ax)
ax.axis(extent)
pfun.dar(ax)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Hypoxic Days  %s %d  (threshold %.1f mg/L)\n'
             'model n=%d days   obs combined %s' %
             (gtx, year, args.hypoxic_threshold_mgL, len(fn_list),
              ','.join(otype_list)),
             fontweight='bold')
ax.legend(loc='lower right', fontsize=8)

fig.tight_layout()

ff_str = 'hypoxic_days_%d_%s' % (year, gtx)
print('Plotting ' + ff_str)
sys.stdout.flush()

if args.testing:
    plt.show()
else:
    plt.savefig(out_dir / (ff_str + '.png'), bbox_inches='tight')
    print('Saved to:\n %s' % (str(out_dir / (ff_str + '.png'))))
