"""
Plot time series of bottom DO at Penn Cove stations: observed vs model.

This reads the obs data directly (not the combined pickle) and extracts
the bottom DO from each cast. For the model, it reads the extracted cast
NetCDF files produced by extract_casts_fast.py.

Can be run standalone or called by one_step_val_plot.py.

Testing on mac:
run plot_penn_cove_DO -gtx wb1_t0_xn11ab -year0 2022 -year1 2024 -test True

"""
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xarray as xr
import gsw

from lo_tools import plotting_functions as pfun
from lo_tools import Lfun, zfun

# command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gtx', '--gtagex', type=str)
parser.add_argument('-year0', type=int)
parser.add_argument('-year1', type=int, default=0)
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

Ldir = Lfun.Lstart()

if '_mac' in Ldir['lo_env']:
    pass
else:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

gtx = args.gtagex
if args.year1 == 0:
    args.year1 = args.year0
year_list = list(range(args.year0, args.year1 + 1))

# Penn Cove station definitions
penn_cove_stations = ['PENNCOVECW', 'PENNCOVEENT', 'PENNCOVEWEST', 'PENNCOVEPNN001']

# DO unit conversion uM -> mg/L
DO_UM_TO_MGL = 32.0 / 1000.0

# source for Penn Cove data
source = 'kc_whidbeyBasin'
otype = 'ctd'

# Collect obs bottom DO and model bottom DO for each station
records = []  # list of dicts

for year in year_list:
    year_str = str(year)

    # load obs
    obs_fn = Ldir['LOo'] / 'obs' / source / otype / (year_str + '.p')
    info_fn = Ldir['LOo'] / 'obs' / source / otype / ('info_' + year_str + '.p')
    if not obs_fn.is_file():
        print('No obs for %s' % year_str)
        continue

    obs_df = pd.read_pickle(obs_fn)
    info_df = pd.read_pickle(info_fn)

    # model extraction directory
    mod_dir = Ldir['LOo'] / 'extract' / gtx / 'cast' / (source + '_' + otype + '_' + year_str)

    # filter to Penn Cove stations
    penn_obs = obs_df[obs_df['name'].str.contains('PENNCOVE', case=False, na=False)]

    for stn in penn_cove_stations:
        stn_obs = penn_obs[penn_obs['name'] == stn]
        if len(stn_obs) == 0:
            continue

        # get unique casts for this station
        cids = stn_obs['cid'].unique()

        for cid in cids:
            cast = stn_obs[stn_obs['cid'] == cid]

            # bottom obs value (most negative z)
            bot_idx = cast['z'].idxmin()
            bot_obs = cast.loc[bot_idx]
            obs_do_uM = bot_obs['DO (uM)'] if 'DO (uM)' in cast.columns else np.nan
            obs_do = obs_do_uM * DO_UM_TO_MGL
            obs_time = bot_obs['time']
            obs_z = bot_obs['z']

            # get model bottom DO from extracted cast
            mod_do = np.nan
            mod_z = np.nan
            cast_fn = mod_dir / (str(int(cid)) + '.nc')
            if cast_fn.is_file():
                ds = xr.open_dataset(cast_fn)
                if 'oxygen' in ds.data_vars:
                    mz = ds.z_rho.values
                    # find model level nearest to obs bottom depth
                    iz = zfun.find_nearest_ind(mz, obs_z)
                    mod_do = float(ds.oxygen[iz].values) * DO_UM_TO_MGL
                    mod_z = float(mz[iz])
                ds.close()

            records.append({
                'station': stn,
                'time': obs_time,
                'year': year,
                'obs_DO': obs_do,
                'mod_DO': mod_do,
                'obs_z': obs_z,
                'mod_z': mod_z,
            })

if len(records) == 0:
    print('No Penn Cove data found.')
    sys.exit()

df = pd.DataFrame(records)
df['time'] = pd.to_datetime(df['time'])

# Get unique stations that have data
stations = [s for s in penn_cove_stations if s in df['station'].values]
n_stn = len(stations)

# Color and marker setup
color_dict = {
    'PENNCOVECW': 'tab:blue',
    'PENNCOVEENT': 'tab:red',
    'PENNCOVEWEST': 'tab:green',
    'PENNCOVEPNN001': 'tab:orange',
}
label_dict = {
    'PENNCOVECW': 'Penn Cove CW',
    'PENNCOVEENT': 'Penn Cove Entrance',
    'PENNCOVEWEST': 'Penn Cove West',
    'PENNCOVEPNN001': 'Penn Cove PNN001',
}

# where to put output figures
out_dir = Ldir['LOo'] / 'obsmod_val_plots'
Lfun.make_dir(out_dir)

# ===== Figure: Time series per station =====
pfun.start_plot(figsize=(14, 4 * n_stn), fs=12)
fig, axes = plt.subplots(n_stn, 1, figsize=(14, 4 * n_stn), sharex=True)
if n_stn == 1:
    axes = [axes]

for i, stn in enumerate(stations):
    ax = axes[i]
    stn_df = df[df['station'] == stn].sort_values('time')

    ax.plot(stn_df['time'], stn_df['obs_DO'], 'o-',
            color=color_dict.get(stn, 'k'), markersize=5,
            label='Obs (bottom)')
    ax.plot(stn_df['time'], stn_df['mod_DO'], 's--',
            color=color_dict.get(stn, 'k'), markersize=5,
            alpha=0.7, markerfacecolor='none',
            label='Model (bottom)')

    ax.set_ylabel('DO (mg/L)')
    ax.set_title(label_dict.get(stn, stn), fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=2.0, color='gray', linestyle=':', linewidth=1, label='hypoxic (2 mg/L)')

    # add bottom depth info
    mean_z = stn_df['obs_z'].mean()
    ax.text(0.02, 0.05, 'mean bottom depth: %.0f m' % mean_z,
            transform=ax.transAxes, fontsize=9, color='gray')

axes[-1].set_xlabel('Date')
fig.suptitle('Penn Cove Bottom DO: %s\nobs vs %s' % (source, gtx),
             fontweight='bold', fontsize=14, y=1.01)
fig.tight_layout()

year_str = '%d' % args.year0 if args.year0 == args.year1 else '%d-%d' % (args.year0, args.year1)
ff_str = 'penn_cove_bottom_DO_' + year_str + '_' + gtx

print('Plotting ' + ff_str)
sys.stdout.flush()

if args.testing:
    plt.show()
else:
    plt.savefig(out_dir / (ff_str + '.png'), bbox_inches='tight')
    print('Saved to:\n %s' % (str(out_dir / (ff_str + '.png'))))

# ===== Figure 2: All stations on one panel =====
pfun.start_plot(figsize=(14, 6), fs=12)
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))

for stn in stations:
    stn_df = df[df['station'] == stn].sort_values('time')
    c = color_dict.get(stn, 'k')
    lbl = label_dict.get(stn, stn)

    ax2.plot(stn_df['time'], stn_df['obs_DO'], 'o-',
             color=c, markersize=4, label=lbl + ' obs')
    ax2.plot(stn_df['time'], stn_df['mod_DO'], 's--',
             color=c, markersize=4, alpha=0.6, markerfacecolor='none',
             label=lbl + ' model')

ax2.axhline(y=2.0, color='gray', linestyle=':', linewidth=1, label='hypoxic (2 mg/L)')
ax2.set_ylabel('DO (mg/L)')
ax2.set_xlabel('Date')
ax2.set_title('Penn Cove Bottom DO: obs vs %s' % gtx, fontweight='bold')
ax2.legend(loc='best', fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

ff_str2 = 'penn_cove_bottom_DO_combined_' + year_str + '_' + gtx

print('Plotting ' + ff_str2)
sys.stdout.flush()

if args.testing:
    plt.show()
else:
    plt.savefig(out_dir / (ff_str2 + '.png'), bbox_inches='tight')
    print('Saved to:\n %s' % (str(out_dir / (ff_str2 + '.png'))))
