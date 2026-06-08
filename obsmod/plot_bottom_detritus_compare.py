"""
Plot bottom detritus time series comparing two model versions.

Since detritus is not directly observed, this shows model0 vs model1 only,
sampled at the same cast locations used for obs/model validation.
Reads model cast NetCDF files produced by extract_casts_fast.py.

Available detritus variables: detritus (small), Ldetritus (large)

Testing on mac:
run plot_bottom_detritus_compare -gtx0 wb1_t0_xn11abbur00 -gtx1 wb1_t1_xn11abbur00 -year 2022 -otype ctd -test True
run plot_bottom_detritus_compare -gtx0 wb1_t0_xn11abbur00 -gtx1 wb1_t1_xn11abbur00 -year 2022 -otype ctd -var Ldetritus -test True
"""

import sys
import pandas as pd
import numpy as np
import pickle
import xarray as xr
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun, zfun

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-gtx0', type=str)
parser.add_argument('-gtx1', type=str)
parser.add_argument('-otype', type=str, default='ctd')
parser.add_argument('-year', type=int)
parser.add_argument('-var', type=str, default='detritus')  # detritus or Ldetritus
parser.add_argument('-stations', type=str, default='')
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
args = parser.parse_args()

Ldir = Lfun.Lstart()

if '_mac' in Ldir['lo_env']:
    pass
else:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

in_dir = Ldir['LOo'] / 'obsmod'
year = str(args.year)
gtx0 = args.gtx0
gtx1 = args.gtx1
otype = args.otype
varname = args.var

# Load combined pickle from gtx0 only for obs metadata (cid, station, z, source, time)
in_fn0 = in_dir / ('combined_' + otype + '_' + year + '_' + gtx0 + '.p')
if not in_fn0.is_file():
    print('Missing combined pickle for %s: %s' % (gtx0, in_fn0))
    sys.exit()

df_dict0 = pickle.load(open(in_fn0, 'rb'))
obs = df_dict0['obs']

meta = pd.DataFrame({
    'cid':     obs['cid'].values,
    'z':       obs['z'].values,
    'time':    pd.to_datetime(obs['time'].values),
    'source':  obs['source'].values if 'source' in obs.columns else '',
    'station': obs['name'].values if 'name' in obs.columns else (
               obs['lon'].round(3).astype(str) + '_' + obs['lat'].round(3).astype(str)),
})

# Keep only the deepest obs per cast (bottom)
bot_meta = meta.loc[meta.groupby('cid')['z'].idxmin()].copy()

if args.stations:
    station_filter = [s.strip() for s in args.stations.split(',')]
    bot_meta = bot_meta[bot_meta['station'].isin(station_filter)]
    if len(bot_meta) == 0:
        print('No data found for requested stations: %s' % args.stations)
        sys.exit()

# Read bottom detritus from extracted cast files for both model versions
records = []
n_missing = {gtx0: 0, gtx1: 0}

for _, row in bot_meta.iterrows():
    cid = int(row['cid'])
    source = row['source']
    obs_z = row['z']

    vals = {}
    for gtx in [gtx0, gtx1]:
        mod_dir = Ldir['LOo'] / 'extract' / gtx / 'cast' / (source + '_' + otype + '_' + year)
        cast_fn = mod_dir / (str(cid) + '.nc')
        val = np.nan
        if cast_fn.is_file():
            ds = xr.open_dataset(cast_fn)
            if varname in ds.data_vars:
                mz = ds.z_rho.values
                iz = zfun.find_nearest_ind(mz, obs_z)
                val = float(ds[varname][iz].values)
            else:
                n_missing[gtx] += 1
            ds.close()
        vals[gtx] = val

    records.append({
        'cid':      cid,
        'station':  row['station'],
        'time':     row['time'],
        'z':        obs_z,
        'mod0_val': vals[gtx0],
        'mod1_val': vals[gtx1],
    })

for gtx, n in n_missing.items():
    if n > 0:
        print('Warning: "%s" not found in %d cast files for %s' % (varname, n, gtx))

if len(records) == 0:
    print('No records found.')
    sys.exit()

df = pd.DataFrame(records)
df = df.dropna(subset=['mod0_val', 'mod1_val'])

if len(df) == 0:
    print('Variable "%s" not found in any cast files for both models. '
          'Check variable name (available: detritus, Ldetritus).' % varname)
    sys.exit()

df['diff'] = df['mod1_val'] - df['mod0_val']

stn_stats = df.groupby('station').agg(
    mean_mod0=('mod0_val', 'mean'),
    mean_mod1=('mod1_val', 'mean'),
    mean_diff=('diff', 'mean'),
    n_casts=('cid', 'count'),
    mean_z=('z', 'mean'),
).reset_index()
stn_stats = stn_stats[stn_stats['n_casts'] >= 2]
stn_stats = stn_stats.sort_values('mean_diff', ascending=False).reset_index(drop=True)

stations = stn_stats['station'].tolist()
n_stn = len(stations)

if n_stn == 0:
    print('No stations with >= 2 casts')
    sys.exit()

print('%d stations with >= 2 bottom %s casts' % (n_stn, varname))

out_dir = Ldir['LOo'] / 'obsmod_val_plots'
Lfun.make_dir(out_dir)

def short_gtx(gtx):
    return gtx.split('_')[-1] if '_' in gtx else gtx

lbl0 = short_gtx(gtx0)
lbl1 = short_gtx(gtx1)

n_per_page = 6
n_pages = int(np.ceil(n_stn / n_per_page))

for page in range(n_pages):
    i0 = page * n_per_page
    i1 = min(i0 + n_per_page, n_stn)
    page_stations = stations[i0:i1]
    n_sub = len(page_stations)

    pfun.start_plot(figsize=(14, 3 * n_sub), fs=11)
    fig, axes = plt.subplots(n_sub, 1, figsize=(14, 3 * n_sub), sharex=True)
    if n_sub == 1:
        axes = [axes]

    for j, stn in enumerate(page_stations):
        ax = axes[j]
        sdf = df[df['station'] == stn].sort_values('time')
        stats = stn_stats[stn_stats['station'] == stn].iloc[0]

        ax.plot(sdf['time'], sdf['mod0_val'], 's--', color='tab:red',
                markersize=5, markerfacecolor='none', label=lbl0)
        ax.plot(sdf['time'], sdf['mod1_val'], '^--', color='tab:blue',
                markersize=5, markerfacecolor='none', label=lbl1)

        ax.set_ylabel('%s (mmol N/m³)' % varname)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        ax.set_title(
            '%s  (n=%d, z=%.0f m | mean %s=%.3f  mean %s=%.3f  Δ=%.3f)' % (
                stn, int(stats['n_casts']), stats['mean_z'],
                lbl0, stats['mean_mod0'],
                lbl1, stats['mean_mod1'],
                stats['mean_diff']),
            fontweight='bold', fontsize=9, loc='left')

    axes[-1].set_xlabel('Date')
    fig.suptitle(
        'Bottom %s Comparison: %s %s\n%s (red) vs %s (blue)  '
        '[page %d/%d, sorted by Δ = %s−%s]' % (
            varname, otype, year, gtx0, gtx1,
            page + 1, n_pages, lbl1, lbl0),
        fontweight='bold', fontsize=12, y=1.01)
    fig.tight_layout()

    ff_str = 'bottom_%s_compare_%s_%s_%s_vs_%s_p%02d' % (
        varname, otype, year, gtx0, gtx1, page + 1)
    print('Plotting ' + ff_str)
    sys.stdout.flush()

    if args.testing:
        plt.show()
    else:
        plt.savefig(out_dir / (ff_str + '.png'), bbox_inches='tight')
        print('Saved to:\n %s' % (str(out_dir / (ff_str + '.png'))))

print('Done. %d stations across %d pages.' % (n_stn, n_pages))
